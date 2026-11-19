"""Neural building blocks; no model registry or external graph runtime."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .multimodal import PROPERTY_GROUPS


def masked_mean(x, mask):
    return (x*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)


def residue_distances(batch, structural):
    shared = batch.get("_trievidence_distances")
    if shared is not None:
        return shared
    coords = batch["coords"].float().masked_fill(~structural[..., None], 0)
    with torch.autocast(device_type=coords.device.type, enabled=False):
        return torch.cdist(coords, coords)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qnorm, self.knorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, keys, query_mask, key_mask, bias=None):
        # An empty modality contributes exactly zero; MHA must not see an all-masked row.
        present = key_mask.any(1)
        safe = key_mask.clone()
        safe[~present, 0] = True
        keys = keys.masked_fill(~key_mask[..., None], 0)
        attn_mask = None
        padding = ~safe
        if bias is not None:
            attn_mask = bias.clone()
            attn_mask = attn_mask.masked_fill(~safe[:, None, None, :], float("-inf"))
            attn_mask = attn_mask.reshape(-1, query.shape[1], keys.shape[1])
            padding = None
        value = self.attention(self.qnorm(query), self.knorm(keys), self.knorm(keys),
                               key_padding_mask=padding, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout(value)*query_mask[..., None]*present[:, None, None]


class FeedForward(nn.Module):
    def __init__(self, dim, expansion, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, expansion*dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(expansion*dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask):
        return (x+self.net(x))*mask[..., None]


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout, expansion=4):
        super().__init__()
        self.attention = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, expansion, dropout)

    def forward(self, x, mask):
        return self.ff(x+self.attention(x, x, mask, mask), mask)


class LocalGlobalBlock(nn.Module):
    def __init__(self, dim, heads, dropout, separable=False):
        super().__init__()
        # HelixSet separates spatial filtering from the shared pointwise mix.
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k//2, groups=dim if separable else 1)
            for k in (3, 5, 7)])
        self.norm = nn.LayerNorm(dim)
        self.mix = nn.Linear(3*dim, dim)
        self.global_block = TransformerBlock(dim, heads, dropout)

    def forward(self, x, mask):
        normalized = (self.norm(x)*mask[..., None]).transpose(1, 2)
        local = torch.cat([F.gelu(c(normalized)).transpose(1, 2) for c in self.convs], -1)
        x = (x+self.mix(local))*mask[..., None]
        return self.global_block(x, mask)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, vocab_size, dim, heads, dropout, depth=4, separable=False,
                 peptide_input_dim=0):
        super().__init__()
        self.plm = nn.Linear(input_dim + peptide_input_dim, dim)
        if peptide_input_dim:
            self.plm_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
            self.peptide_norm = nn.LayerNorm(peptide_input_dim, elementwise_affine=False)
        self.aa = nn.Embedding(21, dim, padding_idx=0)
        self.d = nn.Embedding(2, dim)
        self.chem = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Embedding(256, dim)
        self.input_gate = nn.Linear(2*dim, dim)
        self.layers = nn.ModuleList([LocalGlobalBlock(dim, heads, dropout, separable) for _ in range(depth)])
        self.global_layer = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def normalized_dual_esm(self, batch):
        return torch.cat([self.plm_norm(batch["esm"]), self.peptide_norm(batch["esm_peptide"])], -1)

    def forward(self, batch, return_shallow=False):
        mask = batch["mask"]
        chemical = self.chem(batch["residue_chem"])
        annotated = self.aa(batch["aa"])+self.d(batch["d"])+chemical
        features = batch["esm"]
        if hasattr(self, "peptide_norm"):
            features = batch.get("_trievidence_dual_esm")
            if features is None:
                features = self.normalized_dual_esm(batch)
        plm = self.plm(features)
        x = plm+torch.sigmoid(self.input_gate(torch.cat([plm, annotated], -1)))*annotated
        x = (x+self.position(torch.arange(x.shape[1], device=x.device)))*mask[..., None]
        shallow = x
        for layer in self.layers:
            x = layer(x, mask)
        global_chem = masked_mean(self.chem(batch["chemical"]), batch["chemical"]!=0)
        global_token = self.global_layer(torch.cat([masked_mean(x, mask), global_chem], -1))
        result = torch.cat([x, global_token[:, None]], 1), F.pad(mask, (0, 1), value=True)
        return (*result, shallow) if return_shallow else result


class GeometricLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.edge = nn.Sequential(nn.Linear(20, dim), nn.GELU(), nn.Linear(dim, heads))
        self.backbone = Attention(dim, heads, dropout)
        self.contact = Attention(dim, heads, dropout)
        self.global_read = Attention(dim, heads, dropout)
        self.global_write = Attention(dim, heads, dropout)
        self.film = nn.Linear(dim, 2*dim)
        self.ff = FeedForward(dim, 4, dropout)

    def forward(self, x, mask, batch, property_context=None, quality_aware=True):
        coordinates = batch["coords"]
        distance = torch.cdist(coordinates, coordinates)
        centers = torch.linspace(0, 16, 16, device=x.device)
        rbf = torch.exp(-((distance[..., None]-centers)/1.5).square())
        geometric_pair = batch["struct_mask"][:, :, None]&batch["struct_mask"][:, None, :]
        rbf = rbf*geometric_pair[..., None]
        length = x.shape[1]
        delta = torch.arange(length, device=x.device)[:, None]-torch.arange(length, device=x.device)[None]
        kind = batch["edge_type"]
        features = torch.cat([rbf, (delta.abs()/74)[None, ..., None].expand(x.shape[0], -1, -1, -1),
                              F.one_hot(kind, 3).float()], -1)
        bias = self.edge(features).permute(0, 3, 1, 2)
        eye = torch.eye(length, dtype=torch.bool, device=x.device)[None]
        known = batch["confidence_known"] & batch["struct_mask"]
        quality = torch.where(known, .3+.7*batch["geometry"][..., 9].clamp(0, 1), 1)
        if not quality_aware:
            quality = torch.ones_like(quality)
        reliability = (quality[:, None, :]*quality[:, :, None]).clamp_min(1e-5).log()
        messages = []
        for value, attention in [(1, self.backbone), (2, self.contact)]:
            allowed = (kind==value)|eye
            typed_bias = (bias+reliability[:, None]).masked_fill(~allowed[:, None], -1e4)
            messages.append(attention(x, x, mask, mask, typed_bias))
        message = sum(messages)
        if property_context is not None:
            gamma, beta = self.film(property_context).chunk(2, -1)
            message = message*(1+.2*gamma.tanh()[:, None])+.2*beta[:, None]*mask[..., None]
        x = (x+message)*mask[..., None]
        global_mask = mask.any(1, keepdim=True)
        global_token = masked_mean(x, mask)[:, None]
        global_token = global_token+self.global_read(global_token, x, global_mask, mask)
        x = x+self.global_write(x, global_token, mask, global_mask)
        return self.ff(x, mask)


class GeometryEncoder(nn.Module):
    def __init__(self, dim, heads, dropout, depth=4):
        super().__init__()
        self.aa = nn.Embedding(21, dim, padding_idx=0)
        self.d = nn.Embedding(2, dim)
        self.dssp = nn.Embedding(10, dim)
        self.geometry = nn.Linear(16, dim)
        self.layers = nn.ModuleList([GeometricLayer(dim, heads, dropout) for _ in range(depth)])
        self.skip = nn.Linear((depth+1)*dim, dim)
        self.global_features = nn.Linear(50, dim)

    def forward(self, batch, property_context=None, quality_aware=True):
        mask = batch["struct_mask"]
        x = (self.aa(batch["aa"])+self.d(batch["d"])+self.dssp(batch["dssp"])+self.geometry(batch["geometry"]))*mask[..., None]
        states = [x]
        for layer in self.layers:
            x = layer(x, mask, batch, property_context, quality_aware)
            states.append(x)
        x = self.skip(torch.cat(states, -1))*mask[..., None]
        global_features = self.global_features(torch.cat([batch["struct_global"], batch["struct_global_mask"].float()], -1))
        global_token = (masked_mean(x, mask)+global_features)*mask.any(1, keepdim=True)
        return torch.cat([x, global_token[:, None]], 1), torch.cat([mask, mask.any(1, keepdim=True)], 1)


class PropertyEncoder(nn.Module):
    def __init__(self, dim, heads, dropout, depth=2):
        super().__init__()
        self.groups = PROPERTY_GROUPS
        self.projections = nn.ModuleList([nn.Linear(2*len(g), dim) for g in self.groups])
        self.group_embedding = nn.Parameter(torch.randn(25, dim)*.02)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(depth)])

    def forward(self, batch):
        values, observed = batch["phys"], batch["phys_mask"]
        tokens = torch.stack([projection(torch.cat([values[:, g], observed[:, g].float()], -1))
                              for g, projection in zip(self.groups, self.projections)], 1)
        mask = torch.stack([observed[:, g].any(1) for g in self.groups], 1)
        if "property_token_mask" in batch:
            mask = batch["property_token_mask"]
        tokens = (tokens+self.group_embedding)*mask[..., None]
        for layer in self.layers:
            tokens = layer(tokens, mask)
        return tokens, mask


class MultimodalInteraction(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.links = nn.ModuleList([Attention(dim, heads, dropout) for _ in range(6)])
        self.ff = nn.ModuleList([FeedForward(dim, 4, dropout) for _ in range(3)])
        self.quality = nn.ModuleList([nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, 1)) for _ in range(3)])

    def forward(self, streams, masks, quality, quality_aware=True):
        output, link = [], 0
        for i in range(3):
            x = streams[i]
            for j in range(3):
                if i != j:
                    gate = torch.sigmoid(self.quality[j](quality[:, j]))[:, None] if quality_aware else 1
                    x = x+gate*self.links[link](streams[i], streams[j], masks[i], masks[j])
                    link += 1
            output.append(self.ff[i](x, masks[i]))
        return output


class EvidenceReader(nn.Module):
    def __init__(self, labels, dim, heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.randn(labels, dim)*.02)
        self.readers = nn.ModuleList([Attention(dim, heads, dropout) for _ in range(3)])
        self.quality = nn.Linear(6, dim)
        self.score = nn.Sequential(nn.Linear(3*dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.norm = nn.LayerNorm(dim)

    def read(self, streams, masks, quality, context=None, quality_aware=True):
        query = self.query[None].expand(streams[0].shape[0], -1, -1)
        if context is not None:
            query = query+context
        qmask = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        evidence = torch.stack([reader(query, stream, qmask, mask) for reader, stream, mask in zip(self.readers, streams, masks)], 2)
        qualities = self.quality(quality)[:, None].expand(-1, query.shape[1], -1, -1)
        if not quality_aware:
            qualities = torch.zeros_like(qualities)
        return query, evidence, qualities

    def fuse(self, query, evidence, qualities, masks):
        score = self.score(torch.cat([query[:, :, None].expand_as(evidence), evidence, qualities], -1)).squeeze(-1)
        present = torch.stack([mask.any(1) for mask in masks], 1)
        weights = score.masked_fill(~present[:, None], -1e4).softmax(-1)
        fused = self.norm(query+(evidence*weights[..., None]).sum(2))
        return fused, evidence, weights

    def forward(self, streams, masks, quality, context=None, quality_aware=True):
        query, evidence, qualities = self.read(streams, masks, quality, context, quality_aware)
        return self.fuse(query, evidence, qualities, masks)


class TailClassifier(nn.Module):
    def __init__(self, labels, dim, statistics, rank=32):
        super().__init__()
        self.shared = nn.Parameter(torch.randn(dim)*.02)
        self.basis = nn.Parameter(torch.randn(rank, dim)*.02)
        self.coefficients = nn.Parameter(torch.randn(labels, rank)*.02)
        self.residual = nn.Parameter(torch.randn(labels, dim)*.02)
        support = torch.tensor(statistics["positive_clusters"], dtype=torch.float32)
        self.register_buffer("shrink", support/(support+100))
        prior = torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1-1e-4)
        self.bias = nn.Parameter(torch.logit(prior))

    def forward(self, features):
        weight = self.shared+self.coefficients@self.basis+self.shrink[:, None]*self.residual
        return (features*weight).sum(-1)/math.sqrt(features.shape[-1])+self.bias


class LabelRelations(nn.Module):
    def __init__(self, labels, dim, heads, dropout, statistics, latent=8, depth=2):
        super().__init__()
        self.register_buffer("prior", torch.tensor(statistics["pair_prior"], dtype=torch.float32))
        empirical = torch.tensor(statistics["hyperedges"], dtype=torch.float32).reshape(-1, labels)
        self.register_buffer("incidence", empirical)
        clusters = torch.tensor(statistics["hyperedge_clusters"], dtype=torch.float32)
        self.register_buffer("factor_shrink", clusters/(clusters+20))
        self.latent_membership = nn.Parameter(torch.randn(latent, labels)*.02)
        self.pair = nn.ModuleList([Attention(dim, heads, dropout) for _ in range(depth)])
        self.factor = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim), nn.GELU()) for _ in range(depth)])
        self.ff = nn.ModuleList([FeedForward(dim, 4, dropout) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(2*dim, dim) for _ in range(depth)])
        self.heads = heads

    def forward(self, x, high_order=True, view_dropout=0.):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        prior = self.prior[None, None].expand(x.shape[0], self.heads, -1, -1)
        if view_dropout:
            # Mask learned pair attention as well as the empirical bias. Each
            # label keeps its self edge, including when all other edges drop.
            keep = torch.rand(x.shape[0], 1, x.shape[1], x.shape[1], device=x.device) >= view_dropout
            keep |= torch.eye(x.shape[1], dtype=torch.bool, device=x.device)[None, None]
            prior = prior.masked_fill(~keep, -1e4)
        for pair, factor, ff, gate in zip(self.pair, self.factor, self.ff, self.gates):
            message = pair(x, x, mask, mask, prior)
            if high_order:
                h = self.incidence/self.incidence.sum(-1, keepdim=True).clamp_min(1)
                membership = torch.cat([h, self.latent_membership.softmax(-1)], 0)
                factor_states = factor(torch.einsum("kc,bcd->bkd", membership, x))
                if view_dropout:
                    # Empirical and shared latent factors are both perturbed;
                    # surviving factors retain their original support shrinkage.
                    keep_factor = torch.rand(*factor_states.shape[:2], 1, device=x.device) >= view_dropout
                    factor_states = factor_states * keep_factor
                shrink = torch.cat([self.factor_shrink, torch.ones(len(self.latent_membership), device=x.device)])
                returns = membership*shrink[:, None]
                returns = returns/returns.sum(0, keepdim=True).clamp_min(1e-6)
                message = message+torch.einsum("kc,bkd->bcd", returns, factor_states)
            x = ff(x+torch.sigmoid(gate(torch.cat([x, message], -1)))*message, mask)
        return x


def training_views(batch, modality_dropout, mask_ratio):
    view = dict(batch)
    batch_size, length = batch["mask"].shape
    if modality_dropout:
        remove = torch.rand(batch_size, 2, device=batch["mask"].device)<modality_dropout
        # Sequence stays available: no empty evidence sets.
        view["struct_mask"] = batch["struct_mask"] & ~remove[:, 0, None]
        view["phys_mask"] = batch["phys_mask"] & ~remove[:, 1, None]
    residue_hide = (torch.rand(batch_size, length, device=batch["mask"].device)<mask_ratio)&view["struct_mask"]
    group_hide = torch.rand(batch_size, 25, device=batch["mask"].device)<mask_ratio
    group_hide[:, 20:22] = group_hide[:, 20:22].any(1, keepdim=True)
    group_hide[:, 22:24] = group_hide[:, 22:24].any(1, keepdim=True)
    phys_hide = torch.zeros_like(batch["phys_mask"])
    for j, group in enumerate(PROPERTY_GROUPS):
        phys_hide[:, group] = group_hide[:, j, None]
    phys_hide &= view["phys_mask"]
    view["property_token_mask"] = torch.stack([view["phys_mask"][:, group].any(1) for group in PROPERTY_GROUPS], 1)
    view["phys_mask"] = view["phys_mask"] & ~phys_hide
    view["phys"] = batch["phys"].masked_fill(~view["phys_mask"], 0)
    hide_geometry = residue_hide | ~view["struct_mask"]
    view["geometry"] = batch["geometry"].masked_fill(hide_geometry[..., None], 0)
    view["dssp"] = batch["dssp"].masked_fill(hide_geometry, 9)
    view["confidence_known"] = batch["confidence_known"] & ~hide_geometry
    pair_valid = view["struct_mask"][:, :, None]&view["struct_mask"][:, None]
    view["edge_type"] = batch["edge_type"].masked_fill(~pair_valid, 0)
    view["coords"] = batch["coords"].masked_fill(~view["struct_mask"][..., None], 0)
    view["struct_global_mask"] = batch["struct_global_mask"] & view["struct_mask"].any(1, keepdim=True) & ~residue_hide.any(1, keepdim=True)
    view["struct_global"] = batch["struct_global"].masked_fill(~view["struct_global_mask"], 0)
    # No coordinate/edge target is reconstructed. Geometry targets are derived from the
    # still-observed coordinate view; this is denoising, not independent structural evidence.
    view["quality"] = batch["quality"].clone()
    view["quality"][:, 1, 0] = view["struct_mask"].any(1)
    view["quality"][:, 1] *= view["struct_mask"].any(1, keepdim=True)
    view["quality"][:, 2, 0] = view["phys_mask"].any(1)
    view["quality"][:, 2, 1] = view["phys_mask"].float().mean(1)
    return view, {"residue": residue_hide, "phys": phys_hide}


class Reconstruction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phys = nn.ModuleList([nn.Linear(dim, len(group)) for group in PROPERTY_GROUPS])
        self.geometry = nn.Linear(dim, 8)
        self.dssp = nn.Linear(dim, 9)

    def forward(self, sequence, properties):
        return {"phys": [head(properties[:, j]) for j, head in enumerate(self.phys)],
                "geometry": self.geometry(sequence), "dssp": self.dssp(sequence)}


def reconstruction_loss(output, original, hidden):
    # Macro-average descriptor groups, not 515 scalar terms dominated by DPC.
    total = output["geometry"].sum()*0
    groups = []
    for prediction, indices in zip(output["phys"], PROPERTY_GROUPS):
        mask = hidden["phys"][:, indices]
        if mask.any():
            groups.append(((prediction-original["phys"][:, indices]).square()*mask).sum()/mask.sum())
    if groups:
        total = total+torch.stack(groups).mean()
    target = torch.cat([original["geometry"][..., :6], original["geometry"][..., 10:12]], -1)
    available = torch.cat([original["geometry"][..., 6:9].bool().repeat_interleave(2, -1), original["geometry"][..., 13:15].bool()], -1)
    mask = hidden["residue"][..., None]&available
    total = total+((output["geometry"]-target).square()*mask).sum()/mask.sum().clamp_min(1)
    dssp_mask = hidden["residue"]&(original["dssp"]<9)
    if dssp_mask.any():
        total = total+F.cross_entropy(output["dssp"][dssp_mask], original["dssp"][dssp_mask])
    return total


def unlabeled_weights(targets, statistics, unknown_weight=.3):
    """Shared missing-label semantics for pointwise and annotation-set losses."""
    weak_weight = torch.full_like(targets, unknown_weight)
    if "therapep_background_index" in statistics:
        flag = statistics["therapep_background_index"]
        # Published decoys have assigned zero targets, not missing annotations.
        # Their validity as biological negatives remains an explicit assumption.
        weak_weight = torch.where(targets[:, flag:flag+1] == 0, 1., weak_weight)
        weak_weight[:, flag] = 1.
    labels = statistics["labels"]
    if any(label.endswith("__unknown") for label in labels):
        from .multimodal import function_projection
        projection = torch.tensor(function_projection(labels, "fine39"), device=targets.device)
        unknown = torch.tensor([label.endswith("__unknown") for label in labels], device=targets.device)
        unknown_parent = (targets*unknown)@projection
        ambiguous = (unknown_parent@projection.T)>0
        weak_weight = torch.where(ambiguous & ~unknown, weak_weight*.25, weak_weight)
    return weak_weight


def observed_label_loss(logits, targets, statistics, unknown_weight=.3, mask=None, positive_multiplier=None):
    """ASL with explicitly weak unlabeled terms; no confirmed-negative claim."""
    logits = logits.float()
    positive_weight = torch.as_tensor(statistics["positive_weights"], device=logits.device)
    p = logits.sigmoid()
    negative = (1-p+.03).clamp(max=1)
    pos = -F.logsigmoid(logits)*targets*positive_weight
    if positive_multiplier is not None:
        pos = pos * positive_multiplier
    weak_weight = unlabeled_weights(targets, statistics, unknown_weight)
    neg = -negative.clamp_min(1e-8).log()*(1-negative).square().detach()*(1-targets)*weak_weight
    values = pos+neg
    return (values*mask).sum()/mask.sum().clamp_min(1) if mask is not None else values.mean()
