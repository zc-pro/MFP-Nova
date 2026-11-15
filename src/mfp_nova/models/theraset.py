"""TheraSet: a TheraPep-style CNN with prototype-conditioned evidence corrections.

Architecture and residue-property provenance: docs/TheraSet-Architecture.md.
This is an independent implementation adapted to the project's label protocol,
not the published TheraPep-AI ensemble or its training/data reproduction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..research_layers import (Attention, FeedForward, PropertyEncoder, SequenceEncoder,
                              TransformerBlock, masked_mean, observed_label_loss, training_views)
from .helixset import LabelPairRead, QualityRead, structure_strength


def residue_encoding():
    """20 one-hot + 14 fixed properties, in project ACDEFGHIKLMNPQRSTVWY order.

    Numeric biochemical scales/flags follow TheraPep-AI's published data.py;
    normalization is over the 20 residue types, never fitted on Val/Test.
    Row zero is padding. Propensities are lookup values, not measured structure.
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    charge = [0, 0, -1, -1, 0, 0, .1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    hydro = [1.8, 2.5, -3.5, -3.5, 2.8, -.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -.8, -.7, 4.2, -.9, -1.3]
    volume = [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89, 116.1, 140, 227.8, 193.6]
    size = [0, .3, .5, .6, 1, 0, .7, .7, 1, .7, .7, .5, .4, .6, 1, 0, .4, .6, 1, 1]
    helix = [1.42, .70, 1.01, 1.51, 1.13, .57, 1, 1.08, 1.16, 1.21, 1.45, .67, .57, 1.11, .98, .77, .83, 1.06, 1.08, .69]
    sheet = [.83, 1.19, .54, .37, 1.38, .75, .87, 1.6, .74, 1.3, 1.05, .89, .55, 1.1, .93, .75, 1.19, 1.7, 1.37, 1.47]
    turn = [.66, 1.19, 1.46, .74, .60, 1.56, .95, .47, 1.01, .59, .60, 1.56, 1.52, .98, .95, 1.43, .96, .50, .96, 1.14]
    def flag(residues):
        return [float(a in residues) for a in alphabet]
    def unit(values):
        x = torch.tensor(values, dtype=torch.float32)
        return ((x-x.min())/(x.max()-x.min())).tolist()
    properties = torch.tensor([
        charge, unit(hydro), flag("DEHKNQRSTY"), flag("FWY"), flag("HKNQRSTWY"),
        flag("DEHNQSTY"), unit(volume), flag("G"), flag("P"), flag("C"), size,
        unit(helix), unit(sheet), unit(turn),
    ]).T
    return F.pad(torch.cat([torch.eye(20), properties], -1), (0, 0, 1, 0))


class MotifCNN(nn.Module):
    """Two parallel 3/5/7/9 CNN stages; retain stage-two maps before max pooling."""

    def __init__(self, vocab_size, channels, dropout, chemical_evidence, short_only=False):
        super().__init__()
        self.chemical_evidence = chemical_evidence
        kernels = (3,) if short_only else (3, 5, 7, 9)
        self.output_channels = len(kernels) * channels
        self.register_buffer("encoding", residue_encoding())
        self.input = nn.Linear(34, 64)
        self.d = nn.Embedding(2, 64)
        self.chem = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs1 = nn.ModuleList([nn.Conv1d(64, channels, k, padding=k//2) for k in kernels])
        self.convs2 = nn.ModuleList([] if short_only else
                                    [nn.Conv1d(self.output_channels, channels, k, padding=k//2) for k in kernels])
        # Per-position normalization avoids dependence on padded batch lengths.
        self.norms1 = nn.ModuleList([nn.LayerNorm(channels) for _ in self.convs1])
        self.norms2 = nn.ModuleList([nn.LayerNorm(channels) for _ in self.convs2])
        self.dropout = nn.Dropout(dropout)

    def residue_inputs(self, batch):
        x = self.input(self.encoding[batch["aa"]])
        if self.chemical_evidence:
            global_chem = masked_mean(self.chem(batch["chemical"]), batch["chemical"] != 0)
            x = x + self.d(batch["d"]) + self.chem(batch["residue_chem"]) + global_chem[:, None]
        return x.masked_fill(~batch["mask"][..., None], 0)

    def residue_maps(self, batch, inputs=None):
        mask = batch["mask"]
        x = (self.residue_inputs(batch) if inputs is None else inputs).transpose(1, 2)
        first = [F.gelu(norm(conv(x).transpose(1, 2))).masked_fill(~mask[..., None], 0)
                 for conv, norm in zip(self.convs1, self.norms1)]
        if not self.convs2:
            return [self.dropout(first[0])]
        x = self.dropout(torch.cat(first, -1)).transpose(1, 2)
        return [self.dropout(F.gelu(norm(conv(x).transpose(1, 2)))).masked_fill(~mask[..., None], 0)
                for conv, norm in zip(self.convs2, self.norms2)]

    def forward(self, batch):
        maps = self.residue_maps(batch)
        mask = batch["mask"]
        pooled = torch.cat([h.masked_fill(~mask[..., None], torch.finfo(h.dtype).min).amax(1) for h in maps], -1)
        return maps, pooled


class FactorizedPairs(nn.Module):
    """Endpoint products plus separate measured geometry, without cubic triangles."""

    def __init__(self, dim, pair_dim):
        super().__init__()
        self.left, self.right = nn.Linear(dim, pair_dim), nn.Linear(dim, pair_dim)
        self.separation = nn.Linear(2, pair_dim)
        self.geometry = nn.Linear(22, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, residues, batch, structural, reliability):
        mask = batch["mask"]
        index = torch.arange(mask.shape[1], device=mask.device)
        offset = (index[:, None]-index[None, :]).float()/74
        valid = mask[:, :, None] & mask[:, None, :]
        pairs = self.norm(self.left(residues)[:, :, None]*self.right(residues)[:, None, :]
                          + self.separation(torch.stack([offset, offset.abs()], -1))[None])
        pairs = pairs * valid[..., None]
        observed = structural[:, :, None] & structural[:, None, :]
        observed = observed & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        coords = batch["coords"].float().masked_fill(~structural[..., None], 0)
        with torch.autocast(device_type=coords.device.type, enabled=False):
            distance = torch.cdist(coords, coords)
            centers = torch.linspace(0, 30, 16, device=coords.device)
            rbf = torch.exp(-((distance[..., None]-centers)/2).square())
        kind = batch["edge_type"].masked_fill(~observed, 0)
        known = batch["confidence_known"] & structural
        confidence = batch["geometry"][..., 9].masked_fill(~known, 0).clamp(0, 1)
        features = torch.cat([rbf, F.one_hot(kind, 3).float(), observed[..., None].float(),
                              (known[:, :, None] & known[:, None, :])[..., None].float(),
                              (confidence[:, :, None]*confidence[:, None, :])[..., None]], -1)
        geometric = self.geometry(features) * observed[..., None]
        quality = reliability[:, :, None]*reliability[:, None, :]*observed
        return pairs, geometric, quality


class EvidenceDecoder(nn.Module):
    """Separate modality reads, label-specific scale selection and small factor context."""

    def __init__(self, dim, heads, pair_dim, dropout, factor_count, prototype_count):
        super().__init__()
        self.prototype_count = prototype_count
        self.motif = Attention(dim, heads, dropout)
        self.scale_score = nn.Sequential(nn.Linear(2*dim, dim//2), nn.GELU(), nn.Linear(dim//2, 1))
        self.context = Attention(dim, heads, dropout)
        self.properties = Attention(dim, heads, dropout)
        self.structure = QualityRead(dim, heads, dropout)
        self.pair = LabelPairRead(dim, pair_dim, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(5)])
        self.source_score = nn.Sequential(nn.Linear(2*dim, dim//2), nn.GELU(), nn.Linear(dim//2, 1))
        self.ff = FeedForward(dim, 2, dropout)
        self.factors = nn.Parameter(torch.randn(factor_count, dim)*.02)
        self.factor_read, self.factor_write = Attention(dim, heads, dropout), Attention(dim, heads, dropout)
        self.factor_scale = nn.Parameter(torch.tensor(.1))

    def forward(self, query, query_mask, motifs, context, context_mask, properties, property_mask,
                geometry, geometry_mask, reliability, residues, pairs, geometric_pairs, pair_quality,
                mask, source_present, pair_readout, label_context):
        scales = torch.stack([self.motif(query, h, query_mask, mask) for h in motifs], 2)
        scale_weights = self.scale_score(torch.cat([query[:, :, None].expand_as(scales), scales], -1)).squeeze(-1).softmax(-1)
        local = (scales*scale_weights[..., None]).sum(2)
        context_read = self.context(query, context, query_mask, context_mask)
        prop_read = self.properties(query, properties, query_mask, property_mask)
        geom_read = self.structure(query, geometry, query_mask, geometry_mask, reliability)
        pair_read, pair_attention = torch.zeros_like(query), None
        if pair_readout:
            pair_read, pair_attention = self.pair(query, residues, pairs, mask, geometric_pairs, pair_quality)
        values = [local, context_read, prop_read, geom_read, pair_read]
        values = torch.stack([norm(value) for norm, value in zip(self.norms, values)], 2)
        # Quality is applied after LN; a uniform reduction must not cancel in softmax.
        strength = structure_strength(reliability, geometry_mask)
        gain = torch.ones_like(source_present, dtype=values.dtype)
        gain[:, 3] = strength[:, 0, 0]
        values = values * gain[:, None, :, None] * source_present[:, None, :, None]
        score = self.source_score(torch.cat([query[:, :, None].expand_as(values), values], -1)).squeeze(-1)
        routing = score.masked_fill(~source_present[:, None], float("-inf")).softmax(-1)
        query = self.ff(query + (routing[..., None]*values).sum(2), query_mask)
        if label_context:
            factors = self.factors[None].expand(query.shape[0], -1, -1)
            factor_mask = torch.ones(factors.shape[:2], dtype=torch.bool, device=mask.device)
            # One state per label: labels with more prototypes must not occupy
            # more attention mass solely because they have more queries.
            label_states = query.reshape(query.shape[0], -1, self.prototype_count, query.shape[-1])
            active = query_mask.reshape(query.shape[0], -1, self.prototype_count)
            label_states = (label_states*active[..., None]).sum(2)/active.sum(2).clamp_min(1)[..., None]
            factors = factors + self.factor_read(factors, label_states, factor_mask, active.any(2))
            query = query + self.factor_scale.tanh()*self.factor_write(query, factors, query_mask, factor_mask)
        return query*query_mask[..., None], scale_weights, routing, pair_attention


class TheraSet(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256, heads=8,
                 dropout=.2, cnn_channels=128, context_depth=1, property_depth=1,
                 pair_dim=32, decoder_depth=2, factor_count=8, prototype_count=4,
                 prototype_bank=32, peptide_input_dim=0, evidence_decoder=True,
                 contextual_evidence=True, property_evidence=True, structure_evidence=True,
                 chemical_evidence=True, pair_readout=True, prototype_readout=True,
                 label_context=True, b0_backbone=False, b0_local_max=False, b0_local_cnn=False):
        super().__init__()
        self.statistics = statistics
        self.evidence_decoder = evidence_decoder
        self.contextual_evidence, self.property_evidence = contextual_evidence, property_evidence
        self.structure_evidence, self.chemical_evidence = structure_evidence, chemical_evidence
        self.pair_readout, self.prototype_readout, self.label_context = pair_readout, prototype_readout, label_context
        prior = torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1-1e-4).logit()
        self.b0_backbone = None
        if b0_backbone:
            if evidence_decoder or chemical_evidence:
                raise ValueError("The B0 backbone comparison requires the pure AA34 ablation")
            from .hypothesis_backbone import SequenceBackbone
            cnn = MotifCNN(1, cnn_channels, dropout, chemical_evidence=False) if b0_local_cnn else None
            self.b0_backbone = SequenceBackbone(num_labels, dropout=dropout,
                                                residue_features=residue_encoding() if cnn is None else None,
                                                local_max_pool=b0_local_max or b0_local_cnn, motif_cnn=cnn)
            with torch.no_grad():
                self.b0_backbone.label_bias.copy_(prior)
            return
        self.cnn = MotifCNN(vocab_size, cnn_channels, dropout, chemical_evidence)
        self.direct = nn.Sequential(nn.Linear(4*cnn_channels, 1024), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, num_labels))
        with torch.no_grad():
            self.direct[-1].bias.copy_(prior)
        if not evidence_decoder:
            return
        self.motif_projections = nn.ModuleList([nn.Linear(cnn_channels, dim) for _ in range(4)])
        self.global_projection = nn.Linear(4*cnn_channels, dim)
        self.sequence = SequenceEncoder(input_dim, vocab_size, dim, heads, dropout, depth=0,
                                        peptide_input_dim=peptide_input_dim)
        self.context_layers = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(context_depth)])
        self.properties = PropertyEncoder(dim, heads, dropout, property_depth)
        self.geometry = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.LayerNorm(dim))
        self.dssp = nn.Embedding(10, dim)
        self.global_geometry = nn.Linear(50, dim)
        self.pairs = FactorizedPairs(dim, pair_dim)
        self.residue_norm = nn.LayerNorm(dim)
        self.label_queries = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.prototype_bank = nn.Parameter(torch.randn(prototype_bank, dim)*.1)
        self.prototype_coefficients = nn.Parameter(torch.randn(num_labels, prototype_count, prototype_bank)*.2)
        counts = torch.tensor(statistics["prototype_counts"]).clamp(1, prototype_count)
        if not prototype_readout:
            counts = torch.ones_like(counts)
        self.register_buffer("active", torch.arange(prototype_count)[None] < counts[:, None])
        self.decoder = nn.ModuleList([EvidenceDecoder(dim, heads, pair_dim, dropout, factor_count, prototype_count)
                                      for _ in range(decoder_depth)])
        self.output_norm = nn.LayerNorm(dim)
        self.label_weights = nn.Parameter(torch.randn(num_labels, dim)/math.sqrt(dim))
        self.register_buffer("prior", prior)
        self.correction_gate = nn.Linear(2*dim, 1)
        nn.init.zeros_(self.correction_gate.weight)
        nn.init.constant_(self.correction_gate.bias, -2.)

    def forward(self, batch, return_evidence=False, return_residues=False, return_states=False):
        if self.b0_backbone is not None:
            logits = self.b0_backbone(batch["aa"], batch["mask"]).logits
            return {"logits": logits, "direct": logits}
        maps, pooled = self.cnn(batch)
        direct = self.direct(pooled)
        if not self.evidence_decoder:
            return {"logits": direct, "direct": direct}
        mask = batch["mask"]
        b, length = mask.shape
        motifs = [projection(h)*mask[..., None] for projection, h in zip(self.motif_projections, maps)]
        context_batch = batch
        if not self.chemical_evidence:
            context_batch = dict(batch, d=torch.zeros_like(batch["d"]),
                                 residue_chem=torch.zeros_like(batch["residue_chem"]),
                                 chemical=torch.zeros_like(batch["chemical"]))
        if self.contextual_evidence:
            context, context_mask = self.sequence(context_batch)
            for layer in self.context_layers:
                context = layer(context, context_mask)
        else:
            context = torch.zeros_like(motifs[0])
            context_mask = torch.zeros_like(mask)
        prop_batch = dict(batch, phys=batch["phys"].masked_fill(~batch["phys_mask"], 0))
        properties, property_mask = self.properties(prop_batch)
        if not self.property_evidence:
            properties, property_mask = torch.zeros_like(properties), torch.zeros_like(property_mask)
        structural = batch["struct_mask"] & mask & self.structure_evidence
        known = batch["confidence_known"] & structural
        reliability = torch.where(known, batch["geometry"][..., 9].clamp(0, 1), 1.)*structural
        geometry = (self.geometry(batch["geometry"].masked_fill(~structural[..., None], 0))
                    + self.dssp(batch["dssp"].masked_fill(~structural, 0))) * structural[..., None]
        global_mask = batch["struct_global_mask"] & structural.any(1, keepdim=True)
        global_geometry = self.global_geometry(torch.cat([
            batch["struct_global"].masked_fill(~global_mask, 0), global_mask.float()], -1))
        geometry = torch.cat([geometry, global_geometry[:, None]], 1)
        geometry_mask = torch.cat([structural, global_mask.any(1, keepdim=True)], 1)
        mean_quality = reliability.sum(1, keepdim=True)/structural.sum(1, keepdim=True).clamp_min(1)
        reliability = torch.cat([reliability, mean_quality], 1)
        # Geometry stays separate from residues, preserving a clean sequence route.
        residues = self.residue_norm(sum(motifs)/4 + context[:, :length])*mask[..., None]
        pairs, geometric_pairs, pair_quality = self.pairs(residues, batch, structural, reliability[:, :length])
        prototype = self.prototype_coefficients.softmax(-1) @ self.prototype_bank
        queries = self.label_queries[:, None] + prototype
        global_state = self.global_projection(pooled)
        query = (queries[None] + global_state[:, None, None]).flatten(1, 2)
        query_mask = self.active.flatten()[None].expand(b, -1)
        query = query*query_mask[..., None]
        source_present = torch.stack([mask.any(1), context_mask.any(1), property_mask.any(1),
                                      (reliability*geometry_mask).sum(1)>0,
                                      (mask.sum(1)>1) & self.pair_readout], -1)
        for block in self.decoder:
            query, scales, routing, pair_attention = block(
                query, query_mask, motifs, context, context_mask, properties, property_mask,
                geometry, geometry_mask, reliability, residues, pairs, geometric_pairs, pair_quality,
                mask, source_present, self.pair_readout, self.label_context)
        states = self.output_norm(query).reshape(b, *self.active.shape, -1)
        scores = (states*self.label_weights[None, :, None]).sum(-1)
        # Normalized log-mean-exp: adding identical active prototypes cannot raise logits.
        scores = scores.float().masked_fill(~self.active[None], float("-inf"))
        prototype_weights = scores.softmax(-1)
        delta = scores.logsumexp(-1)-self.active.sum(-1).float().log()
        evidence_state = (states*prototype_weights[..., None]).sum(2)
        gate = self.correction_gate(torch.cat([
            evidence_state, global_state[:, None].expand_as(evidence_state)], -1)).squeeze(-1).sigmoid()
        output = {"logits": direct + gate*delta, "direct": direct,
                  "evidence_logits": self.prior+delta}
        if return_residues or return_states:
            output["residues"] = residues
        if return_states:
            output.update(evidence_state=evidence_state, global_state=global_state)
        if return_evidence:
            output.update(correction_gate=gate, correction=gate*delta, prototype_weights=prototype_weights,
                          scale_weights=scales.reshape(b, *self.active.shape, 4),
                          source_weights=routing.reshape(b, *self.active.shape, 5))
            output["pair_attention"] = (None if pair_attention is None else
                (pair_attention.reshape(b, *self.active.shape, length, length)
                 *prototype_weights[..., None, None]).sum(2))
        return output

    def supervised_terms(self, output, targets, unknown_weight):
        terms = {"classification": observed_label_loss(output["logits"], targets, self.statistics, unknown_weight)}
        if self.evidence_decoder:
            for key, logits in (("direct", output["direct"]), ("evidence", output["evidence_logits"])):
                terms[key] = observed_label_loss(logits, targets, self.statistics, unknown_weight)
        return terms

    def compute_loss(self, batch, settings, return_output=False):
        view = training_views(batch, settings.get("modality_dropout", .1), 0.)[0] if self.training else batch
        output = self(view)
        def supervised(prediction):
            return self.supervised_terms(prediction, batch["y"], settings.get("unlabeled_weight", .3))
        terms = supervised(output)
        agreement = output["logits"].sum()*0
        if self.training and settings.get("rdrop", 0)>0:
            from ..consistency import bernoulli_rdrop
            second = self(view)
            other = supervised(second)
            terms = {key: .5*(value+other[key]) for key, value in terms.items()}
            agreement = bernoulli_rdrop(output["logits"], second["logits"])
        total = terms["classification"]
        for key, default in (("direct", .2), ("evidence", .1)):
            if key in terms:
                total = total + settings.get(key, default)*terms[key]
        terms["rdrop"] = agreement
        total = total + settings.get("rdrop", 0)*agreement
        return (total, terms, output) if return_output else (total, terms)
