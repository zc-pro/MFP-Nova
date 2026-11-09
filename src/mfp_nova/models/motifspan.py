"""MotifSpan: ordered soft regions with conserved, implicit relation statistics."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import FrozenFeatures, ResidueCNN, residue_encoding, prior_bias, supervised_loss
from ..research_layers import masked_mean


def soft_assignment(velocity, mask, max_spans=8, span_size=6):
    """Stable log memberships; inactive regions never enter an empty softmax."""
    velocity = velocity.float()*mask
    n = mask.sum(1)
    count = ((n+span_size-1)//span_size).clamp(1, max_spans)
    z = (velocity.cumsum(1)-velocity/2)/velocity.sum(1, keepdim=True)
    k = torch.arange(max_spans, device=mask.device)[None, None]
    total = count[:, None, None]
    lower = (k-total*z[..., None])/.15
    upper = lower+1/.15
    log_a = F.logsigmoid(upper)+F.logsigmoid(-lower)+torch.log(-torch.expm1(lower-upper))
    log_a = torch.where(k == 0, F.logsigmoid(upper), log_a)
    log_a = torch.where(k == total-1, F.logsigmoid(-lower), log_a)
    log_a = torch.where(total == 1, torch.zeros_like(log_a), log_a)
    active = torch.arange(max_spans, device=mask.device)[None] < count[:, None]
    log_a = log_a.masked_fill(~(mask[..., None] & active[:, None]), -torch.inf)
    return log_a.exp(), log_a, active, count


def relation_statistics(assignment, left, right, rhos=(.5, .9, 1.)):
    """Prefix recurrence and contraction, excluding i=j without dense residue pairs."""
    dtype = left.dtype  # Tests can use double precision; caller uses FP32.
    assignment = assignment.to(dtype)
    b, length, k = assignment.shape
    rho = left.new_tensor(rhos)[None, :, None, None]
    prefix = left.new_zeros(b, len(rhos), k, left.shape[-1])
    mass_prefix = left.new_zeros(b, len(rhos), k)
    prefixes, masses = [], []
    for j in range(length):
        prefixes.append(prefix)
        masses.append(mass_prefix)
        prefix = rho*prefix+assignment[:, j, None, :, None]*left[:, j, None, None]
        mass_prefix = rho[..., 0]*mass_prefix+assignment[:, j, None]
    # The first stored prefix is zero; the preceding write is not decayed at j.
    states = torch.stack(prefixes, 1)
    mass_states = torch.stack(masses, 1)
    values = torch.einsum("bjskr,bjr,bjl->bsklr", states, right, assignment)
    mass = torch.einsum("bjsk,bjl->bskl", mass_states, assignment)
    return values, mass


class MotifSpan(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256, dropout=.2,
                 peptide_input_dim=0, cnn_channels=64, relation_dim=32, max_spans=8,
                 span_size=6, learned_spans=True, relation_mode="conserved", spatial=True,
                 detail_mode="conditional", global_only=False):
        super().__init__()
        if relation_mode not in ("conserved", "mean_product") or detail_mode not in ("conditional", "ordinary", "none"):
            raise ValueError("Unknown MotifSpan relation/detail mode")
        self.statistics, self.dim, self.rank = statistics, dim, relation_dim
        self.max_spans, self.span_size = max_spans, span_size
        self.learned_spans, self.relation_mode = learned_spans, relation_mode
        self.spatial, self.detail_mode = spatial, detail_mode
        self.global_only = global_only
        self.cnn = ResidueCNN(dropout, cnn_channels)
        self.features = FrozenFeatures(input_dim, dim, peptide_input_dim)
        self.cnn_projection = nn.Linear(4*cnn_channels, dim)
        self.fusion = nn.LayerNorm(dim)
        self.global_projection = nn.Sequential(nn.Linear(dim+4*cnn_channels, dim), nn.LayerNorm(dim))
        self.global_head = nn.Linear(dim, num_labels)
        with torch.no_grad():
            self.global_head.bias.copy_(prior_bias(statistics))
        if self.global_only:
            return
        self.velocity_conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.velocity_out = nn.Linear(dim, 1)
        nn.init.zeros_(self.velocity_out.weight)
        nn.init.zeros_(self.velocity_out.bias)
        self.content = nn.Linear(dim, 1)
        self.span_projection = nn.Sequential(nn.Linear(2*dim+2, dim), nn.LayerNorm(dim))
        self.query = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.span_key, self.span_value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.local_null = nn.Parameter(torch.zeros(num_labels))
        self.local_read = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.register_buffer("encoding", residue_encoding())
        self.local_input = nn.Linear(34, 64)
        self.local_norm1, self.local_norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.local_conv = nn.Conv1d(64, 64, 3, padding=1, bias=False)
        self.left, self.right = nn.Linear(64, relation_dim, bias=False), nn.Linear(64, relation_dim, bias=False)
        self.total_read = nn.Parameter(torch.randn(num_labels, 3, relation_dim)*.02)
        self.region_left, self.region_right = nn.Linear(dim, relation_dim, bias=False), nn.Linear(dim, relation_dim, bias=False)
        self.relation_key = nn.Linear(relation_dim, relation_dim, bias=False)
        self.position_key = nn.Linear(4, relation_dim, bias=False)
        self.distance_key = nn.Parameter(torch.randn(3, relation_dim)*.02)
        self.spatial_query = nn.Parameter(torch.randn(num_labels, relation_dim)*.02)
        self.spatial_read = nn.Parameter(torch.randn(num_labels, relation_dim)*.02)
        self.spatial_null = nn.Parameter(torch.zeros(num_labels))
        self.detail_query = nn.Linear(dim, 128, bias=False)
        self.detail_relation = nn.Linear(relation_dim, 128, bias=False)
        self.detail_key, self.detail_value = nn.Linear(dim, 128, bias=False), nn.Linear(dim, 128, bias=False)
        self.detail_read = nn.Parameter(torch.randn(num_labels, 128)*.02)
        self.eta = nn.Parameter(torch.full((4,), .1))

    def regions(self, h, mask):
        hidden = F.gelu(self.velocity_conv(h.transpose(1, 2)).transpose(1, 2)).masked_fill(~mask[..., None], 0)
        speed = 1+.5*torch.tanh(self.velocity_out(hidden).squeeze(-1))
        if not self.learned_spans:
            speed = torch.ones_like(speed)
        a, log_a, active, count = soft_assignment(speed, mask, self.max_spans, self.span_size)
        safe = torch.where(active[:, None], log_a, torch.zeros_like(log_a)).masked_fill(~mask[..., None], -torch.inf)
        mean_weights = safe.softmax(1)*active[:, None]
        content = 2*torch.tanh(self.content(h))
        content_weights = (safe+content).softmax(1)*active[:, None]
        mean = torch.einsum("blk,bld->bkd", mean_weights, h)
        peak = torch.einsum("blk,bld->bkd", content_weights, h)
        n = mask.sum(1, keepdim=True)
        pos = torch.arange(1, h.shape[1]+1, device=h.device)[None]/n
        center = torch.einsum("blk,bl->bk", mean_weights, pos)
        width = a.sum(1)/n
        shape = torch.stack([center, width], -1)
        regions = self.span_projection(torch.cat([mean, peak, shape], -1))*active[..., None]
        return regions, a, active, count, shape, mean_weights

    def relations(self, batch, regions, assignment, shape, mean_weights):
        mask = batch["mask"][..., None]
        local = F.gelu(self.local_norm1(self.local_input(self.encoding[batch["aa"]]))).masked_fill(~mask, 0)
        local = F.gelu(self.local_norm2(self.local_conv(local.transpose(1, 2)).transpose(1, 2))).masked_fill(~mask, 0)
        left, right = torch.tanh(self.left(local)), torch.tanh(self.right(local))
        t, mass = relation_statistics(assignment, left, right)
        if self.relation_mode == "mean_product":
            # Historical k<l reconstruction, on the same local a/b inputs.
            l = torch.einsum("blk,blr->bkr", mean_weights, left)
            r = torch.einsum("blk,blr->bkr", mean_weights, right)
            upper = torch.ones(regions.shape[1], regions.shape[1], device=regions.device).triu(1)
            mass = mass*upper
            t = mass[..., None]*(l[:, None, :, None]*r[:, None, None, :])
        z = mass.sum((-1, -2)).clamp_min(1e-12)
        average = t.sum((-3, -2))/z[..., None]
        total_delta = torch.einsum("bsr,csr->bc", average, self.total_read)/math.sqrt(self.rank)
        position = torch.cat([shape[:, :, None].expand(-1, -1, regions.shape[1], -1),
                              shape[:, None].expand(-1, regions.shape[1], -1, -1)], -1)
        # Reorder [mu_k,width_k,mu_l,width_l] to the documented geometry order.
        position = position[..., [0, 2, 1, 3]]
        keys = (self.region_left(regions)[:, None, :, None]+self.region_right(regions)[:, None, None, :]
                + self.relation_key(t/z[:, :, None, None, None])+self.position_key(position)[:, None]
                + self.distance_key[None, :, None, None])
        score = 4*torch.tanh(torch.einsum("bsklr,cr->bcskl", keys, self.spatial_query)/(4*math.sqrt(self.rank)))
        prior = mass/(3*z[:, :, None, None])
        maximum = score.masked_fill(prior[:, None] == 0, -torch.inf).flatten(2).amax(-1)
        shift = torch.maximum(maximum, self.spatial_null[None])
        exponential = (score-shift[..., None, None, None]).exp()
        denominator = (self.spatial_null[None]-shift).exp()+(exponential*prior[:, None]).sum((-1, -2, -3))
        centered = t-mass[..., None]*average[:, :, None, None]
        p = torch.einsum("bcskl,bsklr->bcr", exponential,
                         centered/(3*z[:, :, None, None, None]))/denominator[..., None]
        spatial_delta = torch.einsum("bcr,cr->bc", p, self.spatial_read)/math.sqrt(self.rank)
        if not self.spatial:
            p, spatial_delta = p*0, spatial_delta*0
        return p, total_delta, spatial_delta, t, mass

    def forward(self, batch, return_evidence=False):
        mask = batch["mask"]
        u = self.cnn(batch)
        h = self.fusion(self.features(batch)+self.cnn_projection(u)).masked_fill(~mask[..., None], 0)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, u = h.float(), u.float()
            pooled = u.masked_fill(~mask[..., None], -torch.inf).amax(1)
            global_logits = self.global_head(self.global_projection(torch.cat([masked_mean(h, mask), pooled], -1)))
            if self.global_only:
                return {"logits": global_logits, "direct": global_logits} if return_evidence else {"logits": global_logits}
            regions, a, active, count, shape, mean_weights = self.regions(h, mask)
            score = torch.einsum("bkd,cd->bck", self.span_key(regions), self.query)/math.sqrt(self.dim)
            score = (score-count.float().log()[:, None, None]).masked_fill(~active[:, None], -torch.inf)
            scores = torch.cat([score, self.local_null[None, :, None].expand(h.shape[0], -1, 1)], -1)
            alpha = scores.softmax(-1)[..., :-1]
            evidence = torch.einsum("bck,bkd->bcd", alpha, self.span_value(regions))
            local = (evidence*self.local_read).sum(-1)/math.sqrt(self.dim)
            p, total, spatial, t, mass = self.relations(batch, regions, a, shape, mean_weights)
            key, value = self.detail_key(h), self.detail_value(h)
            reference = self.detail_query(F.layer_norm(self.query, (self.dim,)))[None].expand(h.shape[0], -1, -1)
            conditional = self.detail_query(F.layer_norm(self.query[None]+evidence, (self.dim,)))+self.detail_relation(p)
            def read(query):
                scores = torch.einsum("bcd,bld->bcl", query, key)/math.sqrt(128)
                return torch.einsum("bcl,bld->bcd", scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1), value)
            detail = (read(conditional)-read(reference)) if self.detail_mode == "conditional" else read(reference)
            detail = (detail*self.detail_read).sum(-1)/math.sqrt(128)
            if self.detail_mode == "none":
                detail = detail*0
            logits = global_logits+self.eta[0]*local+self.eta[1]*total+self.eta[2]*spatial+self.eta[3]*detail
        out = {"logits": logits}
        if return_evidence:
            out.update(assignment=a, regions=regions, relation=t, relation_mass=mass,
                       spatial_summary=p, direct=global_logits, local=local, total=total, spatial=spatial, detail=detail)
        return out

    def compute_loss(self, batch, settings, return_output=False):
        return supervised_loss(self, batch, settings, return_output)
