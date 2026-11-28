"""PharmaField-MFP Probe 1 components:
1. BioGranularityResidualHead: decouples latent biological activity (Fine39-Bio Y_bio, M_obs)
   from additive literature-reporting granularity residuals (no multiplicative bottlenecks).
2. ExactSubsetEnergyHead: low-DoF structured energy over all 2^M - 1 = 255 non-empty
   top-M subsets with granularity-mutex exclusion, listwise Expected Jaccard training,
   and exact Bayes-optimal Expected Jaccard decoding.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .bio_taxonomy import fine39_hierarchy_spec, granularity_mutex_matrix
from .research_layers import observed_label_loss


class BioGranularityResidualHead(nn.Module):
    """Additive biological-to-granularity head: z_obs = z_bio + delta_gran."""

    def __init__(self, dim, labels, prior_logits=None):
        super().__init__()
        spec = fine39_hierarchy_spec(labels)
        num_labels = len(labels)
        fam_proj = torch.tensor(spec["projection"].T, dtype=torch.float32)  # [15, 39]
        fam_norm = fam_proj / fam_proj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("fam_norm", fam_norm)
        self.register_buffer("fam_proj", fam_proj)

        self.bio_weights = nn.Parameter(torch.randn(num_labels, dim) * (dim ** -0.5))
        self.bio_bias = nn.Parameter(prior_logits.clone() if prior_logits is not None else torch.zeros(num_labels))

        # Lightweight additive granularity adapter conditioned on label state + own family summary.
        self.gran_net = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        # Zero-init final projection so z_obs starts identical to z_bio (zero cold-start shock).
        nn.init.zeros_(self.gran_net[-1].weight)
        nn.init.zeros_(self.gran_net[-1].bias)

    def forward(self, pooled_label_states):
        """pooled_label_states: [B, 39, D] -> dict(bio_logits, obs_logits, delta_gran)."""
        bio_logits = (pooled_label_states * self.bio_weights.unsqueeze(0)).sum(dim=-1) + self.bio_bias
        fam_states = torch.einsum("fc,bcd->bfd", self.fam_norm, pooled_label_states)
        label_fam_ctx = torch.einsum("fc,bfd->bcd", self.fam_proj, fam_states)
        feat = torch.cat([pooled_label_states, label_fam_ctx, bio_logits.detach().unsqueeze(-1)], dim=-1)
        delta_gran = self.gran_net(feat).squeeze(-1)
        return {
            "bio_logits": bio_logits,
            "obs_logits": bio_logits + delta_gran,
            "delta_gran": delta_gran,
        }


class BioGranularityAdapter(nn.Module):
    """Zero-initialized additive granularity residual for Path A (TheraSet) bio->fine39 adaptation."""

    def __init__(self, dim, labels):
        super().__init__()
        spec = fine39_hierarchy_spec(labels)
        fam_proj = torch.tensor(spec["projection"].T, dtype=torch.float32)  # [15, 39]
        fam_norm = fam_proj / fam_proj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("fam_norm", fam_norm)
        self.register_buffer("fam_proj", fam_proj)
        self.gran_net = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.gran_net[-1].weight)
        nn.init.zeros_(self.gran_net[-1].bias)

    def forward(self, evidence_state, global_state, bio_logits):
        fam_states = torch.einsum("fc,bcd->bfd", self.fam_norm, evidence_state)
        label_fam_ctx = torch.einsum("fc,bfd->bcd", self.fam_proj, fam_states)
        g_exp = global_state.unsqueeze(1).expand_as(evidence_state)
        feat = torch.cat([evidence_state, g_exp, label_fam_ctx, bio_logits.detach().unsqueeze(-1)], dim=-1)
        delta_gran = self.gran_net(feat).squeeze(-1)
        return bio_logits + delta_gran


def fine39_bio_supervision_loss(bio_logits, y_bio, m_obs, statistics, unknown_weight=0.35, positive_multiplier=None):
    """Masked PU-ASL on biologically upward-closed Fine39-Bio targets (zero grad where m_obs == 0)."""
    return observed_label_loss(
        bio_logits, y_bio, statistics, unknown_weight=unknown_weight,
        mask=m_obs, positive_multiplier=positive_multiplier,
    )


class ExactSubsetEnergyHead(nn.Module):
    """Structured energy over all 2^M - 1 = 255 non-empty subsets of top-M calibrated logits."""

    def __init__(self, labels, log_pos_weights, top_m=8, include_growth_mutex=False):
        super().__init__()
        self.labels = [str(x) for x in labels]
        num_labels = len(self.labels)
        m = min(int(top_m), num_labels)
        self.top_m = m

        mutex = granularity_mutex_matrix(self.labels, include_growth=include_growth_mutex)
        self.register_buffer("mutex_matrix", torch.from_numpy(mutex))
        self.register_buffer("log_pos_w", torch.as_tensor(log_pos_weights, dtype=torch.float32))

        states = np.array([[(s >> j) & 1 for j in range(m)] for s in range(1 << m)], dtype=np.float32)
        cand_s = states[1:]  # [255, M], non-empty subsets
        inter = cand_s @ cand_s.T
        union = cand_s.sum(1, keepdims=True) + cand_s.sum(1, keepdims=True).T - inter
        j_255 = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0).astype(np.float32)
        card_idx = cand_s.sum(1).astype(np.int64) - 1  # [255] in 0..M-1
        pair_in_c = (cand_s[:, :, None] * cand_s[:, None, :]) > 0.5  # [255, M, M]

        self.register_buffer("cand_s", torch.from_numpy(cand_s))
        self.register_buffer("j_mat_255", torch.from_numpy(j_255))
        self.register_buffer("cand_card_idx", torch.from_numpy(card_idx))
        self.register_buffer("pair_in_c", torch.from_numpy(pair_in_c))

        # Low-DoF global parameters (~3 + 2*M scalars = 19 scalars for M=8).
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))  # softplus(0) ~ 0.693; init via set_calibration
        self.beta0 = nn.Parameter(torch.tensor(0.0))
        self.beta1 = nn.Parameter(torch.tensor(0.0))
        self.card_bias = nn.Parameter(torch.zeros(m))
        self.entropy_slope = nn.Parameter(torch.zeros(m))

    def init_from_calibration(self, alpha=1.0, beta0=0.0, beta1=0.0):
        with torch.no_grad():
            a = max(float(alpha), 0.05)
            # inverse softplus: log(exp(a) - 1)
            self.raw_alpha.fill_(float(np.log(np.expm1(a))) if a < 20.0 else a)
            self.beta0.fill_(float(beta0))
            self.beta1.fill_(float(beta1))

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha).clamp(0.1, 6.0)

    def calibrated_logits(self, logits):
        return self.alpha * logits + self.beta0 - self.beta1 * self.log_pos_w.unsqueeze(0)

    def forward(self, logits):
        """Return (energies [B, 255], top_idx [B, M], illegal_mask [B, 255], z_cal [B, C])."""
        z_cal = self.calibrated_logits(logits)
        top_idx = torch.topk(z_cal.detach(), self.top_m, dim=-1).indices  # [B, M]
        z_top = z_cal.gather(1, top_idx)  # [B, M]

        # Normalized binary entropy of top-M calibrated probabilities as sample uncertainty feature.
        p_top = z_top.detach().sigmoid().clamp(1e-5, 1.0 - 1e-5)
        ent = -(p_top * p_top.log() + (1.0 - p_top) * (1.0 - p_top).log()).mean(dim=-1, keepdim=True) / np.log(2.0)

        # Subset score = sum_{c in S} z_cal_c + card_bias[|S|-1] + entropy_slope[|S|-1] * ent
        unary = z_top @ self.cand_s.T  # [B, 255]
        card_term = self.card_bias[self.cand_card_idx].unsqueeze(0) + self.entropy_slope[self.cand_card_idx].unsqueeze(0) * ent
        energies = unary + card_term

        sub_m = self.mutex_matrix[top_idx.unsqueeze(-1), top_idx.unsqueeze(-2)]  # [B, M, M]
        illegal = torch.einsum("bmn,smn->bs", sub_m.float(), self.pair_in_c.float()) > 0.5
        energies = energies.masked_fill(illegal, -1e9)
        return energies, top_idx, illegal, z_cal

    def expected_jaccard_loss(self, logits, targets, reg_weight=1e-3):
        """Listwise Expected Jaccard loss over all 255 non-empty top-M subsets."""
        energies, top_idx, _, _ = self.forward(logits)
        probs_s = F.softmax(energies, dim=-1)  # [B, 255]

        y = targets.float()
        y_top = y.gather(1, top_idx)  # [B, M]
        y_total = y.sum(dim=-1, keepdim=True)  # [B, 1]
        cand_sizes = self.cand_s.sum(dim=-1).unsqueeze(0)  # [1, 255]
        inter = y_top @ self.cand_s.T  # [B, 255]
        union = y_total + cand_sizes - inter
        j_true = torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(inter))

        expected_j = (probs_s * j_true).sum(dim=-1)
        valid = y_total.squeeze(-1) > 0
        loss = -expected_j[valid].clamp_min(1e-6).log().mean() if valid.any() else expected_j.sum() * 0.0
        reg = reg_weight * (self.card_bias.square().sum() + self.entropy_slope.square().sum())
        return loss + reg

    @torch.no_grad()
    def decode(self, logits):
        """Bayes-optimal Expected Jaccard subset selection under P_theta(S | x)."""
        energies, top_idx, illegal, _ = self.forward(logits)
        probs_s = F.softmax(energies, dim=-1)  # [B, 255]
        exp_j = probs_s @ self.j_mat_255  # [B, 255]
        exp_j = exp_j.masked_fill(illegal, -1.0)
        best_idx = exp_j.argmax(dim=-1)
        best_s = self.cand_s[best_idx] > 0.5  # [B, M]
        pred = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
        pred.scatter_(1, top_idx, best_s)
        return pred.cpu().numpy()
