"""DualPath-MFP: gradient-decoupled local precision and global/heterogeneous bottleneck paths without O(L^3) triangles."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..consistency import bernoulli_rdrop
from ..pharmafield_probe import BioGranularityAdapter, BioGranularityResidualHead, fine39_bio_supervision_loss
from ..research_layers import (
    Attention, FeedForward, TransformerBlock, masked_mean,
    observed_label_loss, symmetric_pu_ranking_loss, training_views,
)
from .helixset import LatentFusionBlock, MotifPyramid, SetDecoderBlock
from .theraset import MotifCNN, TheraSet, dual_terminal_features, motif_classifier, residue_encoding


CORE_PHYS_GROUPS = (
    list(range(0, 17)),     # HELM / modification / chirality summary
    list(range(17, 37)),    # global amino-acid composition (AAC)
    list(range(437, 457)),  # L-configuration AAC
    list(range(457, 477)),  # D-configuration AAC
    list(range(477, 502)),  # whole-peptide biophysics (MW, GRAVY, pI, propensity)
    list(range(502, 515)),  # RDKit molecular descriptors (LogP, TPSA, HBD/HBA, rings)
)
DPC_SLICES = (list(range(37, 237)), list(range(237, 437)))
# Eisenberg consensus hydrophobicity scale in project ACDEFGHIKLMNPQRSTVWY order (0 = pad).
EISENBERG_HYDRO = (
    0.0, 0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06,
    0.64, -0.78, 0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26,
)


def _residue_ph_charges(ph):
    """Henderson-Hasselbalch residue side-chain charges in ACDEFGHIKLMNPQRSTVWY order (0=pad)."""
    # Indices: C=2(8.3), D=3(3.9), E=4(4.1), H=7(6.0), K=9(10.5), R=15(12.5), Y=20(10.1)
    q = [0.0] * 21
    for idx, pka in ((3, 3.9), (4, 4.1), (2, 8.3), (20, 10.1)):
        q[idx] = -1.0 / (1.0 + 10.0 ** (pka - ph))
    for idx, pka in ((7, 6.0), (9, 10.5), (15, 12.5)):
        q[idx] = 1.0 / (1.0 + 10.0 ** (ph - pka))
    return torch.tensor(q, dtype=torch.float32)


class InvariantPropertyEncoder(nn.Module):
    """1 always-available intrinsic sequence biophysics token + 8 high-purity external property tokens (+2 v2 tokens)."""

    def __init__(self, dim, heads, dropout, depth=2, biophysics_v2=False):
        super().__init__()
        self.biophysics_v2 = biophysics_v2
        self.register_buffer("aa_props", residue_encoding())
        self.intrinsic = nn.Sequential(nn.Linear(70, dim), nn.GELU(), nn.LayerNorm(dim))
        self.core_proj = nn.ModuleList([nn.Linear(2 * len(g), dim) for g in CORE_PHYS_GROUPS])
        self.dpc_proj = nn.ModuleList([nn.Linear(2 * len(g), dim) for g in DPC_SLICES])
        self.group_embed = nn.Parameter(torch.randn(1 + len(CORE_PHYS_GROUPS) + len(DPC_SLICES), dim) * 0.02)
        if biophysics_v2:
            self.register_buffer("q_74", _residue_ph_charges(7.4))
            self.register_buffer("q_55", _residue_ph_charges(5.5))
            self.register_buffer("eisenberg", torch.tensor(EISENBERG_HYDRO, dtype=torch.float32))
            self.ph_cap_proj = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.LayerNorm(dim))
            self.amphipathic_proj = nn.Sequential(nn.Linear(18, dim), nn.GELU(), nn.LayerNorm(dim))
            self.v2_group_embed = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads, dropout) for _ in range(depth)])

    def _intrinsic_token(self, aa, mask):
        valid = mask.unsqueeze(-1).float()
        length = valid.sum(dim=1).clamp_min(1.0)
        enc = self.aa_props[aa] * valid
        mean = enc.sum(dim=1) / length
        var = ((enc - mean.unsqueeze(1)) * valid).square().sum(dim=1) / length
        std = (var + 1e-6).sqrt()
        net_charge = enc[:, :, 20].sum(dim=1, keepdim=True) / 10.0
        log_len = (length + 1.0).log() / math.log(75.0)
        return self.intrinsic(torch.cat([mean, std, net_charge, log_len], dim=-1))

    def _biophysics_v2_tokens(self, aa, batch):
        mask = batch["mask"]
        valid = mask.float()
        length = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        caps = batch["terminal_caps"].float() if "terminal_caps" in batch else aa.new_zeros((aa.shape[0], 2), dtype=torch.float32)
        n_free = 1.0 - caps[:, 0:1]
        c_free = 1.0 - caps[:, 1:2]

        q74_res = (self.q_74[aa] * valid).sum(dim=1, keepdim=True)
        q55_res = (self.q_55[aa] * valid).sum(dim=1, keepdim=True)
        q74 = q74_res + n_free / (1.0 + 10.0 ** (7.4 - 8.0)) - c_free / (1.0 + 10.0 ** (3.6 - 7.4))
        q55 = q55_res + n_free / (1.0 + 10.0 ** (5.5 - 8.0)) - c_free / (1.0 + 10.0 ** (3.6 - 5.5))
        dq = q55 - q74
        d_tok = batch["d"].float() if "d" in batch else valid.new_zeros(aa.shape)
        frac_d = (d_tok * valid).sum(dim=1, keepdim=True) / length
        ph_feats = torch.cat([
            q74 / 10.0, q55 / 10.0, dq / 5.0, frac_d, caps[:, 0:1], caps[:, 1:2],
            q74 / length, q55 / length,
        ], dim=-1)
        ph_tok = self.ph_cap_proj(ph_feats)

        h_seq = (self.eisenberg[aa] * valid).unsqueeze(1)  # [B, 1, L]
        m_seq = valid.unsqueeze(1)
        moments = []
        for win in (7, 11, 15):
            ones = torch.ones((1, 1, win), device=aa.device, dtype=torch.float32)
            denom = F.conv1d(m_seq, ones, padding=win // 2).clamp_min(1.0)
            rel = torch.arange(win, device=aa.device, dtype=torch.float32) - (win // 2)
            for deg in (100.0, 120.0, 180.0):
                omega = math.radians(deg)
                k_cos = torch.cos(omega * rel).view(1, 1, win)
                k_sin = torch.sin(omega * rel).view(1, 1, win)
                re = F.conv1d(h_seq, k_cos, padding=win // 2)
                im = F.conv1d(h_seq, k_sin, padding=win // 2)
                mu = torch.sqrt(re.square() + im.square() + 1e-8) / denom
                mu = mu.squeeze(1)
                mu_mean = (mu * valid).sum(dim=1, keepdim=True) / length
                mu_max = mu.masked_fill(~mask, 0.0).amax(dim=1, keepdim=True)
                moments.extend([mu_max, mu_mean])
        amphi_tok = self.amphipathic_proj(torch.cat(moments, dim=-1))
        return ph_tok, amphi_tok

    def forward(self, batch, clean_aa):
        mask = batch["mask"]
        bsz = mask.shape[0]
        tokens = [self._intrinsic_token(clean_aa, mask)]
        token_masks = [mask.any(dim=1)]

        if "phys" in batch and "phys_mask" in batch:
            values = batch["phys"].masked_fill(~batch["phys_mask"], 0.0)
            observed = batch["phys_mask"]
            for g, proj in zip((*CORE_PHYS_GROUPS, *DPC_SLICES), (*self.core_proj, *self.dpc_proj)):
                inp = torch.cat([values[:, g], observed[:, g].float()], dim=-1)
                tokens.append(proj(inp))
                token_masks.append(observed[:, g].any(dim=1))
        else:
            zero = tokens[0].new_zeros(bsz, tokens[0].shape[-1])
            false_m = torch.zeros(bsz, dtype=torch.bool, device=mask.device)
            for _ in range(len(CORE_PHYS_GROUPS) + len(DPC_SLICES)):
                tokens.append(zero)
                token_masks.append(false_m)

        x = torch.stack(tokens, dim=1)
        m = torch.stack(token_masks, dim=1)
        x = (x + self.group_embed.unsqueeze(0)) * m.unsqueeze(-1)
        if self.biophysics_v2:
            ph_tok, amphi_tok = self._biophysics_v2_tokens(clean_aa, batch)
            v2_x = torch.stack([ph_tok, amphi_tok], dim=1) + self.v2_group_embed.unsqueeze(0)
            v2_m = mask.any(dim=1, keepdim=True).expand(-1, 2)
            x = torch.cat([x, v2_x * v2_m.unsqueeze(-1)], dim=1)
            m = torch.cat([m, v2_m], dim=1)
        for layer in self.layers:
            x = layer(x, m)
        return x, m


class PairBiasedSequenceBlock(nn.Module):
    """Depthwise multiscale local filtering + O(L^2) distance/separation-biased attention."""

    def __init__(self, dim, pair_dim, heads, dropout):
        super().__init__()
        self.norm_local = nn.LayerNorm(dim)
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
                                    for k in (3, 7, 15)])
        self.mix_local = nn.Linear(3 * dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.res_norm = nn.LayerNorm(dim)
        self.left = nn.Linear(dim, pair_dim, bias=False)
        self.right = nn.Linear(dim, pair_dim, bias=False)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_bias = nn.Linear(pair_dim, heads, bias=False)
        self.geom_bias = nn.Linear(pair_dim, heads, bias=False)
        self.attn = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, 4, dropout)

    def forward(self, residues, mask, seq_pairs, geom_pairs, pair_quality):
        valid_res = mask.unsqueeze(-1)
        h = (self.norm_local(residues) * valid_res).transpose(1, 2)
        conv_outs = [F.gelu(conv(h)).transpose(1, 2) * valid_res for conv in self.convs]
        residues = (residues + self.dropout(self.mix_local(torch.cat(conv_outs, dim=-1)))) * valid_res

        valid_pair = mask[:, :, None] & mask[:, None, :]
        rn = self.res_norm(residues)
        pairs = (seq_pairs + self.left(rn)[:, :, None] * self.right(rn)[:, None, :]) * valid_pair.unsqueeze(-1)
        bias = (self.pair_bias(self.pair_norm(pairs))
                + self.geom_bias(geom_pairs.tanh()) * pair_quality.unsqueeze(-1))
        bias = bias.permute(0, 3, 1, 2)
        residues = self.ff(residues + self.attn(residues, residues, mask, mask, bias=bias), mask)
        return residues, pairs


def _sample_span_drop(mask, drop_prob, exact_rate=False):
    """Sample contiguous 1-3 residue span masks while guaranteeing >=1 surviving residue per peptide."""
    if drop_prob <= 0.0:
        return torch.zeros_like(mask)
    bsz, width = mask.shape
    seed_prob = (1.0 - (1.0 - min(drop_prob, 0.95)) ** (1.0 / 3.0)) if exact_rate else (drop_prob * 0.5)
    triggers = (torch.rand((bsz, width), device=mask.device) < seed_prob) & mask
    expanded = F.max_pool1d(triggers.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0
    drop = expanded & mask
    surviving = mask & ~drop
    all_dropped = mask.any(dim=1) & ~surviving.any(dim=1)
    if all_dropped.any():
        first_valid = mask.float().argmax(dim=1)
        drop[all_dropped, first_valid[all_dropped]] = False
    return drop


class GlobalBottleneckPath(nn.Module):
    """Global bottleneck + optional multi-resolution SetDecoder (zero O(L^3)) targeting low-homology transfer."""

    def __init__(self, input_dim, num_labels, vocab_size, statistics, peptide_input_dim=0,
                 dim=256, heads=8, pair_dim=32, seq_depth=2, prop_depth=2, latent_count=24,
                 latent_depth=4, decoder_depth=0, dropout=0.25, span_mask_prob=0.15,
                 clean_span_mask=False, biophysics_v2=False, use_bio_head=False, bio_statistics=None):
        super().__init__()
        self.span_mask_prob = span_mask_prob
        self.decoder_depth = decoder_depth
        self.use_bio_head = use_bio_head
        self.plm_proj = nn.Linear(input_dim + peptide_input_dim, dim)
        self.aa_emb = nn.Embedding(21, dim, padding_idx=0)
        self.d_emb = nn.Embedding(2, dim)
        self.chem_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.term_proj = nn.Linear(7, dim)
        self.input_gate = nn.Linear(2 * dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim)) if (decoder_depth > 0 or clean_span_mask) else None
        self.global_chem_layer = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))

        self.properties = InvariantPropertyEncoder(dim, heads, dropout, depth=prop_depth, biophysics_v2=biophysics_v2)
        self.geom_node = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.LayerNorm(dim))
        self.dssp_emb = nn.Embedding(10, dim)
        self.geom_global = nn.Sequential(nn.Linear(50, dim), nn.GELU(), nn.LayerNorm(dim))
        self.geom_mix = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.Tanh())

        self.pair_sep = nn.Linear(2, pair_dim)
        self.pair_geom = nn.Sequential(nn.Linear(22, pair_dim), nn.LayerNorm(pair_dim))
        self.seq_blocks = nn.ModuleList([
            PairBiasedSequenceBlock(dim, pair_dim, heads, dropout) for _ in range(seq_depth)
        ])
        self.pyramid = MotifPyramid(dim, dropout)
        self.latents = nn.Parameter(torch.randn(latent_count, dim) * 0.02)
        self.latent_blocks = nn.ModuleList([
            LatentFusionBlock(dim, heads, dropout) for _ in range(latent_depth)
        ])

        self.label_queries = nn.Parameter(torch.randn(num_labels, dim) * 0.02)
        prior = torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1.0 - 1e-4).logit()
        if decoder_depth > 0:
            self.decoder = nn.ModuleList([
                SetDecoderBlock(dim, heads, dropout, pair_dim) for _ in range(decoder_depth)
            ])
            self.latent_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_labels))
            with torch.no_grad():
                self.latent_head[-1].bias.copy_(prior)
        else:
            self.decoder = None
            self.latent_head = None
            self.read_latent = Attention(dim, heads, dropout)
            self.read_prop = Attention(dim, heads, dropout)
        self.out_norm = nn.LayerNorm(dim)
        if use_bio_head:
            b_prior = torch.tensor((bio_statistics or statistics)["observed_prior"]).clamp(1e-4, 1.0 - 1e-4).logit()
            self.bio_head = BioGranularityResidualHead(dim, statistics["labels"], b_prior)
        else:
            self.bio_head = None
            self.label_weights = nn.Parameter(torch.randn(num_labels, dim) / math.sqrt(dim))
            self.label_bias = nn.Parameter(prior)

    def _pair_inputs(self, batch, structural, reliability):
        mask = batch["mask"]
        bsz, width = mask.shape
        pos = torch.arange(width, device=mask.device, dtype=torch.float32)
        sep = (pos[:, None] - pos[None, :]) / 74.0
        valid = mask[:, :, None] & mask[:, None, :]
        seq_pairs = self.pair_sep(torch.stack([sep, sep.abs()], dim=-1)).unsqueeze(0) * valid.unsqueeze(-1)

        observed = structural[:, :, None] & structural[:, None, :]
        observed = observed & ~torch.eye(width, dtype=torch.bool, device=mask.device).unsqueeze(0)
        if observed.any():
            coords = batch["coords"].float().masked_fill(~structural.unsqueeze(-1), 0.0)
            with torch.autocast(device_type=coords.device.type, enabled=False):
                distance = batch.get("_shared_distances")
                if distance is None:
                    distance = torch.cdist(coords, coords)
                centers = torch.linspace(0, 30, 16, device=coords.device)
                rbf = torch.exp(-((distance.unsqueeze(-1) - centers) / 2.0).square())
            kind = batch["edge_type"].masked_fill(~observed, 0)
            known = batch["confidence_known"] & structural
            conf = batch["geometry"][..., 9].masked_fill(~known, 0.0).clamp(0.0, 1.0)
            g_feat = torch.cat([
                rbf, F.one_hot(kind, 3).float(), observed.unsqueeze(-1).float(),
                (known[:, :, None] & known[:, None, :]).unsqueeze(-1).float(),
                (conf[:, :, None] * conf[:, None, :]).unsqueeze(-1),
            ], dim=-1)
            geom_pairs = self.pair_geom(g_feat) * observed.unsqueeze(-1)
            pair_quality = reliability[:, :, None] * reliability[:, None, :] * observed
        else:
            geom_pairs = seq_pairs.new_zeros((bsz, width, width, seq_pairs.shape[-1]))
            pair_quality = seq_pairs.new_zeros((bsz, width, width))
        return seq_pairs, geom_pairs, pair_quality

    def forward(self, batch):
        mask = batch["mask"]
        bsz, width = mask.shape
        clean_aa = batch["aa"]
        d_tok = batch.get("d", torch.zeros_like(clean_aa))
        res_chem = batch.get("residue_chem", torch.zeros_like(clean_aa))
        chem = batch.get("chemical", clean_aa.new_zeros((bsz, 1)))
        plm_source = batch.get("_path_b_plm", batch["_shared_dual_esm"])

        if self.mask_token is not None:
            # Stage 3: clean unscaled fusion followed by learnable [MASK] replacement (zero train/eval norm shift).
            plm = self.plm_proj(plm_source)
            annotated = (self.aa_emb(clean_aa) + self.d_emb(d_tok)
                         + self.chem_emb(res_chem) + self.term_proj(dual_terminal_features(batch)))
            residues = plm + torch.sigmoid(self.input_gate(torch.cat([plm, annotated], dim=-1))) * annotated
            if self.training and self.span_mask_prob > 0.0:
                drop = _sample_span_drop(mask, self.span_mask_prob, exact_rate=True)
                residues = torch.where(drop.unsqueeze(-1), self.mask_token.to(residues.dtype), residues)
            residues = residues * mask.unsqueeze(-1)
        else:
            # Legacy Stage 2/2b checkpoint compatibility path.
            aa = clean_aa
            dual_esm = plm_source
            if self.training and self.span_mask_prob > 0.0:
                drop = _sample_span_drop(mask, self.span_mask_prob, exact_rate=False)
                keep_scale = (~drop).unsqueeze(-1).float() / max(1.0 - self.span_mask_prob, 1e-4)
                aa = aa.masked_fill(drop, 0)
                d_tok = d_tok.masked_fill(drop, 0)
                res_chem = res_chem.masked_fill(drop, 0)
                dual_esm = dual_esm * keep_scale * mask.unsqueeze(-1).float()
            plm = self.plm_proj(dual_esm)
            annotated = (self.aa_emb(aa) + self.d_emb(d_tok)
                         + self.chem_emb(res_chem) + self.term_proj(dual_terminal_features(batch)))
            residues = (plm + torch.sigmoid(self.input_gate(torch.cat([plm, annotated], dim=-1))) * annotated) * mask.unsqueeze(-1)

        global_chem = masked_mean(self.chem_emb(chem), chem != 0)
        seq_global = self.global_chem_layer(torch.cat([masked_mean(residues, mask), global_chem], dim=-1))

        prop_tokens, prop_mask = self.properties(batch, clean_aa)

        has_struct = "struct_mask" in batch and "geometry" in batch and "coords" in batch
        if has_struct:
            structural = batch["struct_mask"] & mask
            known = batch["confidence_known"] & structural
            reliability = torch.where(known, 0.3 + 0.7 * batch["geometry"][..., 9].clamp(0.0, 1.0), 1.0) * structural
            geometry = (self.geom_node(batch["geometry"].masked_fill(~structural.unsqueeze(-1), 0.0))
                        + self.dssp_emb(batch["dssp"].masked_fill(~structural, 9))) * structural.unsqueeze(-1)
        else:
            structural = torch.zeros_like(mask)
            reliability = mask.new_zeros((bsz, width), dtype=residues.dtype)
            geometry = residues.new_zeros((bsz, width, residues.shape[-1]))

        residues = residues + self.geom_mix(torch.cat([residues, geometry], dim=-1)) * geometry * reliability.unsqueeze(-1)
        seq_pairs, geom_pairs, pair_quality = self._pair_inputs(batch, structural, reliability)
        for block in self.seq_blocks:
            residues, seq_pairs = block(residues, mask, seq_pairs, geom_pairs, pair_quality)

        motifs, motif_mask, motif_weights = self.pyramid(residues, mask)
        present_seq = mask.any(dim=1, keepdim=True)
        motifs = torch.cat([motifs, seq_global.unsqueeze(1)], dim=1)
        motif_mask = torch.cat([motif_mask, present_seq], dim=1)
        motif_weights = torch.cat([motif_weights, present_seq.float()], dim=1)

        struct_present = structural.any(dim=1, keepdim=True)
        if has_struct and "struct_global" in batch and "struct_global_mask" in batch:
            g_mask = batch["struct_global_mask"] & struct_present
            g_struct = self.geom_global(torch.cat([batch["struct_global"].masked_fill(~g_mask, 0.0), g_mask.float()], dim=-1))
            g_struct = g_struct + (geometry * reliability.unsqueeze(-1)).sum(dim=1) / reliability.sum(dim=1, keepdim=True).clamp_min(1e-4)
        else:
            g_struct = residues.new_zeros((bsz, residues.shape[-1]))
        geom_mem = torch.cat([geometry, (g_struct.unsqueeze(1) * struct_present.unsqueeze(-1))], dim=1)
        geom_mask = torch.cat([structural, struct_present], dim=1)
        mean_rel = (reliability * structural).sum(dim=1, keepdim=True) / structural.sum(dim=1, keepdim=True).clamp_min(1.0)
        geom_rel = torch.cat([reliability, mean_rel], dim=1)

        memories = (motifs, prop_tokens, geom_mem)
        masks = (motif_mask, prop_mask, geom_mask)
        latent = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        latent_mask = torch.ones((bsz, latent.shape[1]), dtype=torch.bool, device=mask.device)
        for block in self.latent_blocks:
            latent = block(latent, latent_mask, memories, masks, geom_rel, motif_weights)

        queries = self.label_queries.unsqueeze(0).expand(bsz, -1, -1)
        q_mask = torch.ones((bsz, queries.shape[1]), dtype=torch.bool, device=mask.device)
        if self.decoder is not None:
            for dec in self.decoder:
                queries, _ = dec(
                    queries, q_mask, latent, latent_mask, memories, masks, geom_rel,
                    residues, seq_pairs, mask, geom_pairs, pair_quality, motif_weights,
                    label_context=False, pair_readout=False, fine_evidence=True, pair_interactions=False,
                )
            pooled = self.out_norm(queries)
            if self.bio_head is not None:
                bio_out = self.bio_head(pooled)
                return {
                    "logits": bio_out["obs_logits"],
                    "bio_logits": bio_out["bio_logits"],
                    "latent_logits": self.latent_head(latent.mean(dim=1)),
                }
            logits = (pooled * self.label_weights.unsqueeze(0)).sum(dim=-1) + self.label_bias
            return {"logits": logits, "latent_logits": self.latent_head(latent.mean(dim=1))}

        ev_lat = self.read_latent(queries, latent, q_mask, latent_mask)
        ev_prop = self.read_prop(queries, prop_tokens, q_mask, prop_mask)
        pooled = self.out_norm(queries + ev_lat + ev_prop)
        if self.bio_head is not None:
            bio_out = self.bio_head(pooled)
            return {"logits": bio_out["obs_logits"], "bio_logits": bio_out["bio_logits"]}
        logits = (pooled * self.label_weights.unsqueeze(0)).sum(dim=-1) + self.label_bias
        return {"logits": logits}


class PureAA34Path(nn.Module):
    """Pure sequence AA34+terminal anchor (~2.0M, zero ESM, zero chemistry, zero structure)."""

    def __init__(self, num_labels, statistics, cnn_channels=128, dropout=0.2):
        super().__init__()
        self.cnn = MotifCNN(1, cnn_channels, dropout, chemical_evidence=False, terminal_features=True)
        self.classifier = motif_classifier(self.cnn.output_channels, num_labels, dropout)
        prior = torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1.0 - 1e-4).logit()
        with torch.no_grad():
            self.classifier[-1].bias.copy_(prior)

    def forward(self, batch):
        _, pooled = self.cnn(batch)
        return {"logits": self.classifier(pooled)}


class DualPathMFP(nn.Module):
    """Gradient-decoupled local anchor (Path A) + global/heterogeneous bottleneck (Path B) + optional Path C."""

    def __init__(self, input_dim, num_labels, vocab_size, statistics, peptide_input_dim=0,
                 orth_input_dim=0, dim=256, heads=8, dropout=0.2, cnn_channels=128, pair_dim=32,
                 b_seq_depth=2, b_prop_depth=2, b_latent_count=24, b_latent_depth=4,
                 b_decoder_depth=0, b_dropout=0.25, span_mask_prob=0.15, use_aa34_path=False,
                 full_branch_rdrop=False, fusion_weights=(0.4, 0.4, 0.2),
                 clean_span_mask=False, b_use_peptide_esm=True,
                 biophysics_v2=False, use_bio_head=False, bio_statistics=None):
        super().__init__()
        self.statistics = statistics
        self.bio_statistics = bio_statistics
        self.input_dim = input_dim
        self.peptide_input_dim = peptide_input_dim
        self.orth_input_dim = orth_input_dim
        self.use_aa34_path = use_aa34_path
        self.full_branch_rdrop = full_branch_rdrop
        self.fusion_weights = tuple(fusion_weights)
        self.b_use_peptide_esm = b_use_peptide_esm
        self.use_bio_head = use_bio_head
        self.path_a = TheraSet(
            input_dim=input_dim, num_labels=num_labels, vocab_size=vocab_size, statistics=statistics,
            dim=dim, heads=heads, dropout=dropout, cnn_channels=cnn_channels,
            context_depth=1, property_depth=1, pair_dim=pair_dim, decoder_depth=2,
            peptide_input_dim=peptide_input_dim, label_context=False, prototype_readout=False,
            terminal_features=True,
        )
        self.path_a_gran = BioGranularityAdapter(dim, statistics["labels"]) if use_bio_head else None
        b_in_dim = orth_input_dim if orth_input_dim > 0 else input_dim
        b_pep_dim = peptide_input_dim if (b_use_peptide_esm or orth_input_dim <= 0) else 0
        self.path_b = GlobalBottleneckPath(
            input_dim=b_in_dim, num_labels=num_labels, vocab_size=vocab_size, statistics=statistics,
            peptide_input_dim=b_pep_dim, dim=dim, heads=heads, pair_dim=pair_dim,
            seq_depth=b_seq_depth, prop_depth=b_prop_depth, latent_count=b_latent_count,
            latent_depth=b_latent_depth, decoder_depth=b_decoder_depth,
            dropout=b_dropout, span_mask_prob=span_mask_prob,
            clean_span_mask=clean_span_mask,
            biophysics_v2=biophysics_v2, use_bio_head=use_bio_head, bio_statistics=bio_statistics,
        )
        self.path_c = PureAA34Path(num_labels, statistics, cnn_channels=cnn_channels, dropout=dropout) if use_aa34_path else None

    def branch_parameter_groups(self):
        group_a = list(self.path_a.parameters())
        if self.path_a_gran is not None:
            group_a.extend(self.path_a_gran.parameters())
        groups = [group_a, list(self.path_b.parameters())]
        if self.path_c is not None:
            groups.append(list(self.path_c.parameters()))
        return groups

    def _prepare_shared(self, batch):
        shared = dict(batch)
        mask = batch["mask"].unsqueeze(-1)
        esm = F.layer_norm(batch["esm"].masked_fill(~mask, 0.0), (self.input_dim,))
        pep = None
        if self.peptide_input_dim and "esm_peptide" in batch:
            pep = F.layer_norm(batch["esm_peptide"].masked_fill(~mask, 0.0), (self.peptide_input_dim,))
            dual = torch.cat([esm, pep], dim=-1)
        else:
            dual = esm
        shared["_shared_dual_esm"] = dual.masked_fill(~mask, 0.0)
        if self.orth_input_dim > 0:
            if "esm_orth" not in batch:
                raise ValueError("DualPathMFP configured with orth_input_dim requires 'esm_orth' in batch")
            orth = F.layer_norm(batch["esm_orth"].masked_fill(~mask, 0.0), (self.orth_input_dim,))
            b_plm = torch.cat([orth, pep], dim=-1) if (pep is not None and self.b_use_peptide_esm) else orth
            shared["_path_b_plm"] = b_plm.masked_fill(~mask, 0.0)
        else:
            shared["_path_b_plm"] = shared["_shared_dual_esm"]
        if "struct_mask" in batch and "coords" in batch:
            structural = batch["struct_mask"] & batch["mask"]
            if structural.any():
                coords = batch["coords"].float().masked_fill(~structural.unsqueeze(-1), 0.0)
                with torch.autocast(device_type=coords.device.type, enabled=False):
                    shared["_shared_distances"] = torch.cdist(coords, coords)
        return shared

    def forward(self, batch):
        shared = self._prepare_shared(batch)
        out_a = self.path_a(shared, return_states=self.use_bio_head)
        out_b = self.path_b(shared)
        if self.path_a_gran is not None:
            logits_a = self.path_a_gran(out_a["evidence_state"], out_a["global_state"], out_a["logits"])
        else:
            logits_a = out_a["logits"]
        logits_b = out_b["logits"]
        out = {
            "path_a_logits": logits_a,
            "path_b_logits": logits_b,
            "path_a_direct": out_a["direct"],
            "path_a_evidence": out_a["evidence_logits"],
        }
        if self.use_bio_head:
            out["path_a_bio_logits"] = out_a["logits"]
            out["path_b_bio_logits"] = out_b["bio_logits"]
        if "latent_logits" in out_b:
            out["path_b_latent"] = out_b["latent_logits"]
        if self.path_c is not None:
            logits_c = self.path_c(shared)["logits"]
            wa, wb, wc = self.fusion_weights
            # Detach fused output so any external loss on output["logits"] cannot leak across branches.
            fused = wa * logits_a.detach() + wb * logits_b.detach() + wc * logits_c.detach()
            out["path_c_logits"] = logits_c
        else:
            fused = 0.5 * (logits_a.detach() + logits_b.detach())
        out["logits"] = fused
        return out

    def _branch_terms(self, output, targets, settings, batch=None):
        weak_a = settings.get("unlabeled_weight", 0.3)
        weak_b = settings.get("b_unlabeled_weight", weak_a)
        pos_mult = batch.get("positive_multiplier") if (batch is not None and self.training) else None
        has_bio = (
            self.use_bio_head
            and batch is not None
            and "y_bio" in batch
            and "m_obs" in batch
        )
        bio_stats = self.bio_statistics or self.statistics
        bio_target = batch["y_bio"] if has_bio else targets
        bio_mask = batch["m_obs"] if has_bio else None

        terms = {
            "path_a_cls": observed_label_loss(output["path_a_logits"], targets, self.statistics, weak_a,
                                              positive_multiplier=pos_mult),
            "direct": observed_label_loss(output["path_a_direct"], bio_target,
                                          bio_stats if has_bio else self.statistics, weak_a,
                                          mask=bio_mask, positive_multiplier=pos_mult),
            "evidence": observed_label_loss(output["path_a_evidence"], bio_target,
                                            bio_stats if has_bio else self.statistics, weak_a,
                                            mask=bio_mask, positive_multiplier=pos_mult),
            "path_b_cls": observed_label_loss(output["path_b_logits"], targets, self.statistics, weak_b,
                                              positive_multiplier=pos_mult),
        }
        if has_bio:
            terms["path_a_bio"] = fine39_bio_supervision_loss(
                output["path_a_bio_logits"], bio_target, bio_mask, bio_stats, weak_a, positive_multiplier=pos_mult,
            )
            terms["path_b_bio"] = fine39_bio_supervision_loss(
                output["path_b_bio_logits"], bio_target, bio_mask, bio_stats, weak_b, positive_multiplier=pos_mult,
            )
        if "path_b_latent" in output:
            terms["b_latent"] = observed_label_loss(
                output["path_b_latent"], bio_target, bio_stats if has_bio else self.statistics,
                weak_b, mask=bio_mask, positive_multiplier=pos_mult,
            )
        if "path_c_logits" in output:
            terms["path_c_cls"] = observed_label_loss(output["path_c_logits"], targets, self.statistics, weak_a,
                                                      positive_multiplier=pos_mult)
        if settings.get("sym_rank", 0.0) > 0.0:
            terms["sym_rank"] = symmetric_pu_ranking_loss(
                output["path_b_logits"], targets, self.statistics,
                top_k=settings.get("sym_rank_topk", 3),
                pos_margin=settings.get("sym_rank_pos_margin", 0.5),
                neg_margin=settings.get("sym_rank_neg_margin", -0.5),
                gamma=settings.get("sym_rank_gamma", 2.0),
                cooccur_beta=settings.get("sym_rank_beta", 1.0),
            )
        return terms

    def compute_loss(self, batch, settings, return_output=False):
        view_a = training_views(batch, settings.get("modality_dropout", 0.1), 0.0)[0] if self.training else batch
        output = self(view_a)
        terms = self._branch_terms(output, batch["y"], settings, batch=batch)
        rdrop_weight = settings.get("rdrop", 0.0)
        agreement_mean = output["path_a_logits"].new_zeros(())
        agreement_loss = agreement_mean
        if self.training and rdrop_weight > 0.0:
            second = self(view_a)
            other = self._branch_terms(second, batch["y"], settings, batch=batch)
            terms = {k: 0.5 * (v + other[k]) for k, v in terms.items()}
            kls = [
                bernoulli_rdrop(output["path_a_logits"], second["path_a_logits"]),
                bernoulli_rdrop(output["path_b_logits"], second["path_b_logits"]),
            ]
            if "path_c_logits" in output:
                kls.append(bernoulli_rdrop(output["path_c_logits"], second["path_c_logits"]))
            kl_sum = sum(kls)
            agreement_mean = kl_sum / len(kls)
            agreement_loss = kl_sum if self.full_branch_rdrop else agreement_mean
        bio_w = settings.get("bio_weight", 0.5)
        loss_a = (terms["path_a_cls"]
                  + bio_w * terms.get("path_a_bio", 0.0)
                  + settings.get("direct", 0.2) * terms["direct"]
                  + settings.get("evidence", 0.1) * terms["evidence"])
        loss_b = (terms["path_b_cls"]
                  + bio_w * terms.get("path_b_bio", 0.0)
                  + settings.get("b_latent", 0.15) * terms.get("b_latent", 0.0)
                  + settings.get("sym_rank", 0.0) * terms.get("sym_rank", 0.0))
        loss_c = terms.get("path_c_cls", 0.0)
        terms["classification"] = 0.5 * (terms["path_a_cls"].detach() + terms["path_b_cls"].detach())
        terms["rdrop"] = agreement_mean.detach() if self.full_branch_rdrop else agreement_mean
        total = loss_a + loss_b + loss_c + rdrop_weight * agreement_loss
        return (total, terms, output) if return_output else (total, terms)
