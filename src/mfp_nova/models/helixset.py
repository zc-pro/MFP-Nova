"""HelixSet: residue-pair reasoning, multiscale memory and latent set decoding.

Consumes the existing frozen-ESM cache and multimodal batch. Inference reads no
label targets, empirical label graph or retrieved examples and predicts no coordinates.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..research_layers import (
    Attention, FeedForward, PropertyEncoder, SequenceEncoder, TransformerBlock,
    observed_label_loss, training_views,
)


class ResiduePairBlock(nn.Module):
    """Sequence-conditioned pair updates; triangles are latent, not physical bonds."""

    def __init__(self, dim, pair_dim, heads, dropout, incoming=False):
        super().__init__()
        self.incoming = incoming
        self.residue_norm = nn.LayerNorm(dim)
        self.left = nn.Linear(dim, pair_dim)
        self.right = nn.Linear(dim, pair_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.triangle_left = nn.Linear(pair_dim, pair_dim)
        self.triangle_right = nn.Linear(pair_dim, pair_dim)
        self.triangle_out = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, pair_dim))
        self.geometry_norm = nn.LayerNorm(pair_dim)
        self.geometry_left = nn.Linear(pair_dim, pair_dim, bias=False)
        self.geometry_right = nn.Linear(pair_dim, pair_dim, bias=False)
        # No affine offset or output normalization: zero/weak geometric paths
        # must remain zero/weak after triangle aggregation.
        self.geometry_out = nn.Linear(pair_dim, pair_dim, bias=False)
        self.geometry_bias = nn.Linear(pair_dim, heads, bias=False)
        self.geometry_scale = nn.Parameter(torch.full((pair_dim,), .1))
        self.pair_ff = FeedForward(pair_dim, 2, dropout)
        self.bias = nn.Linear(pair_dim, heads, bias=False)
        self.read = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, 4, dropout)
        self.pair_scale = nn.Parameter(torch.full((pair_dim,), .1))

    def forward(self, residues, pairs, mask, geometry_pairs, pair_quality, structural, triangles=True):
        valid = mask[:, :, None] & mask[:, None, :]
        normalized = self.residue_norm(residues)
        product = self.left(normalized)[:, :, None] * self.right(normalized)[:, None, :]
        pairs = (pairs + product * self.pair_scale.tanh()) * valid[..., None]
        if triangles:
            z = self.pair_norm(pairs)
            left = self.triangle_left(z).tanh() * valid[..., None]
            right = self.triangle_right(z).tanh() * valid[..., None]
            # Sum through real intermediate residues only. Bound the product and
            # accumulate in FP32 for the V100 autocast path.
            with torch.autocast(device_type=residues.device.type, enabled=False):
                equation = "bkid,bkjd->bijd" if self.incoming else "bikd,bjkd->bijd"
                triangle = torch.einsum(equation, left.float(), right.float())
                triangle = triangle / mask.sum(1).clamp_min(1)[:, None, None, None]
            pairs = pairs + self.triangle_out(triangle.to(z.dtype)) * self.pair_scale.tanh()
            geometry = self.geometry_norm(geometry_pairs)
            left = self.geometry_left(geometry).tanh() * pair_quality[..., None]
            right = self.geometry_right(geometry).tanh() * pair_quality[..., None]
            with torch.autocast(device_type=residues.device.type, enabled=False):
                triangle = torch.einsum(equation, left.float(), right.float())
                triangle = triangle / structural.sum(1).clamp_min(1)[:, None, None, None]
            geometry_pairs = geometry_pairs + (
                self.geometry_out(triangle.to(geometry.dtype))
                * pair_quality[..., None] * self.geometry_scale.tanh()
            )
        pairs = self.pair_ff(pairs, valid)
        # Keep explicit geometry separate from the learned residue-pair state.
        # Apply quality AFTER normalization/projection, only on the geometry term.
        bias = (self.bias(self.pair_norm(pairs))
                + self.geometry_bias(self.geometry_norm(geometry_pairs).tanh()) * pair_quality[..., None])
        bias = bias.permute(0, 3, 1, 2)
        residues = self.ff(residues + self.read(residues, residues, mask, mask, bias), mask)
        return residues, pairs, geometry_pairs


class GeometryContextBlock(nn.Module):
    """Contextualize angular/DSSP nodes with distance- and edge-biased attention."""

    def __init__(self, dim, pair_dim, heads, dropout):
        super().__init__()
        self.edge = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, heads, bias=False))
        self.read = Attention(dim, heads, dropout)
        self.message_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.scale = nn.Parameter(torch.full((dim,), .1))
        self.ff = FeedForward(dim, 2, dropout)

    def forward(self, geometry, mask, pairs, pair_quality, reliability):
        bias = (self.edge(pairs) * pair_quality[..., None]).permute(0, 3, 1, 2)
        bias = bias + reliability.float().clamp_min(1e-4).log()[:, None, None, :]
        message = self.read(geometry, geometry, mask, mask, bias)
        # Query quality and mean source quality survive the read's normalization.
        message = (self.message_norm(message) * reliability[..., None]
                   * structure_strength(reliability, mask))
        return self.ff(geometry + self.scale.tanh() * message, mask)


class MotifPyramid(nn.Module):
    """Retain residue memory plus overlapping, mask-normalized 3/7/15-residue motifs."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.widths = (3, 7, 15)
        self.norm = nn.LayerNorm(dim)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(dim, dim, width, padding=width // 2, groups=dim, bias=False)
            for width in self.widths
        ])
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout))
            for _ in self.widths
        ])
        self.scale_embedding = nn.Parameter(torch.randn(4, dim) * .02)

    def forward(self, residues, mask):
        tokens = [(residues + self.scale_embedding[0]) * mask[..., None]]
        masks = [mask]
        values = (self.norm(residues) * mask[..., None]).transpose(1, 2)
        for index, (width, convolution, projection) in enumerate(zip(
                self.widths, self.convolutions, self.projections), 1):
            stride = max(1, width // 2)
            count = F.avg_pool1d(mask[:, None].float(), width, stride, width // 2,
                                 count_include_pad=True) * width
            pooled = F.avg_pool1d(values, width, stride, width // 2,
                                  count_include_pad=True) * width / count.clamp_min(1)
            # Window centers must belong to the peptide: batch padding must not
            # create additional copies of a short peptide's terminal motif.
            present = (count[:, 0] > 0) & mask[:, ::stride]
            ordered = F.gelu(convolution(values))[:, :, ::stride]
            local = torch.cat([pooled, ordered], 1).transpose(1, 2)
            tokens.append((projection(local) + self.scale_embedding[index]) * present[..., None])
            masks.append(present)
        # Equal prior mass per resolution, independent of its valid token count.
        weights = [present.float() / present.sum(1, keepdim=True).clamp_min(1) for present in masks]
        return torch.cat(tokens, 1), torch.cat(masks, 1), torch.cat(weights, 1)


class QualityRead(nn.Module):
    """Read with positive key priors (structure quality or inverse scale size).

    Structure callers also apply absolute quality after output normalization.
    """

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.read = Attention(dim, heads, dropout)

    def forward(self, query, memory, query_mask, memory_mask, reliability):
        bias = reliability.float().clamp_min(1e-4).log()[:, None, None, :]
        bias = bias.expand(-1, self.heads, query.shape[1], -1)
        return self.read(query, memory, query_mask, memory_mask, bias)


def structure_strength(reliability, mask):
    """Mean quality of available structure tokens; absent structure contributes zero."""
    strength = (reliability * mask).sum(1) / mask.sum(1).clamp_min(1)
    return strength[:, None, None]


class LabelPairRead(nn.Module):
    """Label-conditioned ordered pairs, read without expanding pair values to dim."""

    def __init__(self, dim, pair_dim, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.residue_norm = nn.LayerNorm(dim)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.geometry_norm = nn.LayerNorm(pair_dim)
        self.queries = nn.Linear(dim, 3 * pair_dim, bias=False)
        self.left = nn.Linear(dim, pair_dim, bias=False)
        self.right = nn.Linear(dim, pair_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(2 * dim + pair_dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(dim, elementwise_affine=False),
        )
        self.scale = math.sqrt(3 * pair_dim)

    def forward(self, labels, residues, pairs, mask, geometry_pairs, pair_quality, interactions=True):
        queries = self.queries(self.query_norm(labels)).chunk(3, -1)
        residues = self.residue_norm(residues)
        pairs = (self.pair_norm(pairs)
                 + self.geometry_norm(geometry_pairs).tanh() * pair_quality[..., None])
        if not interactions:
            # Keep endpoint scores/values and the same temperature/output size.
            # This ablates only explicit pair-state evidence in the decoder.
            pairs = torch.zeros_like(pairs)
        left, right = self.left(residues), self.right(residues)
        valid = mask[:, :, None] & mask[:, None, :]
        # A self-pair is not evidence of interaction between two residues.
        valid = valid & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        with torch.autocast(device_type=labels.device.type, enabled=False):
            score = (torch.einsum("bcp,bip->bci", queries[0].float(), left.float())[..., None]
                     + torch.einsum("bcp,bjp->bcj", queries[1].float(), right.float())[:, :, None, :]
                     + torch.einsum("bcp,bijp->bcij", queries[2].float(), pairs.float())) / self.scale
            score = score.masked_fill(~valid[:, None], torch.finfo(score.dtype).min)
            weights = score.flatten(2).softmax(-1).reshape_as(score) * valid[:, None]
            # Marginals read endpoints; only the narrow pair state is read per pair.
            # A single-residue sequence has no valid pair and produces zero evidence.
            left_value = torch.bmm(weights.sum(3), residues.float())
            right_value = torch.bmm(weights.sum(2), residues.float())
            pair_value = torch.einsum("bcij,bijp->bcp", weights, pairs.float())
            values = torch.cat([left_value, right_value, pair_value], -1)
        evidence = self.output(values.to(labels.dtype))
        present = valid.flatten(1).any(1)[:, None, None]
        return evidence * present, weights


class LatentFusionBlock(nn.Module):
    """Separate reads prevent long residue streams from crowding out property tokens."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.motif_read = QualityRead(dim, heads, dropout)
        self.property_read = Attention(dim, heads, dropout)
        self.structure_read = QualityRead(dim, heads, dropout)
        self.evidence_norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.combine = nn.Sequential(nn.Linear(3 * dim, dim), nn.Dropout(dropout))
        self.process = TransformerBlock(dim, heads, dropout)

    def forward(self, latent, latent_mask, memories, masks, reliability, motif_weights):
        evidence = [self.motif_read(latent, memories[0], latent_mask, masks[0], motif_weights),
                    self.property_read(latent, memories[1], latent_mask, masks[1]),
                    self.structure_read(latent, memories[2], latent_mask, masks[2], reliability)]
        evidence = [norm(x) for norm, x in zip(self.evidence_norms, evidence)]
        # Uniform low reliability cancels inside softmax. Scale after LN so the
        # absolute source quality survives both attention and normalization.
        evidence[2] = evidence[2] * structure_strength(reliability, masks[2])
        return self.process(latent + self.combine(torch.cat(evidence, -1)), latent_mask)


class SetDecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout, pair_dim):
        super().__init__()
        self.context = Attention(dim, heads, dropout)
        self.latent_read = Attention(dim, heads, dropout)
        self.motif_read = QualityRead(dim, heads, dropout)
        self.property_read = Attention(dim, heads, dropout)
        self.structure_read = QualityRead(dim, heads, dropout)
        self.evidence_norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(4)])
        self.combine = nn.Linear(4 * dim, dim)
        self.pair_read = LabelPairRead(dim, pair_dim, dropout)
        self.pair_scale = nn.Parameter(torch.full((dim,), .1))
        self.context_scale = nn.Parameter(torch.full((dim,), .1))
        self.ff = FeedForward(dim, 4, dropout)

    def forward(self, labels, label_mask, latent, latent_mask, memories, masks, reliability,
                residues, pairs, residue_mask, geometry_pairs, pair_quality, motif_weights,
                label_context=True, pair_readout=True, fine_evidence=True, pair_interactions=True):
        evidence = [self.latent_read(labels, latent, label_mask, latent_mask)]
        if fine_evidence:
            evidence.extend([self.motif_read(labels, memories[0], label_mask, masks[0], motif_weights),
                             self.property_read(labels, memories[1], label_mask, masks[1]),
                             self.structure_read(labels, memories[2], label_mask, masks[2], reliability)])
        else:
            evidence.extend([torch.zeros_like(labels) for _ in range(3)])
        evidence = [norm(x) for norm, x in zip(self.evidence_norms, evidence)]
        evidence[3] = evidence[3] * structure_strength(reliability, masks[2])
        reads = torch.cat(evidence, -1)
        labels = labels + self.combine(reads)
        pair_attention = None
        if pair_readout:
            pair_evidence, pair_attention = self.pair_read(
                labels, residues, pairs, residue_mask, geometry_pairs, pair_quality, pair_interactions)
            labels = labels + self.pair_scale.tanh() * pair_evidence
        if label_context:
            # Relation context is conditioned on the peptide before the first
            # label interaction; its residual starts small but remains trainable.
            labels = labels + self.context_scale.tanh() * self.context(labels, labels, label_mask, label_mask)
        return self.ff(labels, label_mask), pair_attention


class HelixSet(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=384,
                 heads=8, dropout=.2, sequence_depth=3, property_depth=2,
                 pair_dim=48, pair_depth=4, latent_count=32, latent_depth=6,
                 decoder_depth=3, triangles=True, label_context=True,
                 pair_readout=True, fine_evidence=True, geometry_depth=4,
                 geometry_context=True, geometry_quality=True, distance_multiscale=True,
                 scale_balance=True, pair_interactions=True, structure_evidence=True):
        super().__init__()
        self.statistics = statistics
        self.triangles, self.label_context = triangles, label_context
        self.pair_readout, self.fine_evidence = pair_readout, fine_evidence
        self.geometry_context, self.geometry_quality = geometry_context, geometry_quality
        self.distance_multiscale = distance_multiscale
        self.scale_balance, self.pair_interactions = scale_balance, pair_interactions
        self.structure_evidence = structure_evidence
        self.sequence = SequenceEncoder(input_dim, vocab_size, dim, heads, dropout, sequence_depth)
        self.properties = PropertyEncoder(dim, heads, dropout, property_depth)
        self.geometry = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.LayerNorm(dim))
        self.dssp = nn.Embedding(10, dim)
        self.structure_global = nn.Sequential(nn.Linear(50, dim), nn.GELU(), nn.LayerNorm(dim))
        self.geometry_mix = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.Tanh())
        # Keep sequence separation and explicit geometry in separate pair streams.
        self.pair_input = nn.Linear(2, pair_dim)
        self.pair_geometry = nn.Linear(32, pair_dim)
        self.geometry_blocks = nn.ModuleList([
            GeometryContextBlock(dim, pair_dim, heads, dropout) for _ in range(geometry_depth)
        ])
        self.pair_blocks = nn.ModuleList([
            ResiduePairBlock(dim, pair_dim, heads, dropout, incoming=(i % 2 == 1)) for i in range(pair_depth)
        ])
        self.pyramid = MotifPyramid(dim, dropout)
        self.latents = nn.Parameter(torch.randn(latent_count, dim) * .02)
        self.latent_blocks = nn.ModuleList([
            LatentFusionBlock(dim, heads, dropout) for _ in range(latent_depth)
        ])
        self.label_queries = nn.Parameter(torch.randn(num_labels, dim) * .02)
        self.decoder = nn.ModuleList([SetDecoderBlock(dim, heads, dropout, pair_dim) for _ in range(decoder_depth)])
        self.output_norm = nn.LayerNorm(dim)
        self.label_weights = nn.Parameter(torch.randn(num_labels, dim) / math.sqrt(dim))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        prior = torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.label_bias.copy_(torch.logit(prior))
        # A pooled latent head teaches the bottleneck without reconstruction losses.
        self.latent_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_labels))
        with torch.no_grad():
            self.latent_classifier[-1].bias.copy_(torch.logit(prior))

    def pair_features(self, batch, structural, reliability):
        mask = batch["mask"]
        observed = structural[:, :, None] & structural[:, None, :]
        coords = batch["coords"].float().masked_fill(~structural[..., None], 0)
        with torch.autocast(device_type=coords.device.type, enabled=False):
            distances = torch.cdist(coords, coords)
            centers = torch.linspace(0, 20, 16, device=coords.device)
            near = torch.exp(-((distances[..., None] - centers) / 1.5).square())
            far_centers = torch.linspace(24, 56, 8, device=coords.device)
            far = torch.exp(-((distances[..., None] - far_centers) / 5.).square())
            # Continuous tails still distinguish distances beyond the last RBF.
            tails = torch.stack([torch.log1p(distances) / math.log(65),
                                 (distances - 56).clamp_min(0) / (distances + 1)], -1)
            if not self.distance_multiscale:
                far, tails = torch.zeros_like(far), torch.zeros_like(tails)
            distances_encoded = torch.cat([near, far, tails], -1) * observed[..., None]
        position = torch.arange(mask.shape[1], device=mask.device)
        separation = (position[:, None] - position[None, :]).float() / 74
        separation = separation[None, ..., None].expand(mask.shape[0], -1, -1, -1)
        kind = batch["edge_type"].masked_fill(~observed, 0)
        known = batch["confidence_known"] & structural
        confidence = torch.where(known, batch["geometry"][..., 9].clamp(0, 1), 0)
        pair_known = known[:, :, None] & known[:, None, :]
        pair_confidence = confidence[:, :, None] * confidence[:, None, :]
        features = torch.cat([distances_encoded, F.one_hot(kind, 3).float(),
                              observed[..., None].float(), pair_known[..., None].float(),
                              pair_confidence[..., None]], -1)
        valid = mask[:, :, None] & mask[:, None, :]
        # Geometry triangles represent relations between distinct observed nodes.
        observed = observed & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        pair_quality = reliability[:, :, None] * reliability[:, None, :] * observed
        pairs = self.pair_input(torch.cat([separation, separation.abs()], -1)) * valid[..., None]
        geometry_pairs = self.pair_geometry(features) * observed[..., None]
        return pairs, geometry_pairs, pair_quality

    def forward(self, batch, return_evidence=False):
        if not self.structure_evidence:
            # Remove every explicit structure route, including global descriptors.
            # Sequence-derived physicochemical features remain available.
            batch = dict(batch)
            batch["struct_mask"] = torch.zeros_like(batch["struct_mask"])
            batch["struct_global_mask"] = torch.zeros_like(batch["struct_global_mask"])
        seq, seq_mask = self.sequence(batch)
        prop, prop_mask = self.properties(batch)
        mask = batch["mask"]
        structural = batch["struct_mask"] & mask
        geometry = (self.geometry(batch["geometry"].masked_fill(~structural[..., None], 0))
                    + self.dssp(batch["dssp"])) * structural[..., None]
        known = batch["confidence_known"] & structural
        # Unknown confidence is explicitly neutral, not fabricated low confidence.
        reliability = torch.where(known, .3 + .7 * batch["geometry"][..., 9].clamp(0, 1), 1.)
        if not self.geometry_quality:
            reliability = torch.ones_like(reliability)
        reliability = reliability * structural
        pairs, geometry_pairs, pair_quality = self.pair_features(batch, structural, reliability)
        if self.geometry_context:
            for block in self.geometry_blocks:
                geometry = block(geometry, structural, geometry_pairs, pair_quality, reliability)
        residues = seq[:, :-1]
        residues = residues + self.geometry_mix(torch.cat([residues, geometry], -1)) * geometry * reliability[..., None]
        for block in self.pair_blocks:
            residues, pairs, geometry_pairs = block(
                residues, pairs, mask, geometry_pairs, pair_quality, structural, self.triangles)
        motifs, motif_mask, motif_weights = self.pyramid(residues, mask)
        # Preserve the sequence encoder's global chemical annotation token.
        motifs = torch.cat([motifs, seq[:, -1:]], 1)
        motif_mask = torch.cat([motif_mask, seq_mask[:, -1:]], 1)
        # The global chemical token is a fifth resolution with one valid token.
        motif_weights = torch.cat([motif_weights, seq_mask[:, -1:].float()], 1)
        if not self.scale_balance:
            motif_weights = motif_mask.float()
        structure_present = structural.any(1, keepdim=True)
        global_structure = self.structure_global(torch.cat([
            batch["struct_global"].masked_fill(~batch["struct_global_mask"], 0),
            batch["struct_global_mask"].float()], -1))
        global_structure = global_structure + (
            (geometry * reliability[..., None]).sum(1) / reliability.sum(1, keepdim=True).clamp_min(1e-4))
        geometry = torch.cat([geometry, global_structure[:, None] * structure_present[..., None]], 1)
        geometry_mask = torch.cat([structural, structure_present], 1)
        global_reliability = (reliability * structural).sum(1, keepdim=True) / structural.sum(1, keepdim=True).clamp_min(1)
        reliability = torch.cat([reliability, global_reliability], 1)
        memories, masks = (motifs, prop, geometry), (motif_mask, prop_mask, geometry_mask)
        latent = self.latents[None].expand(mask.shape[0], -1, -1)
        latent_mask = torch.ones(latent.shape[:2], dtype=torch.bool, device=mask.device)
        for block in self.latent_blocks:
            latent = block(latent, latent_mask, memories, masks, reliability, motif_weights)
        labels = self.label_queries[None].expand(mask.shape[0], -1, -1)
        label_mask = torch.ones(labels.shape[:2], dtype=torch.bool, device=mask.device)
        intermediate = []
        for block in self.decoder:
            labels, pair_attention = block(
                labels, label_mask, latent, latent_mask, memories, masks, reliability,
                residues, pairs, mask, geometry_pairs, pair_quality, motif_weights,
                self.label_context, self.pair_readout, self.fine_evidence, self.pair_interactions,
            )
            intermediate.append((self.output_norm(labels) * self.label_weights).sum(-1) + self.label_bias)
        output = {"logits": intermediate[-1], "direct": self.latent_classifier(latent.mean(1)),
                  "decoder_logits": intermediate[:-1]}
        if return_evidence:
            output["pair_attention"] = pair_attention
        return output

    def compute_loss(self, batch, settings, return_output=False):
        # Only modality dropout by default; no competing descriptor reconstruction.
        view, _ = training_views(batch, settings.get("modality_dropout", .1), 0.)
        output = self(view)
        def classification(logits):
            return observed_label_loss(logits, batch["y"], self.statistics, settings.get("unlabeled_weight", .3))
        terms = {"classification": classification(output["logits"]),
                 "direct": classification(output["direct"])}
        terms["branch"] = (torch.stack([classification(x) for x in output["decoder_logits"]]).mean()
                           if output["decoder_logits"] else output["logits"].sum() * 0)
        total = terms["classification"] + settings.get("direct", .2) * terms["direct"] + settings.get("branch", .1) * terms["branch"]
        return (total, terms, output) if return_output else (total, terms)
