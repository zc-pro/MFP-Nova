"""HH-SetReread: sample-dependent function-axis energy over Fine39 label sets."""
import itertools
import math

import numpy as np
import torch
from torch import nn

from .theraset import TheraSet
from ..consistency import bernoulli_rdrop
from ..multimodal import function_projection, independent_functions
from ..research_layers import observed_label_loss, training_views, unlabeled_weights


def axis_statistics(source):
    labels = ("antibacterial antifungal antiparasitic antiviral antimicrobial immunoactive "
              "anticancer antihypertensive thrombolytic neuropeptide antioxidant metabolic_regulatory "
              "growth_regeneration cell_communication drug_delivery").split()
    y = source.data.y[source.data.indices("train")]
    support = independent_functions(y, source.statistics["function_projection"]).sum(0)
    weights = np.sqrt(max(support.max(), 1) / np.maximum(support, 1)).clip(1, 4)
    return {"labels": labels, "positive_weights": (weights / weights.mean()).tolist(), "support": support.tolist()}


class HHSetReread(nn.Module):
    def __init__(self, statistics, axis_statistics, relation="bilinear", lambda_pair=4.,
                 fixed_overlap=False, **backbone_args):
        super().__init__()
        if backbone_args["num_labels"] != 39 or not backbone_args.get("evidence_decoder", True):
            raise ValueError("HH-SetReread requires the full Fine39 TheraSet evidence model")
        if relation not in ("off", "shared", "bilinear", "mlp"):
            raise ValueError("HH relation must be off, shared, bilinear or mlp")
        self.backbone = TheraSet(statistics=statistics, **backbone_args)
        self.statistics, self.axis_statistics = statistics, axis_statistics
        self.relation, self.lambda_pair, self.fixed_overlap = relation, lambda_pair, fixed_overlap
        self.backbone_frozen = False
        projection = torch.tensor(function_projection(statistics["labels"], "fine39"))
        self.register_buffer("projection", projection)
        pairs = [(i, j) for i, j in itertools.combinations(range(15), 2) if not (i < 4 and j == 4)]
        self.register_buffer("pair_left", torch.tensor([i for i, _ in pairs]))
        self.register_buffer("pair_right", torch.tensor([j for _, j in pairs]))
        self.register_buffer("bits", 1 << torch.arange(39))
        dim = backbone_args.get("dim", 256)
        self.query = nn.Parameter(torch.randn(15, 128) * .02)
        self.keys = nn.Linear(dim, 128, bias=False)
        self.axis_weight = nn.Parameter(torch.randn(15, dim) / math.sqrt(dim))
        self.axis_bias = nn.Parameter(torch.zeros(15))
        if relation != "off":
            if not fixed_overlap:
                self.compat_residue = nn.Linear(dim, 32, bias=False)
                self.compat_query = nn.Linear(128, 32, bias=False)
                self.compatibility = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
            self.shared_projection = nn.Linear(dim, 128, bias=False)
            self.shared_readout = nn.Linear(128, 1, bias=False)
            nn.init.zeros_(self.shared_readout.weight)
        if relation in ("bilinear", "mlp"):
            self.private_projection = nn.Linear(dim, 128, bias=False)
            self.position_gate = nn.Linear(3, 1)
            nn.init.zeros_(self.position_gate.weight)
            nn.init.zeros_(self.position_gate.bias)
            if relation == "bilinear":
                self.private_u = nn.Linear(128, 32, bias=False)
                self.private_v = nn.Linear(128, 32, bias=False)
                self.private_readout = nn.Linear(32, 1, bias=False)
            else:
                self.private_readout = nn.Sequential(nn.Linear(384, 21), nn.GELU(), nn.Linear(21, 1, bias=False))
            nn.init.zeros_((self.private_readout if relation == "bilinear" else self.private_readout[-1]).weight)

    def freeze_backbone(self, frozen):
        self.backbone_frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    @staticmethod
    def effective(raw):
        generic = raw[..., 4:5] & ~raw[..., :4].any(-1, keepdim=True)
        return torch.cat((raw[..., :4], generic, raw[..., 5:]), dim=-1)

    def axes(self, sets):
        return self.effective((sets.float() @ self.projection) > 0)

    def relation_energy(self, axes, psi):
        active_pairs = axes[..., self.pair_left] & axes[..., self.pair_right]
        return (active_pairs * psi[:, None]).sum(-1) * (self.lambda_pair / 14)

    def conditional_logits(self, unary, psi):
        base = unary >= 0
        count = base.float() @ self.projection
        without = count[:, None] - base[..., None] * self.projection
        present = self.effective(without + self.projection > 0)
        absent = self.effective(without > 0)
        return unary + (self.relation_energy(present, psi) - self.relation_energy(absent, psi))

    def pair_potentials(self, residues, attention, mask):
        left, right = self.pair_left, self.pair_right
        shared = torch.minimum(attention[:, left], attention[:, right])
        if not self.fixed_overlap:
            # Process small groups of pairs to avoid a full [B,101,L,96] concat.
            condition = self.compat_residue(residues)[:, None] * self.compat_query(self.query).sigmoid()[None, :, None]
            compatibility = []
            for i, j in zip(left.split(16), right.split(16)):
                a, b = condition[:, i], condition[:, j]
                features = torch.cat((a+b, (a-b).abs(), a*b), dim=-1)
                compatibility.append(self.compatibility(features).squeeze(-1).sigmoid())
            shared = shared * torch.cat(compatibility, dim=1)
        mass = shared.sum(-1)
        shared_mean = (shared @ self.shared_projection(residues)) / mass.clamp_min(1e-6)[..., None]
        psi = mass * self.shared_readout(shared_mean).squeeze(-1).tanh()
        if self.relation == "shared":
            return psi

        private_left = attention[:, left] - shared
        private_right = attention[:, right] - shared
        rho = (1 - mass).clamp_min(0)
        denom = rho.clamp_min(1e-6)[..., None]
        private = self.private_projection(residues)
        axis_pool = attention @ private
        shared_private_pool = shared @ private
        x = (axis_pool[:, left] - shared_private_pool) / denom
        y = (axis_pool[:, right] - shared_private_pool) / denom
        positions = torch.arange(mask.shape[1], device=mask.device) / (mask.sum(-1, keepdim=True)-1).clamp_min(1)
        positions = positions[:, None]
        px, py = private_left / denom, private_right / denom
        mx, my = (px * positions).sum(-1), (py * positions).sum(-1)
        vx = (px * (positions - mx[..., None]).square()).sum(-1)
        vy = (py * (positions - my[..., None]).square()).sum(-1)
        gate = self.position_gate(torch.stack(((mx-my).abs(), vx+vy, (vx-vy).abs()), -1)).squeeze(-1).sigmoid()
        if self.relation == "bilinear":
            features = self.private_u(x)*self.private_v(y) + self.private_u(y)*self.private_v(x)
        else:
            features = torch.cat((x+y, (x-y).abs(), x*y), dim=-1)
        return psi + rho * gate * self.private_readout(features).squeeze(-1).tanh()

    def forward(self, batch):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.backbone_frozen):
            output = self.backbone(batch, return_residues=True)
        residues = output.pop("residues")
        mask = batch["mask"]
        with torch.autocast(residues.device.type, enabled=False):
            residues = residues.float()
            scores = self.keys(residues) @ self.query.T / math.sqrt(128)
            attention = scores.transpose(1, 2).masked_fill(~mask[:, None], -torch.inf).softmax(-1)
            axis_logits = ((attention @ residues) * self.axis_weight).sum(-1) + self.axis_bias
            unary = output["logits"].float()
            psi = (unary.new_zeros((len(unary), len(self.pair_left))) if self.relation == "off"
                   else self.pair_potentials(residues, attention, mask))
            logits = self.conditional_logits(unary, psi)
        return {**output, "logits": logits, "unary": unary, "psi": psi, "axis_logits": axis_logits}

    @torch.no_grad()
    def candidates(self, unary, conditional):
        base = unary >= 0
        available = ~base
        first = unary.masked_fill(base, -torch.inf).argsort(descending=True, stable=True)[:, :2]
        remaining = available.scatter(1, first, False)
        second = conditional.masked_fill(~remaining, -torch.inf).argsort(descending=True, stable=True)[:, :2]
        additions = torch.cat((first, second), dim=-1)
        add_valid = torch.cat((available.gather(1, first), remaining.gather(1, second)), dim=-1)
        deletion = (-conditional).masked_fill(~base, -torch.inf).argsort(descending=True, stable=True)[:, :2]
        drop_valid = base.gather(1, deletion)
        labels = torch.arange(unary.shape[-1], device=unary.device)
        add = (additions[..., None] == labels) & add_valid[..., None]
        drop = (deletion[..., None] == labels) & drop_valid[..., None]
        patterns = [base[:, None], base[:, None] | add, base[:, None] & ~drop]
        swaps = (base[:, None, None] & ~drop[:, :, None]) | add[:, None]
        patterns.append(swaps.flatten(1, 2))
        first, second = zip(*itertools.combinations(range(4), 2))
        patterns.extend((base[:, None] | add[:, first, :] | add[:, second, :],
                         base[:, None] & ~drop.any(1, keepdim=True), (conditional >= 0)[:, None]))
        sets = torch.cat(patterns, dim=1)
        codes = (sets * self.bits).sum(-1)
        duplicate = (codes[:, :, None] == codes[:, None, :]).tril(-1).any(-1)
        return sets, ~duplicate

    def set_energy(self, sets, unary, psi):
        return (sets * unary[:, None]).sum(-1) + self.relation_energy(self.axes(sets), psi)

    def ranking_loss(self, output, target, mode, unknown_weight):
        sets, unique = self.candidates(output["unary"], output["logits"])
        matches = (sets * target[:, None]).sum(-1)
        weights = (unlabeled_weights(target, self.statistics, unknown_weight) if mode == "weak"
                   else torch.ones_like(target))
        denominator = target.sum(-1, keepdim=True) + (sets * ((1-target)*weights)[:, None]).sum(-1)
        utility = matches / denominator.clamp_min(1e-6)
        energy = self.set_energy(sets, output["unary"], output["psi"])
        utility_gap = utility[:, :, None] - utility[:, None, :]
        energy_gap = energy[:, :, None] - energy[:, None, :]
        weight = utility_gap.abs() * (unique[:, :, None] & unique[:, None, :]).triu(1)
        loss = torch.nn.functional.softplus(-utility_gap.sign() * energy_gap)
        return ((weight*loss).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1e-12)).mean()

    def compute_loss(self, batch, settings, return_output=False):
        target = batch["y"]
        weak = settings.get("unlabeled_weight", .3)
        view = (training_views(batch, settings.get("modality_dropout", .1), 0.)[0]
                if self.training and not self.backbone_frozen else batch)
        axis_target = self.axes(target).float()
        ranking = settings.get("hh_ranking", "record")
        set_weight = settings.get("hh_set", .05) if ranking != "none" else 0.

        def supervised(prediction):
            terms = self.backbone.supervised_terms({**prediction, "logits": prediction["unary"]}, target, weak)
            terms["hh_fine"] = observed_label_loss(prediction["logits"], target, self.statistics, weak)
            terms["hh_axis"] = observed_label_loss(prediction["axis_logits"], axis_target, self.axis_statistics, .3)
            if set_weight:
                terms["hh_set"] = self.ranking_loss(prediction, target, ranking, weak)
            terms["hh_relation"] = prediction["psi"].square().mean()
            return terms

        output = self(view)
        terms = supervised(output)
        rdrop = settings.get("rdrop", 0.) if self.training and not self.backbone_frozen else 0.
        agreement = output["unary"].new_zeros(())
        if rdrop:
            second = self(view)
            other = supervised(second)
            terms = {key: (value + other[key]) * .5 for key, value in terms.items()}
            agreement = bernoulli_rdrop(output["unary"], second["unary"])
        terms["rdrop"] = agreement
        total = (terms["classification"] + settings.get("direct", .2)*terms["direct"]
                 + settings.get("evidence", .1)*terms["evidence"] + rdrop*agreement)
        additional = settings.get("hh_fine", .5)*terms["hh_fine"] + settings.get("hh_axis", .1)*terms["hh_axis"]
        if set_weight:
            additional = additional + set_weight*terms["hh_set"]
        total = total + settings.get("hh_ramp", 1.)*additional + settings.get("hh_relation", 1e-4)*terms["hh_relation"]
        return (total, terms, output) if return_output else (total, terms)

    @torch.no_grad()
    def decode(self, unary, conditional, psi, exact=False):
        base = unary >= 0
        if exact:
            return self.exact_decode(unary, psi)
        sets, unique = self.candidates(unary, conditional)
        energy = self.set_energy(sets, unary, psi).masked_fill(~unique, -torch.inf)
        edits = (sets != base[:, None]).sum(-1)
        codes = (sets * self.bits).sum(-1)
        _, code = best_set(energy, edits, codes)
        return (code[:, None] & self.bits) != 0

    def exact_decode(self, unary, psi):
        # Optimize each nonempty group independently, keeping zero logits in T0.
        membership = self.projection.T.bool()
        base = unary >= 0
        selected = base[:, None] & membership
        top = unary[:, None].masked_fill(~membership, -torch.inf).argmax(-1)
        fallback = top[..., None] == torch.arange(39, device=unary.device)
        selected = torch.where(selected.any(-1, keepdim=True), selected, fallback)
        values = (selected * unary[:, None]).sum(-1)
        group_codes = (selected * self.bits).sum(-1)
        off_edits = (base.float() @ self.projection).long()
        on_edits = (selected & ~base[:, None]).sum(-1)
        left, right = self.pair_left, self.pair_right
        nongeneric = (left != 4) & (right != 4)
        generic = ~nongeneric
        neighbors = torch.where(left[generic] == 4, right[generic], left[generic])
        axes = torch.tensor([*range(4), *range(5, 15)], device=unary.device)
        powers = 1 << torch.arange(14, device=unary.device)
        alpha = self.lambda_pair / 14
        best_score = unary.new_full((len(unary),), -torch.inf)
        best_edits = torch.full_like(best_score, 40, dtype=torch.long)
        best_code = torch.zeros_like(best_edits)
        for start in range(0, 16384, 1024):
            states = torch.zeros((1024, 15), device=unary.device, dtype=torch.bool)
            states[:, axes] = (torch.arange(start, start+1024, device=unary.device)[:, None] & powers) != 0
            active_pairs = states[:, left[nongeneric]] & states[:, right[nongeneric]]
            score = values @ states.float().T + alpha * (psi[:, nongeneric] @ active_pairs.float().T)
            gain = values[:, 4:5] + alpha * (psi[:, generic] @ states[:, neighbors].float().T) * (~states[:, :4].any(-1))
            use_generic = (gain > 0) | ((gain == 0) & (off_edits[:, 4:5] > 0))
            score = score + gain.clamp_min(0.)
            # All counts are <40 and exactly representable under this matrix product.
            edits = off_edits.sum(-1, keepdim=True) + ((on_edits-off_edits).float() @ states.float().T).long()
            edits += use_generic * (on_edits[:, 4:5]-off_edits[:, 4:5])
            # Int64 preserves Fine39 set codes, which exceed FP32's exact integer range.
            codes = (group_codes[:, None] * states).sum(-1) + use_generic * group_codes[:, 4:5]
            chosen, code = best_set(score, edits, codes)
            index = (codes == code[:, None]).long().argmax(-1, keepdim=True)
            distance = edits.gather(1, index).squeeze(-1)
            better = (chosen > best_score) | ((chosen == best_score) &
                      ((distance < best_edits) | ((distance == best_edits) & (code < best_code))))
            best_score = torch.where(better, chosen, best_score)
            best_edits = torch.where(better, distance, best_edits)
            best_code = torch.where(better, code, best_code)
        return (best_code[:, None] & self.bits) != 0


def best_set(energy, edits, codes):
    best = energy.amax(-1, keepdim=True)
    tied = energy == best
    distance = edits.masked_fill(~tied, 40)
    tied &= distance == distance.amin(-1, keepdim=True)
    return best.squeeze(-1), codes.masked_fill(~tied, 1 << 39).amin(-1)
