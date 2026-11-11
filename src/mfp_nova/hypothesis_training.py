"""Training phases and train-only assets for the sequence hypothesis models."""

import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .hypothesis_data import LengthBatches, study_loader
from .models.hypothesis_backbone import SequenceBackbone
from .models.hypothesis_evidence import evidence_auxiliary, evidence_selection
from .models.hypothesis_relations import RankingPairSampler, paired_rank_loss
from .models.hypothesis_set import cardinality_nll, count_support
from .trainer import seed_everything


def element_bce(logits, targets):
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")


def weighted_bce(logits, targets, weights):
    return (element_bce(logits, targets) * weights).sum()


class SequenceTraining:
    def __init__(self, config, data, train_indices, run=None):
        self.config, self.data, self.train_indices = config, data, train_indices
        self.run = run
        self.settings = config["train"]
        self.device = torch.device(self.settings["device"])
        self.precision = self.settings["precision"]
        if self.device.type != "cuda" and self.precision != "fp32":
            raise ValueError("Use FP32 for CPU training")
        self.seed = self.settings["seed"]
        self.batch = self.settings["batch_size"]
        self.micro_batch = math.ceil(self.batch / self.settings["accumulation_steps"])
        self.phases = []
        self.initialized_from_diagnostic = False
        self.rng = np.random.default_rng(self.seed)
        self.weights = np.where(data.observed, 1., self.settings["unlabeled_weight"]).astype(np.float32)
        self.weights[~data.observed & data.targets.astype(bool)] = 0.  # Conflicting positive records are excluded.
        self.folds = None

    def make_backbone(self, seed):
        seed_everything(seed)
        return SequenceBackbone(len(self.data.labels), **self.config["model"]).to(self.device)

    def loader(self, indices, shuffle=False, seed=None):
        return study_loader(self.data, indices, self.batch, shuffle=shuffle,
                            seed=self.seed if seed is None else seed, workers=self.settings["num_workers"])

    def fit(self, module, phase, batches, loss_for_batch, *, kind="head"):
        epoch_key, lr_key = {"full": ("epochs", "learning_rate"), "head": ("head_epochs", "head_learning_rate"),
                             "tune": ("finetune_epochs", "finetune_learning_rate")}[kind]
        epochs = self.settings[epoch_key]
        parameters = [param for param in module.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.settings[lr_key], weight_decay=self.settings["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")
        cap = self.settings.get("max_steps")
        total = epochs * len(batches)
        total = min(total, cap) if cap else total
        warmup = math.ceil(total * self.settings["warmup_fraction"])

        def scale(step):
            if step < warmup:
                return (step + 1) / warmup
            return .5 * (1 + math.cos(math.pi * min(step - warmup, total - warmup) / max(1, total - warmup)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        started = time.perf_counter()
        updates = 0
        for epoch in range(epochs):
            module.train()
            epoch_loss = torch.zeros((), device=self.device)
            count = 0
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                has_loss = False
                # Advance the generator under autocast. Backward executes outside it.
                losses = iter(loss_for_batch(batch))
                while True:
                    with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=self.precision != "fp32"):
                        loss = next(losses, None)
                    if loss is None:
                        break
                    has_loss = True
                    scaler.scale(loss).backward()
                    epoch_loss += loss.detach()
                if not has_loss:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, self.settings["gradient_clip"],
                                               error_if_nonfinite=self.precision != "fp16")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                count += 1
                updates += 1
                if cap and updates >= cap:
                    break
            print(f"{phase} epoch {epoch + 1}/{epochs} loss={epoch_loss.item() / max(1, count):.5f} steps={count}", flush=True)
            if cap and updates >= cap:
                break
        self.phases.append({"phase": phase, "epochs": epoch + 1, "updates": updates,
                            "seconds": time.perf_counter() - started})
        module.eval()

    def labels(self, rows):
        return tuple(torch.as_tensor(values[rows], device=self.device) for values in
                     (self.data.targets, self.data.observed, self.weights))

    def fit_backbone(self, backbone, indices, phase, *, kind="full", seed=None, balanced=False,
                     evidence=None, pairs=None, soft=None, observation=None):
        pair_sampler = RankingPairSampler(pairs) if pairs is not None and len(pairs) else None
        generator = torch.Generator(device=self.device).manual_seed(self.seed if seed is None else seed)

        def loss_for_batch(batch):
            rows, tokens, mask = batch
            tokens, mask = tokens.to(self.device), mask.to(self.device)
            targets, observed, weights = self.labels(rows)
            weights = weights if observation is None else torch.as_tensor(observation[rows], device=self.device)
            if balanced:
                weights = observed.float()
                weights = weights / weights.sum(0).clamp_min(1)
            weight_sum = weights.sum()
            has_supervision = bool(weight_sum)
            active_evidence = evidence in ("attention", "random")
            eligible = ((self.data.observed[rows] & self.data.targets[rows].astype(bool)).any(1)
                        & (self.data.lengths[rows] > 1)).sum() if active_evidence else 0
            eligible_soft = int(soft[2][rows].sum()) if soft is not None else 0
            if has_supervision or eligible_soft:
                for start in range(0, len(rows), self.micro_batch):
                    chunk = slice(start, start + self.micro_batch)
                    encoded = backbone(tokens[chunk], mask[chunk], return_attention=bool(eligible))
                    logits = encoded.logits
                    loss = weighted_bce(logits, targets[chunk], weights[chunk]) / weight_sum.clamp_min(1e-12)
                    if eligible:
                        selected, labels, keep = evidence_selection(encoded.attention, mask[chunk], targets[chunk],
                                                                   observed[chunk], method=evidence, generator=generator)
                        if len(selected):
                            kept = backbone(tokens[chunk][selected], keep).logits
                            loss = loss + evidence_auxiliary(logits, kept, selected, labels, reduction="sum") / eligible
                    if eligible_soft:
                        probabilities, accepted, _ = (torch.as_tensor(value[rows[chunk]], device=self.device) for value in soft)
                        loss = loss + .25 * weighted_bce(logits, probabilities, accepted) / eligible_soft
                    yield loss
            if pair_sampler is not None:
                sampled = pair_sampler.sample(self.rng, size=32)
                for start in range(0, len(sampled), self.micro_batch):
                    positive, negative, labels = sampled[start:start + self.micro_batch].T
                    rows_ = np.concatenate([positive, negative])
                    pair_tokens, pair_mask = self.data.tokens(rows_)
                    pair_logits = backbone(pair_tokens.to(self.device), pair_mask.to(self.device)).logits
                    positions = torch.arange(len(labels), device=self.device)
                    labels = torch.as_tensor(labels, device=self.device)
                    yield .1 * paired_rank_loss(pair_logits[positions, labels],
                                                 pair_logits[positions + len(labels), labels], reduction="sum") / len(sampled)

        self.fit(backbone, phase, self.loader(indices, shuffle=True, seed=seed), loss_for_batch, kind=kind)

    @torch.no_grad()
    def encode(self, backbone, rows, fields=("evidence", "summary", "logits")):
        backbone.eval()
        order = np.argsort(rows)
        encoded = {}
        for indices, tokens, mask in self.loader(rows):
            output = backbone(tokens.to(self.device), mask.to(self.device))
            positions = order[np.searchsorted(rows[order], indices)]
            for field in fields:
                values = getattr(output, field).cpu()
                if field not in encoded:
                    dtype = torch.float16 if field == "evidence" and self.device.type == "cuda" else torch.float32
                    encoded[field] = torch.empty((len(rows), *values.shape[1:]), dtype=dtype)
                encoded[field][positions] = values.to(encoded[field].dtype)
        return encoded

    def fit_cached(self, head, phase, rows, encoded, objective):
        def loss_for_batch(batch):
            positions = np.asarray(batch)
            values = {field: x[positions].to(self.device, dtype=torch.float32) for field, x in encoded.items()}
            loss = objective(positions, rows[positions], **values)
            if loss is not None:
                yield loss

        batches = LengthBatches(self.data.lengths[rows], self.batch, True, self.seed)
        self.fit(head, phase, batches, loss_for_batch)

    def training_folds(self):
        if self.folds is None:
            self.folds = self.data.train_folds(self.train_indices, self.settings["folds"])
        for fold in np.unique(self.folds):
            held = self.folds == fold
            if held.all():
                raise ValueError("Out-of-fold training needs at least two independent families")
            yield int(fold), np.flatnonzero(~held), np.flatnonzero(held)

    def fold_backbone(self, fold, fitted, *, balanced=False):
        seed = self.seed + int(balanced)
        backbone = self.make_backbone(seed)
        directory = self.settings.get("fold_dir")
        path = None
        if directory and not balanced:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{self.data.track}_s{seed}_fold{fold}.pt"
        settings = {key: self.settings.get(key) for key in ("epochs", "batch_size", "accumulation_steps", "precision",
                    "learning_rate", "weight_decay", "warmup_fraction", "gradient_clip", "unlabeled_weight", "max_steps")}
        signature = {"root": str(self.data.root.resolve()), "labels": self.data.labels,
                     "config": self.config["model"], "settings": settings,
                     "row_id": self.data.rows[fitted].tolist(), "annotations": self.config["data"].get("annotations")}
        if path and path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            targets = torch.from_numpy(self.data.targets[fitted])
            observed = torch.from_numpy(self.data.observed[fitted])
            if (state["signature"] != signature or not torch.equal(state["targets"], targets)
                    or not torch.equal(state["observed"], observed)):
                raise ValueError(f"Out-of-fold B0 asset does not match this training cohort: {path}")
            backbone.load_state_dict(state["backbone"])
            return backbone.eval()
        self.fit_backbone(backbone, fitted, f"{'balanced teacher' if balanced else 'B0'} fold {fold}",
                          balanced=balanced, seed=seed)
        if path:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                torch.save({"signature": signature, "backbone": backbone.state_dict(),
                            "targets": torch.from_numpy(self.data.targets[fitted]),
                            "observed": torch.from_numpy(self.data.observed[fitted])}, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return backbone

    def setup(self, model, initialize_from=None):
        from .hypothesis_relation_training import fit_h1, fit_h2, fit_h4, fit_h5
        from .hypothesis_supervision_training import fit_h7, fit_h8

        name = model.name
        rows = self.train_indices
        seed_everything(self.seed)
        model.to(self.device)
        backbone = model.backbone
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if initialize_from:
                state = torch.load(initialize_from, map_location="cpu", weights_only=True)
                if (state["model_name"] != "b0" or state["labels"] != self.data.labels
                        or state["config"]["data"]["track"] != self.data.track):
                    raise ValueError("Initialize from B0 trained for the same task and label order")
                self.initialized_from_diagnostic = state["diagnostic_subset"]
                model.load_state_dict(state["model"])
            else:
                self.fit_backbone(backbone, rows, "B0", seed=self.seed)
        if name == "b0":
            self.fit_backbone(backbone, rows, "B0", seed=self.seed)
        elif name == "h1":
            fit_h1(self, model)
        elif name == "h2":
            fit_h2(self, model)
        elif name == "h3":
            self.fit_h3(model)
        elif name == "h4":
            fit_h4(self, model)
        elif name == "h5":
            fit_h5(self, model)
        elif name == "h6":
            mode = self.settings.get("evidence_selection", "attention")
            available = (self.data.targets[rows].astype(bool) & self.data.observed[rows]).any(1) & (self.data.lengths[rows] > 1)
            model.assets.update(evidence_mode=mode, eligible_rows=int(available.sum()), mechanism_active=mode != "none" and bool(available.any()))
            self.fit_backbone(backbone, rows, "H6 re-encoding", kind="tune", evidence=mode)
        elif name == "h7":
            fit_h7(self, model)
        elif name == "h8":
            fit_h8(self, model)

    def enable_head(self, model, **assets):
        seed_everything(self.seed)
        model.assets.update(head_active=True, **assets)
        model.head = model.make_head().to(self.device)
        return model.head

    def fit_h3(self, model):
        data, rows = self.data, self.train_indices
        observed = data.observed[rows]
        complete = observed.all(1)
        partial = data.partial_ignorable[rows] & observed.any(1) & ~complete
        support = count_support(data.targets[rows], observed, data.clusters[rows])
        model.assets.update(mechanism_active=bool(complete.any()), complete_rows=int(complete.sum()),
                            partial_rows=int(partial.sum()), count_family_support=support.tolist())
        if not complete.any():
            model.assets["reason"] = "no_complete_training_annotations"
            return
        head = self.enable_head(model)
        head.set_support(support)
        rows = rows[complete | partial]
        encoded = self.encode(model.backbone, rows)

        def objective(positions, rows, evidence, summary, logits):
            targets, observed, _ = self.labels(rows)
            partial = torch.as_tensor(data.partial_ignorable[rows], device=self.device)
            counts, energies = head(evidence, summary, logits)
            return cardinality_nll(counts, energies, targets, observed, partial)

        self.fit_cached(head, "H3 cardinality head", rows, encoded, objective)
