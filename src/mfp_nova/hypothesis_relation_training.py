"""Train-only assets and fitting phases for H1, H2, H4, and H5."""

import numpy as np
import torch
from torch.nn import functional as F

from .models.hypothesis_relations import (
    EvidenceMessages,
    align_edge_probes,
    build_prototypes,
    four_state_support,
    joint_loss,
    matched_pairs,
    message_probe_values,
    ranking_backgrounds,
    select_edges,
    select_joint_pairs,
    shuffled_gate_targets,
)


def _support(data, rows):
    return four_state_support(data.targets[rows], data.observed[rows], data.clusters[rows])


def _bce(logits, targets, weights):
    return (F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none") * weights).sum()


def fit_h1(session, model):
    data, rows = session.data, session.train_indices
    mode = session.settings.get("prototype_mode", "background")
    support = _support(data, rows)
    model.assets.update(prototype_mode=mode, mechanism_active=False)
    if not len(select_edges(support, data.labels, max_incoming=1)):
        model.assets.update(prototype_labels=0, reason="no_supported_four_state_background")
        return
    encoded = session.encode(model.backbone, rows, fields=("evidence", "logits"))

    def bank(indices, seed, state_support=None):
        selected = rows[indices]
        return build_prototypes(encoded["evidence"], data.targets[selected], data.observed[selected],
                                data.clusters[selected], data.labels, mode=mode, seed=seed,
                                row_indices=indices, support=state_support)

    deploy_bank, deploy_active, companions = bank(np.arange(len(rows)), session.seed, support)
    train_bank, train_active = [], []
    bank_by_row = np.empty(len(rows), dtype=int)
    fold_support = {}
    for fold, fitted, held in session.training_folds():
        prototypes, active, _ = bank(fitted, session.seed + fold + 1)
        bank_by_row[held] = len(train_bank)
        train_bank.append(prototypes)
        train_active.append(active)
        fold_support[str(fold)] = int(active.sum())
    train_bank, train_active = torch.stack(train_bank), torch.stack(train_active)
    train_active_rows = train_active[bank_by_row].numpy() & data.observed[rows]
    deploy_active &= torch.from_numpy(train_active_rows.any(0))
    companions[~deploy_active.numpy()] = -1
    model.assets.update(prototype_labels=int(deploy_active.sum()),
                        companions=companions.tolist(), prototype_labels_by_fold=fold_support,
                        prototype_training_bits=int(train_active_rows.sum()))
    if not deploy_active.any():
        model.assets["reason"] = "no_supported_out_of_fold_prototypes"
        return
    head = session.enable_head(model, mechanism_active=True)
    head.prototypes.copy_(deploy_bank)
    head.active.copy_(deploy_active)
    train_bank, train_active = train_bank.to(session.device), train_active.to(session.device)

    def objective(positions, rows, evidence, logits):
        targets, observed, _ = session.labels(rows)
        folds = bank_by_row[positions]
        active = train_active[folds]
        if not (observed & active).any():
            return None
        predicted = head(evidence, logits, train_bank[folds], active)
        return _bce(predicted, targets, observed) / observed.sum()

    session.fit_cached(head, "H1 background prototype readout", rows, encoded, objective)


def _fit_messages(session, head, rows, encoded, phase):
    def objective(positions, rows, evidence, logits):
        targets, observed, _ = session.labels(rows)
        delta = head.messages(evidence, logits)
        loss, _, valid = message_probe_values(delta, logits, targets, observed, head.edges)
        return (loss * valid).sum() / valid.sum() if valid.any() else None

    session.fit_cached(head, phase, rows, encoded, objective)


@torch.no_grad()
def _held_probes(session, head, rows, encoded):
    head.eval()
    all_gains, all_masks = [], []
    for start in range(0, len(rows), session.batch):
        chunk = slice(start, start + session.batch)
        targets, observed, _ = session.labels(rows[chunk])
        evidence, logits = (encoded[field][chunk].to(session.device, dtype=torch.float32)
                            for field in ("evidence", "logits"))
        _, gain, valid = message_probe_values(head.messages(evidence, logits), logits, targets, observed, head.edges)
        all_gains.append(gain.cpu().numpy())
        all_masks.append(valid.cpu().numpy())
    return np.concatenate(all_gains), np.concatenate(all_masks)


def fit_h2(session, model):
    data, rows = session.data, session.train_indices
    mode = session.settings.get("gate_mode", "learned")
    model.assets.update(gate_mode=mode, mechanism_active=False)
    edges = select_edges(_support(data, rows), data.labels)
    model.assets["candidate_edges"] = len(edges)
    if not len(edges):
        model.assets["reason"] = "no_supported_four_state_edges"
        return
    fields = ("evidence", "logits")
    fold_probes = []
    for fold, fitted, held in session.training_folds():
        fitted_rows, held_rows = rows[fitted], rows[held]
        fold_edges = select_edges(_support(data, fitted_rows), data.labels)
        if not len(fold_edges):
            continue
        backbone = session.fold_backbone(fold, fitted_rows)
        torch.manual_seed(session.seed)
        messenger = EvidenceMessages(len(data.labels), model.backbone.dim, fold_edges, gate_mode="open")
        messenger.to(session.device)
        _fit_messages(session, messenger, fitted_rows, session.encode(backbone, fitted_rows, fields),
                      f"H2 message fold {fold}")
        gains, valid = _held_probes(session, messenger, held_rows, session.encode(backbone, held_rows, fields))
        fold_probes.append((held, fold_edges, gains, valid))
    edges, probe_targets, probe_mask = align_edge_probes(edges, fold_probes, len(rows))
    model.assets.update(probe_bits=int(probe_mask.sum()), positive_probes=int(probe_targets.sum()))
    if not len(edges):
        model.assets["reason"] = "no_out_of_fold_edge_probes"
        return
    head = session.enable_head(model, edges=edges.tolist(), mechanism_active=mode != "closed")
    encoded = session.encode(model.backbone, rows, fields)
    _fit_messages(session, head, rows, encoded, "H2 full message")
    if mode in ("open", "closed"):
        return
    head.message.requires_grad_(False)
    head.message_projection.requires_grad_(False)
    if mode == "shuffled":
        probe_targets = shuffled_gate_targets(probe_targets, probe_mask, session.seed)

    def objective(positions, rows, evidence, logits):
        targets, observed, _ = session.labels(rows)
        if not observed.any():
            return None
        delta, gate_logits = head.edge_values(evidence, logits)
        predicted = head.aggregate(logits, delta, gate_logits.sigmoid())
        target = torch.from_numpy(probe_targets[positions]).to(session.device)
        valid = torch.from_numpy(probe_mask[positions]).to(session.device)
        return (_bce(predicted, targets, observed) / observed.sum() +
                .5 * _bce(gate_logits, target, valid) / valid.sum().clamp_min(1))

    session.fit_cached(head, "H2 evidence gates", rows, encoded, objective)


def fit_h4(session, model):
    data, rows = session.data, session.train_indices
    mode = session.settings.get("pair_mode", "matched")
    groups = data.clusters[rows]
    targets, observed = data.targets[rows], data.observed[rows]
    backgrounds = ranking_backgrounds(targets, observed, groups, data.labels)
    scores = (session.encode(model.backbone, rows, fields=("logits",))["logits"].numpy()
              if mode == "hard" and any(backgrounds) else None)
    pairs = (matched_pairs(targets, observed, groups, data.labels, data.lengths[rows], data.sources[rows],
                           data.ids[rows], backgrounds=backgrounds, mode=mode, scores=scores, seed=session.seed)
             if mode != "none" and any(backgrounds) else np.empty((0, 3), dtype=int))
    positive, negative, labels = pairs.T
    eligible = (targets.astype(bool) & observed).sum(0)
    matched = np.bincount(labels, minlength=len(data.labels))
    model.assets.update(pair_mode=mode, ranking_backgrounds=backgrounds, training_pairs=len(pairs),
                        pairs_by_label=matched.tolist(), positive_bits_by_label=eligible.tolist(),
                        distinct_negative_rows=len(np.unique(negative)),
                        missing_source_matches=int((data.sources[rows[positive]] == "").sum()),
                        mechanism_active=bool(len(pairs)))
    if not len(pairs):
        model.assets["reason"] = "ranking_pairs_disabled" if mode == "none" else "no_reliable_matched_pairs"
    global_pairs = np.column_stack((rows[positive], rows[negative], labels))
    session.fit_backbone(model.backbone, rows, "H4 ranking", kind="tune", pairs=global_pairs)
    head = session.enable_head(model)
    encoded = session.encode(model.backbone, rows)

    def objective(positions, rows, evidence, summary, logits):
        targets, observed, weights = session.labels(rows)
        if not weights.any():
            return None
        decision, residual = head(evidence, summary, logits)
        penalty = (residual.float().square() * observed).sum() / observed.sum().clamp_min(1)
        return _bce(decision, targets, weights) / weights.sum() + .01 * penalty

    session.fit_cached(head, "H4 decision residual", rows, encoded, objective)


def fit_h5(session, model):
    data, rows = session.data, session.train_indices
    mode = session.settings.get("joint_mode", "both")
    support = _support(data, rows)
    model.assets.update(joint_mode=mode, mechanism_active=False)
    if not len(select_edges(support, data.labels, max_incoming=1)):
        model.assets["reason"] = "no_supported_four_state_pairs"
        return
    logits = np.empty((len(rows), len(data.labels)), dtype=np.float32)
    for fold, fitted, held in session.training_folds():
        backbone = session.fold_backbone(fold, rows[fitted])
        logits[held] = session.encode(backbone, rows[held], fields=("logits",))["logits"].numpy()
    pairs, candidates = select_joint_pairs(data.targets[rows], data.observed[rows], data.clusters[rows],
                                          data.labels, logits, return_details=True)
    model.assets["pair_candidates"] = candidates
    if not len(pairs):
        model.assets["reason"] = "no_supported_pairs_with_out_of_fold_scores"
        return
    head = session.enable_head(model, pairs=pairs.tolist(), mechanism_active=mode != "weighted_bce")
    encoded = session.encode(model.backbone, rows, fields=("evidence", "logits"))

    def objective(positions, rows, evidence, logits):
        targets, observed, weights = session.labels(rows)
        if not weights.any():
            return None
        if mode == "weighted_bce":
            a, b = head.pairs.unbind(1)
            singleton = observed[:, a] & observed[:, b] & (targets[:, a] != targets[:, b])
            weights[:, a] += singleton
            weights[:, b] += singleton
            return _bce(head(evidence, logits), targets, weights) / weights.sum()
        marginal, states = head(evidence, logits)
        supervised = joint_loss(states, marginal, targets, observed, head.pairs, reduction="sum")
        weak = _bce(marginal, targets, weights * ~observed)
        return (supervised + weak) / weights.sum()

    session.fit_cached(head, "H5 four-state readout", rows, encoded, objective)
