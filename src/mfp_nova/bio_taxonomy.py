"""Fine39 biological taxonomy, Fine39-Bio (Y_bio, M_obs) targets, and Partial-Label evaluation."""

from pathlib import Path

import numpy as np

from .data import PeptideData
from .multimodal import function_projection, independent_functions

BIO_AXES = (
    "antibacterial",
    "antifungal",
    "antiparasitic",
    "antiviral",
    "antimicrobial",
    "immunoactive",
    "anticancer",
    "antihypertensive",
    "thrombolytic",
    "neuropeptide",
    "antioxidant",
    "metabolic_regulatory",
    "growth_regeneration",
    "cell_communication",
    "drug_delivery",
)

COEXISTING_FAMILIES = (
    "antibacterial",
    "anticancer",
    "antiviral",
    "neuropeptide",
    "metabolic_regulatory",
    "cell_communication",
    "immunoactive",
    "antiparasitic",
    "growth_regeneration",
)

PURE_LEAF_FAMILIES = (
    "antifungal",
    "antihypertensive",
    "antioxidant",
    "thrombolytic",
)

ANTIMICROBIAL_BRANCH_AXES = (0, 1, 2, 3)  # antibacterial, antifungal, antiparasitic, antiviral
ANTIMICROBIAL_GENERIC_AXIS = 4


def fine39_bio_names(labels):
    """Map Fine39 literal names (with __unknown) to clean biological names."""
    return [str(name).removesuffix("__unknown") for name in labels]


def fine39_hierarchy_spec(labels):
    """Build exact axis, __unknown, subtype, and antimicrobial subsumption index tables."""
    labels = [str(x) for x in labels]
    proj = np.asarray(function_projection(labels, "fine39"), dtype=bool)
    families = []
    for axis_idx, axis_name in enumerate(BIO_AXES):
        members = np.flatnonzero(proj[:, axis_idx]).tolist()
        unknown_indices = [i for i in members if labels[i].endswith("__unknown")]
        subtype_indices = [i for i in members if not labels[i].endswith("__unknown")]
        families.append({
            "axis_idx": axis_idx,
            "axis_name": axis_name,
            "unknown_idx": unknown_indices[0] if unknown_indices else None,
            "subtype_indices": subtype_indices,
            "members": members,
        })
    am_unknown_idx = labels.index("antimicrobial__unknown")
    am_member_indices = np.flatnonzero(proj[:, ANTIMICROBIAL_BRANCH_AXES].any(axis=1)).tolist()
    return {
        "labels": labels,
        "bio_names": fine39_bio_names(labels),
        "projection": proj,
        "families": families,
        "am_unknown_idx": am_unknown_idx,
        "am_member_indices": am_member_indices,
    }


def granularity_mutex_matrix(labels, include_growth=False):
    """Boolean [C, C] matrix marking pairs with strict 0 co-occurrence in Fine39.

    Covers the 8 clean co-existing families (excluding growth_regeneration's 5-row
    upstream casing exception unless include_growth=True) plus antimicrobial__unknown
    against all 4 antimicrobial branches and their subtypes (33 pairs total).
    """
    spec = fine39_hierarchy_spec(labels)
    c = len(spec["labels"])
    mutex = np.zeros((c, c), dtype=bool)
    for fam in spec["families"]:
        u = fam["unknown_idx"]
        subs = fam["subtype_indices"]
        if u is not None and subs:
            if not include_growth and fam["axis_name"] == "growth_regeneration":
                continue
            for s in subs:
                mutex[u, s] = True
                mutex[s, u] = True
    am_u = spec["am_unknown_idx"]
    for s in spec["am_member_indices"]:
        mutex[am_u, s] = True
        mutex[s, am_u] = True
    return mutex


def build_fine39_bio_targets(data_root, include_omitted_parent_positives=True):
    """Construct biologically consistent Fine39-Bio (Y_bio, M_obs) aligned to Fine39.

    - Y_bio restores biological upward closure (subtype -> parent family, 4 antimicrobial
      branches -> antimicrobial) and optionally restores omitted 47-label parent family
      positives from major15 (aligned on exact row_id).
    - M_obs sets 0.0 on unmeasured subtypes when only F__unknown == 1 is recorded
      (and on all 4 antimicrobial branches/subtypes when only antimicrobial__unknown == 1),
      eliminating false negative gradients without changing the 39-column output space.
    """
    root = Path(data_root)
    f39 = PeptideData(root, "fine39")
    m15 = PeptideData(root, "major15")
    if not np.array_equal(f39.rows, m15.rows):
        raise ValueError("fine39 and major15 row_id order mismatch")
    if list(m15.labels) != list(BIO_AXES):
        raise ValueError(f"Unexpected major15 label order: {m15.labels}")

    spec = fine39_hierarchy_spec(f39.labels)
    y39 = f39.y.astype(bool)
    y15 = m15.y.astype(bool)
    n, c = y39.shape

    y_bio = y39.copy()
    m_obs = np.ones((n, c), dtype=np.float32)

    for fam in spec["families"]:
        u = fam["unknown_idx"]
        subs = fam["subtype_indices"]
        ax = fam["axis_idx"]
        if u is not None and subs:
            has_sub = y39[:, subs].any(axis=1)
            # Biological upward closure: any recorded subtype implies parent family activity.
            y_bio[:, u] |= has_sub
            # Unobserved subtype mask: F__unknown=1 without subtypes => mask subtypes (0.0).
            only_unk = y39[:, u] & ~has_sub
            if only_unk.any():
                m_obs[np.ix_(only_unk, subs)] = 0.0
        if include_omitted_parent_positives and u is not None:
            # Restore omitted 47-leaf parent family positives (e.g. antimalarial -> antiparasitic).
            y_bio[:, u] |= y15[:, ax]

    am_u = spec["am_unknown_idx"]
    am_members = spec["am_member_indices"]
    has_am_branch = y39[:, am_members].any(axis=1)
    if include_omitted_parent_positives:
        has_am_branch = has_am_branch | y15[:, list(ANTIMICROBIAL_BRANCH_AXES)].any(axis=1)
    y_bio[:, am_u] |= has_am_branch

    # When only generic antimicrobial__unknown is annotated, the 4 branches are unmeasured.
    only_am_unk = y39[:, am_u] & ~y39[:, am_members].any(axis=1)
    if only_am_unk.any():
        m_obs[np.ix_(only_am_unk, am_members)] = 0.0

    # Family membership matrix [15, 39] and parent->children metadata for BioGranularityHead.
    parent_indices = [fam["unknown_idx"] for fam in spec["families"]]
    return {
        "row_id": f39.rows,
        "labels": f39.labels,
        "bio_names": spec["bio_names"],
        "y_bio": y_bio.astype(np.float32),
        "m_obs": m_obs,
        "y_fine39": f39.y.astype(np.float32),
        "mutex_matrix": granularity_mutex_matrix(f39.labels),
        "spec": spec,
        "parent_indices": parent_indices,
    }


def partial_label_bio_eval(y_true, y_pred, labels):
    """Evaluate biological consistency and granularity vs cross-axis error breakdown.

    Computes:
    - bio_axis_jaccard_pct: sample Jaccard after projecting to the 15 independent axes.
    - partial_label_jaccard_pct: granularity-tolerant hierarchical Jaccard where:
        * Known subtype(s) in family F are graded strictly on subtype(s), with 0.5 partial
          credit if the model only predicts coarse F__unknown (or antimicrobial__unknown).
        * Unmeasured F__unknown=1 accepts either F__unknown or subtype refinement in F
          (capped at 1 family slot to prevent multi-guess inflation; illegal mixing penalized).
    - Zero-hit decomposition: splits literal zero-Jaccard predictions into
      granularity-only mismatches (right bio axis, wrong __unknown/subtype) vs
      cross-axis confusions.
    """
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    spec = fine39_hierarchy_spec(labels)
    proj = spec["projection"]
    mutex = granularity_mutex_matrix(labels)

    ax_y = independent_functions(y, proj)
    ax_p = independent_functions(p, proj)
    ax_inter = (ax_y & ax_p).sum(axis=1)
    ax_union = (ax_y | ax_p).sum(axis=1)
    bio_axis_j = np.where(ax_union > 0, ax_inter / np.maximum(ax_union, 1), 0.0)

    literal_inter = (y & p).sum(axis=1)
    literal_zero = literal_inter == 0
    nonempty_zero = literal_zero & p.any(axis=1)
    bio_hit_on_zero = literal_zero & ((ax_y & ax_p).sum(axis=1) > 0)
    gran_only_zero = int(bio_hit_on_zero.sum())
    cross_axis_zero = int((literal_zero & ~bio_hit_on_zero).sum())
    nonempty_gran_zero = int((nonempty_zero & bio_hit_on_zero).sum())
    illegal_mutex_rows = int(((p[:, :, None] & p[:, None, :]) & mutex[None, :, :]).any(axis=(1, 2)).sum())

    am_u = spec["am_unknown_idx"]
    am_members = spec["am_member_indices"]
    families = spec["families"]

    n = len(y)
    pl_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        yi = y[i]
        pi = p[i]
        inter_i = 0.0
        union_i = 0.0

        # Check antimicrobial super-group [axes 0..4].
        gt_am_only = bool(yi[am_u]) and not bool(yi[am_members].any())
        pred_am_u = bool(pi[am_u])
        pred_am_sub = bool(pi[am_members].any())
        if gt_am_only:
            if pred_am_u and pred_am_sub:
                inter_i += 1.0
                union_i += float(1 + int(pi[am_members].sum()))
            elif pred_am_u or pred_am_sub:
                inter_i += 1.0
                union_i += 1.0
            else:
                union_i += 1.0
            skip_axes = set(ANTIMICROBIAL_BRANCH_AXES) | {ANTIMICROBIAL_GENERIC_AXIS}
        elif bool(yi[am_members].any()) and pred_am_u and not pred_am_sub:
            # Coarse antimicrobial__unknown fallback for specific antimicrobial branch(es).
            inter_i += 0.5
            union_i += float(int(yi[am_members].sum()))
            skip_axes = set(ANTIMICROBIAL_BRANCH_AXES) | {ANTIMICROBIAL_GENERIC_AXIS}
        else:
            skip_axes = set()

        for fam in families:
            ax = fam["axis_idx"]
            if ax in skip_axes:
                continue
            members = fam["members"]
            u = fam["unknown_idx"]
            subs = fam["subtype_indices"]

            if ax == ANTIMICROBIAL_GENERIC_AXIS:
                if pred_am_u and not yi[am_u]:
                    union_i += 1.0
                continue

            ym = yi[members]
            pm = pi[members]
            if not ym.any() and not pm.any():
                continue
            if not ym.any():
                union_i += float(pm.sum())
                continue

            # Ground truth only records F__unknown in a co-existing family.
            if u is not None and subs and yi[u] and not yi[subs].any():
                pred_u = bool(pi[u])
                pred_sub_cnt = int(pi[subs].sum())
                if pred_u and pred_sub_cnt > 0:
                    inter_i += 1.0
                    union_i += float(1 + pred_sub_cnt)
                elif pred_u or pred_sub_cnt > 0:
                    inter_i += 1.0
                    union_i += 1.0
                else:
                    union_i += 1.0
                continue

            # Ground truth has specific subtype(s) or leaf-only family.
            exact_hits = float((ym & pm).sum())
            if exact_hits > 0:
                inter_i += exact_hits
                union_i += float((ym | pm).sum())
            elif u is not None and subs and pi[u] and not pi[subs].any():
                # Coarse F__unknown fallback when subtype(s) were known.
                inter_i += 0.5
                union_i += float(max(int(ym.sum()), 1))
            else:
                union_i += float((ym | pm).sum())

        pl_scores[i] = inter_i / union_i if union_i > 0 else 0.0

    n_zero = int(literal_zero.sum())
    return {
        "bio_axis_jaccard_pct": float(bio_axis_j.mean() * 100.0),
        "partial_label_jaccard_pct": float(pl_scores.mean() * 100.0),
        "granularity_only_zero_rows": gran_only_zero,
        "granularity_only_zero_pct_of_zero": float(gran_only_zero / max(1, n_zero) * 100.0),
        "nonempty_granularity_zero_rows": nonempty_gran_zero,
        "cross_axis_zero_rows": cross_axis_zero,
        "cross_axis_zero_pct_of_zero": float(cross_axis_zero / max(1, n_zero) * 100.0),
        "illegal_mutex_rows": illegal_mutex_rows,
    }


def cluster_balanced_positive_multipliers(data, alpha=0.35):
    """Per-row, per-label positive risk weights n_{g(i), c}^{-alpha}, normalized per label on Train.

    Prevents large mutation-series clusters (up to 546 peptides) from dominating CNN/attention
    gradients while preserving exact per-label positive gradient mass and @0.50 calibration.
    """
    if alpha < 0.0:
        raise ValueError("cluster_balance_alpha must be >= 0")
    train = data.indices("train")
    multipliers = np.ones(data.y.shape, dtype=np.float32)
    if alpha == 0.0 or len(train) == 0:
        return multipliers
    targets = data.y[train].astype(bool)
    clusters = data.clusters[train]
    for col in range(data.y.shape[1]):
        pos_mask = targets[:, col]
        if not pos_mask.any():
            continue
        pos_rows = train[pos_mask]
        _, inv, counts = np.unique(clusters[pos_mask], return_inverse=True, return_counts=True)
        raw_w = np.power(counts[inv].astype(np.float64), -float(alpha))
        raw_w *= len(raw_w) / max(float(raw_w.sum()), 1e-12)
        multipliers[pos_rows, col] = raw_w.astype(np.float32)
    return multipliers

