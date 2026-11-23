"""Compare single branches and independent AA34 ensembles on aligned Fine39 predictions."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from mfp_nova.models.trievidence import center_evidence, empty_top10
from mfp_nova.reference_fallback import paired_cluster_interval


def audit(base, members, nearest=None):
    y = base["targets"].astype(bool)
    original = base["predicted"].astype(bool)
    clusters = base["cluster_id"]
    aa34 = base["aa34_logits"]
    thera = base["theraset_logits"]
    offsets = base["offsets"]
    specs = {"original_aa34": aa34}
    for name, export in members.items():
        for key in ("split", "row_id", "label_names", "targets", "cluster_id", "offsets"):
            np.testing.assert_array_equal(base[key], export[key])
        np.testing.assert_allclose(thera, export["theraset_logits"], rtol=1e-5, atol=1e-4)
        specs[name] = export["aa34_logits"]
    if members:
        specs["independent_members_mean"] = np.stack([specs[name] for name in members]).mean(0)
    size = original.sum(1, keepdims=True)
    original_hit = (original & y).any(1)
    low = None
    if nearest is not None:
        order = np.argsort(nearest["row_id"])
        np.testing.assert_array_equal(base["row_id"], nearest["row_id"][order])
        low = nearest["similarity"][order] < .7

    def jaccard(pred):
        return (pred & y).sum(1) / (pred | y).sum(1).clip(min=1)

    def rank(scores):
        return np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1)

    def single(scores, offset):
        ranks = rank(scores)
        active = y.any(0)
        return {
            "raw_sample_jaccard": float(jaccard(scores >= offset).mean()),
            "matched_size_sample_jaccard": float(jaccard(ranks < size).mean()),
            "macro_ap": float(average_precision_score(y[:, active], scores[:, active], average="macro")),
            "top_hit_rows": {k: int((y & (ranks < k)).any(1).sum()) for k in (1, 5, 10)},
        }

    t_ranks = rank(thera)
    report = {"rows": len(y), "reference_acc": float(jaccard(original).mean()),
              "theraset": single(thera, offsets[0]), "aa34": {}}
    for name, scores in specs.items():
        ranks = rank(scores)
        # Preserve the same per-row output count across the two branches.
        t_hit = (y & (t_ranks < size)).any(1)
        a_hit = (y & (ranks < size)).any(1)
        centered = center_evidence(torch.from_numpy(np.stack((thera, scores))), torch.from_numpy(offsets))
        fused = centered.mean(0)
        raw = (fused >= 0).numpy()
        final = empty_top10(fused, torch.from_numpy(thera)).numpy()
        if name == "original_aa34":
            np.testing.assert_array_equal(final, original)
        hit = (final & y).any(1)
        delta = jaccard(final) - jaccard(original)
        report["aa34"][name] = {
            "standalone": single(scores, offsets[1]),
            "matched_size": {"only_theraset": int((t_hit & ~a_hit).sum()),
                             "only_aa34": int((a_hit & ~t_hit).sum()),
                             "both_hit": int((t_hit & a_hit).sum()),
                             "neither_hit": int((~t_hit & ~a_hit).sum()),
                             "at_least_one_branch_hit_fusion_missed": int(((t_hit | a_hit) & ~hit).sum())},
            "fusion": {"raw_acc": float(jaccard(raw).mean()), "final_acc": float(jaccard(final).mean()),
                       "delta_pp": float(100 * delta.mean()), "ci95_pp": paired_cluster_interval(delta, clusters),
                       "rescued": int((~original_hit & hit).sum()), "new_zero": int((original_hit & ~hit).sum())},
        }
        if low is not None and low.any():
            report["aa34"][name]["similarity_lt70"] = {
                "rows": int(low.sum()), "only_theraset": int((low & t_hit & ~a_hit).sum()),
                "only_aa34": int((low & a_hit & ~t_hit).sum()),
                "theraset_top5_hits": int((low & (y & (t_ranks < 5)).any(1)).sum()),
                "aa34_top5_hits": int((low & (y & (ranks < 5)).any(1)).sum()),
                "fusion_delta_pp": float(100 * delta[low].mean()),
            }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("members", type=Path, nargs="*")
    parser.add_argument("--nearest", type=Path)
    args = parser.parse_args()
    with ExitStack() as files:
        base = files.enter_context(np.load(args.base))
        members = {p.parent.name: files.enter_context(np.load(p)) for p in args.members}
        nearest = files.enter_context(np.load(args.nearest)) if args.nearest else None
        print(json.dumps(audit(base, members, nearest), indent=2))
