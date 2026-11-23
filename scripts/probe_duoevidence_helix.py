"""Test a fixed HelixSet contribution on nonempty DuoEvidence predictions."""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from mfp_nova.reference_fallback import paired_cluster_interval


def compare(base, helix, masks=None, cohorts=None):
    order = np.argsort(helix["row_id"])
    idx = order[np.searchsorted(helix["row_id"][order], base["row_id"])]
    for key in ("row_id", "targets", "cluster_id"):
        np.testing.assert_array_equal(base[key], helix[key][idx])
    np.testing.assert_array_equal(base["label_names"], helix["label_names"])
    y, raw, old = (base[key].astype(bool) for key in ("targets", "predicted_raw", "predicted"))
    limit = np.log(999999.)
    thera, aa34 = (base[key].clip(-limit, limit) for key in ("theraset_logits", "aa34_logits"))
    a, b = base["offsets"]
    old_score = (thera - a + aa34 - b) / 2
    latent = helix["latent_logits"][idx].clip(-limit, limit)
    new_raw = (2 * old_score + latent - np.log(.64 / .36)) >= 0
    nonempty = raw.any(1)
    new = np.where((nonempty & new_raw.any(1))[:, None], new_raw, old)
    old_hit, new_hit = ((p & y).any(1) for p in (old, new))
    scores = [(p & y).sum(1) / (p | y).sum(1).clip(min=1) for p in (old, new)]
    delta = scores[1] - scores[0]
    rows = np.arange(len(y))
    truth = np.where(y, old_score, -np.inf).argmax(1)
    wrong = np.where(old, old_score, -np.inf).argmax(1)
    ranks = np.argsort(np.argsort(-old_score, axis=1, kind="stable"), axis=1)
    both_wrong = (thera[rows, wrong] > thera[rows, truth]) & (aa34[rows, wrong] > aa34[rows, truth])
    groups = {"nonempty_zero": nonempty & ~old_hit,
              "both_prefer_error_true_top5": nonempty & ~old_hit & both_wrong & (ranks[rows, truth] < 5),
              "nonempty_already_hit": nonempty & old_hit}
    if cohorts is not None:
        cohorts.update(groups)
    if masks is not None:
        masks.update({name: group & ~old_hit & new_hit for name, group in groups.items()
                      if name != "nonempty_already_hit"})
    return {"rows": len(y), "acc_before": float(scores[0].mean()), "acc_after": float(scores[1].mean()),
            "delta_pp": float(100 * delta.mean()), "ci95_pp": paired_cluster_interval(delta, base["cluster_id"]),
            "rescued": int((~old_hit & new_hit).sum()), "new_zero": int((old_hit & ~new_hit).sum()),
            "groups": {k: {"rows": int(g.sum()), "rescued": int((g & ~old_hit & new_hit).sum()),
                           "new_zero": int((g & old_hit & ~new_hit).sum()),
                           "delta_contribution_pp": float(100 * delta[g].sum() / len(y))}
                       for k, g in groups.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("helix", type=Path)
    parser.add_argument("--member", help="DuoEvidence prediction member in a tar archive")
    parser.add_argument("--reference-helix", type=Path, help="Full HelixSet export for paired rescue overlap")
    args = parser.parse_args()
    source = args.base
    if args.member:
        with tarfile.open(source) as archive:
            source = io.BytesIO(archive.extractfile(args.member).read())
    with np.load(source) as base, np.load(args.helix) as helix:
        candidate = {}
        report = compare(base, helix, candidate)
        if args.reference_helix:
            reference = {}
            with np.load(args.reference_helix) as full:
                compare(base, full, reference)
            report["rescue_overlap_vs_full"] = {
                name: {"retained": int((old & candidate[name]).sum()),
                       "lost": int((old & ~candidate[name]).sum()),
                       "gained": int((~old & candidate[name]).sum())}
                for name, old in reference.items()
            }
        print(json.dumps(report, indent=2))
