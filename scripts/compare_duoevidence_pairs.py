"""Compare three two-branch combinations without calling a third branch."""

import argparse
import io
import itertools
import json
import tarfile
from pathlib import Path

import numpy as np

from mfp_nova.reference_fallback import paired_cluster_interval


def compare(base, helix):
    order = np.argsort(helix["row_id"])
    idx = order[np.searchsorted(helix["row_id"][order], base["row_id"])]
    for key in ("row_id", "targets", "cluster_id"):
        np.testing.assert_array_equal(base[key], helix[key][idx])
    np.testing.assert_array_equal(base["label_names"], helix["label_names"])
    if any(str(export["split"]) != "val" for export in (base, helix)):
        raise ValueError("Use the aligned Fine39 Validation exports")
    y = base["targets"].astype(bool)
    rows = np.arange(len(y))
    old = base["predicted"].astype(bool)
    clusters = base["cluster_id"]
    limit = np.log(999999.)
    a, b = base["offsets"]
    logits = {"theraset": base["theraset_logits"].clip(-limit, limit),
              "aa34": base["aa34_logits"].clip(-limit, limit),
              "helix": helix["latent_logits"][idx].clip(-limit, limit)}
    offsets = {"theraset": a, "aa34": b, "helix": np.log(.64/.36)}
    reference_zero = ~(old & y).any(1)
    def jaccard(p):
        return (p & y).sum(1) / (p | y).sum(1).clip(min=1)

    results, outputs = {}, {}
    for left, right in itertools.combinations(logits, 2):
        key = f"{left}_{right}"
        score = ((logits[left] - offsets[left]) + (logits[right] - offsets[right])) / 2
        raw = score >= 0
        if key == "theraset_aa34":
            np.testing.assert_array_equal(raw, base["predicted_raw"])
        empty = ~raw.any(1)
        proposed = raw.copy()
        proposed[empty, score[empty].argmax(1)] = True
        delta = jaccard(proposed) - jaccard(old)
        hit = (proposed & y).any(1)
        lhit, rhit = (y[rows, logits[name].argmax(1)] for name in (left, right))
        results[key] = {
            "raw_acc": float(jaccard(raw).mean()),
            "fused_top1_fallback_acc": float(jaccard(proposed).mean()),
            "delta_vs_original_protocol_pp": float(100 * delta.mean()),
            "empty_before_fallback": int(empty.sum()),
            "rescued_original_nonempty_zero": int((reference_zero & base["predicted_raw"].any(1) & hit).sum()),
            "new_zero_vs_original": int((~reference_zero & ~hit).sum()),
            "zero_hit": int((~hit).sum()),
            "left_only_top1": int((lhit & ~rhit).sum()),
            "right_only_top1": int((rhit & ~lhit).sum()),
        }
        outputs[key] = jaccard(proposed)
    same_rule = outputs["theraset_aa34"]
    for key, result in results.items():
        d = outputs[key] - same_rule
        result.update(delta_vs_theraset_aa34_same_rule_pp=float(100 * d.mean()),
                      ci95_vs_theraset_aa34_same_rule_pp=paired_cluster_interval(d, clusters))
    return {"rows": len(y), "original_protocol_acc": float(jaccard(old).mean()), "pairs": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("helix", type=Path)
    parser.add_argument("--member", help="Base prediction member in a tar archive")
    args = parser.parse_args()
    source = args.base
    if args.member:
        with tarfile.open(source) as archive:
            source = io.BytesIO(archive.extractfile(args.member).read())
    with np.load(source) as base, np.load(args.helix) as helix:
        print(json.dumps(compare(base, helix), indent=2))
