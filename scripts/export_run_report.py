"""Export summary tables (CSV/Markdown) and publication-ready plots (PNG) for multi-arm MFP-Nova runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnose_fine39_panel import apply_empty_top1, compare_exports, evaluate_panel, load_export, sample_jaccard


def discover_arms(run_dir: Path):
    arms = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        mdir = child / "model" if (child / "model").is_dir() else child
        if (mdir / "history.jsonl").is_file() or (mdir / "results.json").is_file():
            arms.append((child.name, mdir))
    return arms


def load_history(mdir: Path):
    hp = mdir / "history.jsonl"
    if not hp.is_file():
        return []
    return [json.loads(line) for line in hp.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def build_summary_rows(arms, baseline_name=None, data_root=None):
    low_mask = None
    exports = {}
    for name, mdir in arms:
        npz = mdir / "best_validation.npz"
        if npz.is_file():
            exports[name] = load_export(npz)

    if data_root is not None and Path(data_root).exists() and exports:
        from mfp_nova.data import PeptideData
        from analyze_xbridge_validation import nearest_similarity
        data = PeptideData(Path(data_root), "fine39")
        val_idx, tr_idx = data.indices("val"), data.indices("train")
        sim = nearest_similarity(data, val_idx, tr_idx)
        rmap = {int(r): float(s) for r, s in zip(data.rows[val_idx], sim)}
        first_ex = next(iter(exports.values()))
        row_sim = np.array([rmap[int(r)] for r in first_ex["row_id"]])
        low_mask = row_sim < 0.70

    base_ex = exports.get(baseline_name) if baseline_name else None
    rows = []
    for name, mdir in arms:
        res_file = mdir / "results.json"
        res = json.loads(res_file.read_text(encoding="utf-8")) if res_file.is_file() else {}
        ex = exports.get(name)
        panel, *_ = evaluate_panel(ex, threshold=0.5, low_mask=low_mask) if ex is not None else ({},)
        ci_str = "-"
        if ex is not None and base_ex is not None and name != baseline_name:
            comp = compare_exports(ex, base_ex, threshold=0.5, low_mask=low_mask)["paired_comparison"]
            ci_str = f"{comp['raw_delta_pp']:+.2f} [{comp['raw_ci95_pp'][0]:+.2f}, {comp['raw_ci95_pp'][1]:+.2f}]"
        rows.append({
            "Arm": name,
            "BestEp": res.get("best_epoch", "-"),
            "Epochs": res.get("epochs_completed", "-"),
            "Hours": f"{res['elapsed_seconds'] / 3600:.2f}" if "elapsed_seconds" in res else "-",
            "VRAM_GB": f"{res['peak_cuda_memory_bytes'] / (1024**3):.2f}" if res.get("peak_cuda_memory_bytes") else "-",
            "RawACC%": f"{panel['raw_acc_pct']:.2f}" if panel else "-",
            "Delta_CI95_pp": ci_str,
            "FallbackACC%": f"{panel['fallback_acc_pct']:.2f}" if panel else "-",
            "MacroAP%": f"{panel['macro_ap_pct']:.2f}" if panel else "-",
            "MicroF1%": f"{panel['micro_f1_pct']:.2f}" if panel else "-",
            "MFP_F1%": f"{panel['mfp_detection_f1_pct']:.2f}" if panel else "-",
            "Ge2Axes%": f"{panel['ge2_axes_recovered_pct']:.2f}" if panel else "-",
            "Top1%": f"{panel['top1_hit_pct']:.2f}" if panel else "-",
            "Top5%": f"{panel['top5_hit_pct']:.2f}" if panel else "-",
            "Empty": panel.get("raw_empty", "-"),
            "NonEmptyZero": panel.get("nonempty_zero", "-"),
            "FinalZero": panel.get("final_zero", "-"),
            "LowSimACC%": f"{panel['low_sim_raw_acc_pct']:.2f}" if "low_sim_raw_acc_pct" in panel else "-",
        })
    return rows, exports, low_mask


def write_tables(rows, out_dir: Path):
    if not rows:
        return
    headers = list(rows[0].keys())
    with (out_dir / "summary_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    (out_dir / "summary_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def plot_reports(arms, exports, low_mask, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped PNG plots (tables still generated).")
        return

    # 1. Learning curves (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    specs = [
        ("Val Sample Jaccard (%)", lambda h: h["validation"]["sample_jaccard"] * 100),
        ("Val Macro-AP (%)", lambda h: h["validation"]["macro_ap"] * 100),
        ("Val Micro-F1 (%)", lambda h: h["validation"]["micro_f1"] * 100),
        ("Validation Loss", lambda h: h["validation_loss"]),
    ]
    for ax, (title, fn) in zip(axes.flat, specs):
        for name, mdir in arms:
            hist = load_history(mdir)
            if not hist:
                continue
            eps = [h["epoch"] for h in hist]
            vals = [fn(h) for h in hist]
            line, = ax.plot(eps, vals, label=name, linewidth=1.8)
            best_ep = hist[-1].get("best_epoch")
            if best_ep and 1 <= best_ep <= len(vals):
                ax.scatter([best_ep], [vals[best_ep - 1]], color=line.get_color(), marker="*", s=90, zorder=5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png")
    plt.close(fig)

    # 2. Threshold sweep & subgroup/error breakdown
    if exports:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
        ths = np.arange(0.38, 0.65, 0.02)
        for name, ex in exports.items():
            y, p = ex["targets"], ex["probabilities"]
            accs = [sample_jaccard(y, p >= t).mean() * 100 for t in ths]
            axes[0].plot(ths, accs, marker="o", markersize=3, label=name)
        axes[0].axvline(0.50, color="gray", linestyle=":", linewidth=1)
        axes[0].set_title("Threshold Sweep (Raw ACC %)", fontweight="bold")
        axes[0].set_xlabel("Threshold")
        axes[0].grid(alpha=0.3, linestyle="--")
        axes[0].legend(fontsize=8)

        names = list(exports.keys())
        x = np.arange(len(names))
        width = 0.28
        raw_vals = [sample_jaccard(exports[n]["targets"], exports[n]["probabilities"] >= 0.5).mean() * 100 for n in names]
        fb_vals = [sample_jaccard(exports[n]["targets"], apply_empty_top1(exports[n]["probabilities"] >= 0.5, exports[n]["probabilities"])).mean() * 100 for n in names]
        axes[1].bar(x - width / 2, raw_vals, width, label="Raw @0.50")
        axes[1].bar(x + width / 2, fb_vals, width, label="Fallback Top-1")
        if low_mask is not None and low_mask.any():
            low_vals = [sample_jaccard(exports[n]["targets"][low_mask], exports[n]["probabilities"][low_mask] >= 0.5).mean() * 100 for n in names]
            axes[1].bar(x + 1.5 * width, low_vals, width, label="Low-Sim <0.70")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        axes[1].set_title("Overall & Low-Sim ACC (%)", fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        axes[1].legend(fontsize=8)

        empties = [int((~(exports[n]["probabilities"] >= 0.5).any(1)).sum()) for n in names]
        nzs = [int((~((exports[n]["probabilities"] >= 0.5) & exports[n]["targets"]).any(1) & (exports[n]["probabilities"] >= 0.5).any(1)).sum()) for n in names]
        axes[2].bar(x, empties, label="Empty Preds")
        axes[2].bar(x, nzs, bottom=empties, label="NonEmpty Zero-Hit")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        axes[2].set_title("Zero-Hit Breakdown (Lower Better)", fontweight="bold")
        axes[2].grid(axis="y", alpha=0.3, linestyle="--")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "subgroups_and_thresholds.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Export tables and curves for a multi-arm MFP-Nova run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", help="Optional baseline arm name for paired cluster CI")
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--output", type=Path, help="Output directory (defaults to <run_dir>/report)")
    args = parser.parse_args()

    arms = discover_arms(args.run_dir)
    if not arms:
        raise SystemExit(f"No experiment arms found in {args.run_dir}")
    out_dir = args.output or (args.run_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, exports, low_mask = build_summary_rows(arms, args.baseline, args.data_root)
    write_tables(rows, out_dir)
    plot_reports(arms, exports, low_mask, out_dir)
    print(f"Report exported to {out_dir}")


if __name__ == "__main__":
    main()
