"""Live Rich terminal dashboard for multi-arm MFP-Nova experiments with switchable detail views."""

import argparse
import json
import time
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

VIEWS = ("core", "full", "loss", "mfp")


def discover_arms(run_dir: Path):
    if (run_dir / "history.jsonl").is_file() or (run_dir / "results.json").is_file():
        return [(run_dir.name, run_dir)]
    arms = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        model_dir = child / "model" if (child / "model").is_dir() else child
        if (model_dir / "history.jsonl").is_file() or (model_dir / "results.json").is_file() or (child / "train.log").is_file():
            arms.append((child.name, model_dir))
    return arms


def parse_arm(name: str, model_dir: Path):
    hist_path = model_dir / "history.jsonl"
    res_path = model_dir / "results.json"
    meta_path = model_dir / "metadata.json"

    planned = patience_lim = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg = meta.get("config", {}).get("train", {})
            planned = cfg.get("epochs")
            patience_lim = cfg.get("patience")
        except Exception:
            pass

    lines = []
    if hist_path.is_file():
        for raw in hist_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = raw.strip()
            if raw:
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue

    done = res_path.is_file()
    if not lines:
        init_elapsed = None
        log_path = model_dir.parent / "train.log"
        if log_path.is_file():
            init_elapsed = max(0.0, time.time() - log_path.stat().st_mtime)
        return {
            "name": name, "status": "DONE" if done else "INIT",
            "epoch": 0, "planned": planned, "lr": None,
            "tr_loss": None, "val_loss": None, "val_acc": None, "tr_acc": None,
            "val_f1": None, "val_ap": None, "best_acc": None, "best_ep": None,
            "stale": 0, "patience_lim": patience_lim, "mem_gb": None,
            "ep_sec": None, "elapsed_sec": init_elapsed, "eta_sec": None,
            "loss_a": None, "loss_b": None, "loss_c": None, "sym_rank": None, "rdrop": None,
            "prec": None, "rec": None, "exact": None, "mfp_f1": None,
            "head_ap": None, "tail_ap": None, "wait_pct": None,
        }

    last = lines[-1]
    val = last.get("validation", {})
    tr = last.get("train", {})
    losses = last.get("losses", {})
    best_acc = max((h.get("validation", {}).get("sample_jaccard", 0.0) for h in lines), default=0.0)
    peak_bytes = max(((h.get("peak_cuda_memory_bytes") or 0) for h in lines), default=0)

    recent_secs = [h["seconds"] for h in lines[1:] if "seconds" in h][-5:]
    if not recent_secs:
        recent_secs = [h["seconds"] for h in lines if "seconds" in h][-5:]
    ep_sec = sum(recent_secs) / len(recent_secs) if recent_secs else None
    elapsed_sec = last.get("elapsed_seconds")
    if not done and elapsed_sec is not None and hist_path.is_file():
        elapsed_sec += max(0.0, time.time() - hist_path.stat().st_mtime)
    cur_ep = last.get("epoch", len(lines))
    eta_sec = 0.0 if done else (
        max(0, planned - cur_ep) * ep_sec if (planned and ep_sec is not None) else None
    )

    tr_sec = last.get("train_seconds") or 0.0
    wait_sec = last.get("data_wait_seconds") or 0.0
    wait_pct = (wait_sec / tr_sec) if tr_sec > 0 else None
    mfp = val.get("mfp_detection", {})

    return {
        "name": name,
        "status": "DONE" if done else "RUN",
        "epoch": cur_ep,
        "planned": planned,
        "lr": last.get("lr"),
        "tr_loss": losses.get("total") if isinstance(losses, dict) else None,
        "val_loss": last.get("validation_loss"),
        "tr_acc": tr.get("sample_jaccard") if isinstance(tr, dict) else None,
        "val_acc": val.get("sample_jaccard"),
        "val_f1": val.get("micro_f1"),
        "val_ap": val.get("macro_ap"),
        "best_acc": best_acc,
        "best_ep": last.get("best_epoch"),
        "stale": last.get("early_stopping_count", 0),
        "patience_lim": patience_lim,
        "mem_gb": peak_bytes / (1024 ** 3) if peak_bytes else None,
        "ep_sec": ep_sec,
        "elapsed_sec": elapsed_sec,
        "eta_sec": eta_sec,
        "loss_a": losses.get("path_a_cls") if isinstance(losses, dict) else None,
        "loss_b": losses.get("path_b_cls") if isinstance(losses, dict) else None,
        "loss_c": losses.get("path_c_cls") if isinstance(losses, dict) else None,
        "sym_rank": losses.get("sym_rank") if isinstance(losses, dict) else None,
        "rdrop": losses.get("rdrop") if isinstance(losses, dict) else None,
        "prec": val.get("sample_precision"),
        "rec": val.get("sample_recall"),
        "exact": val.get("exact_match"),
        "mfp_f1": mfp.get("f1") if isinstance(mfp, dict) else None,
        "head_ap": val.get("head_macro_ap"),
        "tail_ap": val.get("tail_macro_ap"),
        "wait_pct": wait_pct,
    }


def _fmt_pct(v):
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v, fmt=".4f"):
    return "-" if v is None else format(v, fmt)


def _fmt_dur(sec):
    if sec is None:
        return "-"
    sec = int(max(0, round(sec)))
    if sec < 60:
        return f"{sec}s"
    mins, _ = divmod(sec, 60)
    if mins < 60:
        return f"{mins}m{sec % 60:02d}s" if mins < 10 else f"{mins}m"
    hrs, m = divmod(mins, 60)
    return f"{hrs}h{m:02d}m"


def _gap_str(tr_acc, val_acc):
    if tr_acc is None or val_acc is None:
        return "-"
    return f"{(tr_acc - val_acc) * 100:+.1f}"


def build_rich_view(run_dir: Path, rows: list, view: str = "core"):
    table = Table(box=box.ROUNDED, expand=True, header_style="bold cyan")
    best_overall = max((r["best_acc"] or -1.0 for r in rows), default=-1.0)

    def _base_cols():
        table.add_column("Arm", style="bold white", no_wrap=True)
        table.add_column("State", justify="center")
        table.add_column("Epoch", justify="right")

    def _row_prefix(r):
        st = r["status"]
        st_text = Text(st, style="bold green" if st == "DONE" else "bold yellow" if st == "RUN" else "dim")
        ep_str = f"{r['epoch']}/{r['planned']}" if r["planned"] else str(r["epoch"])
        return [r["name"], st_text, ep_str]

    def _best_str(r):
        s = _fmt_pct(r["best_acc"])
        if r["best_acc"] is not None and abs(r["best_acc"] - best_overall) < 1e-9 and best_overall > 0:
            return f"* {s}"
        return s

    _base_cols()
    if view == "core":
        for col in ("EpTime", "Elapsed", "ETA", "LR", "TrLoss", "ValLoss", "ValAcc", "BestAcc", "ValF1", "ValAP", "Best@Ep", "Stale", "VRAM"):
            table.add_column(col, justify="right", style=("bold green" if col == "BestAcc" else "cyan" if col == "ETA" else None))
        for r in rows:
            stale = f"{r['stale']}/{r['patience_lim']}" if r["patience_lim"] else str(r["stale"])
            mem = "-" if r["mem_gb"] is None else f"{r['mem_gb']:.1f}G"
            table.add_row(
                *_row_prefix(r), _fmt_dur(r["ep_sec"]), _fmt_dur(r["elapsed_sec"]), _fmt_dur(r["eta_sec"]),
                _fmt_num(r["lr"], ".1e"), _fmt_num(r["tr_loss"]), _fmt_num(r["val_loss"]),
                _fmt_pct(r["val_acc"]), _best_str(r), _fmt_pct(r["val_f1"]), _fmt_pct(r["val_ap"]),
                "-" if r["best_ep"] is None else f"@{r['best_ep']}", stale, mem,
            )
    elif view == "full":
        for col in ("EpTime", "ETA", "TrLoss", "ValLoss", "TrAcc", "ValAcc", "BestAcc", "Gap", "Prec", "Rec", "MFP-F1", "ValAP", "Best@Ep", "Stale", "Wait", "VRAM"):
            table.add_column(col, justify="right", style=("bold green" if col == "BestAcc" else None))
        for r in rows:
            stale = f"{r['stale']}/{r['patience_lim']}" if r["patience_lim"] else str(r["stale"])
            mem = "-" if r["mem_gb"] is None else f"{r['mem_gb']:.1f}G"
            table.add_row(
                *_row_prefix(r), _fmt_dur(r["ep_sec"]), _fmt_dur(r["eta_sec"]),
                _fmt_num(r["tr_loss"]), _fmt_num(r["val_loss"]),
                _fmt_pct(r["tr_acc"]), _fmt_pct(r["val_acc"]), _best_str(r),
                _gap_str(r["tr_acc"], r["val_acc"]), _fmt_pct(r["prec"]), _fmt_pct(r["rec"]),
                _fmt_pct(r["mfp_f1"]), _fmt_pct(r["val_ap"]),
                "-" if r["best_ep"] is None else f"@{r['best_ep']}", stale, _fmt_pct(r["wait_pct"]), mem,
            )
    elif view == "loss":
        for col in ("LR", "TrTotal", "PathA", "PathB", "PathC", "SymRank", "RDrop", "ValLoss", "TrAcc", "ValAcc", "Gap(pp)", "BestAcc"):
            table.add_column(col, justify="right", style=("bold green" if col == "BestAcc" else None))
        for r in rows:
            table.add_row(
                *_row_prefix(r), _fmt_num(r["lr"], ".1e"), _fmt_num(r["tr_loss"]),
                _fmt_num(r["loss_a"]), _fmt_num(r["loss_b"]), _fmt_num(r["loss_c"]),
                _fmt_num(r["sym_rank"]), _fmt_num(r["rdrop"], ".5f"), _fmt_num(r["val_loss"]),
                _fmt_pct(r["tr_acc"]), _fmt_pct(r["val_acc"]), _gap_str(r["tr_acc"], r["val_acc"]), _best_str(r),
            )
    elif view == "mfp":
        for col in ("ValAcc", "BestAcc", "Prec", "Rec", "Exact", "ValF1", "MFP-F1", "ValAP", "HeadAP", "TailAP", "Best@Ep", "Stale"):
            table.add_column(col, justify="right", style=("bold green" if col == "BestAcc" else None))
        for r in rows:
            stale = f"{r['stale']}/{r['patience_lim']}" if r["patience_lim"] else str(r["stale"])
            table.add_row(
                *_row_prefix(r), _fmt_pct(r["val_acc"]), _best_str(r),
                _fmt_pct(r["prec"]), _fmt_pct(r["rec"]), _fmt_pct(r["exact"]),
                _fmt_pct(r["val_f1"]), _fmt_pct(r["mfp_f1"]), _fmt_pct(r["val_ap"]),
                _fmt_pct(r["head_ap"]), _fmt_pct(r["tail_ap"]),
                "-" if r["best_ep"] is None else f"@{r['best_ep']}", stale,
            )

    done_n = sum(1 for r in rows if r["status"] == "DONE")
    title = f"[bold cyan]MFP-Nova Monitor[/] ([magenta]--view {view}[/]) | [white]{run_dir.name}[/] | Done {done_n}/{len(rows)} | {time.strftime('%H:%M:%S')}"
    return Panel(table, title=title, border_style="cyan", subtitle="[dim]Views: --view core | full | loss | mfp[/]")


def render_plain(run_dir: Path, rows: list, view: str = "core"):
    header = f"=== MFP-Nova Monitor [{view}] | {run_dir.name} | {time.strftime('%H:%M:%S')} ==="
    cols = f"{'Arm':<20} {'State':<5} {'Epoch':>7} {'EpTime':>7} {'ETA':>7} {'TrLoss':>7} {'ValLoss':>7} {'ValAcc':>7} {'BestAcc':>8} {'MFP-F1':>7} {'ValAP':>7} {'Best':>5}"
    out = [header, cols, "-" * len(cols)]
    for r in rows:
        ep = f"{r['epoch']}/{r['planned']}" if r["planned"] else str(r["epoch"])
        best_ep_s = "-" if r["best_ep"] is None else f"@{r['best_ep']}"
        out.append(
            f"{r['name']:<20} {r['status']:<5} {ep:>7} {_fmt_dur(r['ep_sec']):>7} {_fmt_dur(r['eta_sec']):>7} "
            f"{_fmt_num(r['tr_loss']):>7} {_fmt_num(r['val_loss']):>7} {_fmt_pct(r['val_acc']):>7} "
            f"{_fmt_pct(r['best_acc']):>8} {_fmt_pct(r['mfp_f1']):>7} {_fmt_pct(r['val_ap']):>7} {best_ep_s:>5}"
        )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Live Rich dashboard for MFP-Nova runs")
    parser.add_argument("run_dir", type=Path, help="Experiment directory under runs/ (or single model directory)")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--view", choices=VIEWS, default="core", help="Display view preset")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"Directory not found: {args.run_dir}")

    def collect():
        return [parse_arm(name, mdir) for name, mdir in discover_arms(args.run_dir)]

    if not HAS_RICH:
        print(render_plain(args.run_dir, collect(), args.view))
        return

    console = Console()
    if args.once:
        console.print(build_rich_view(args.run_dir, collect(), args.view))
        return

    with Live(build_rich_view(args.run_dir, collect(), args.view), console=console, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(args.interval)
                rows = collect()
                live.update(build_rich_view(args.run_dir, rows, args.view))
                if rows and all(r["status"] == "DONE" for r in rows):
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
