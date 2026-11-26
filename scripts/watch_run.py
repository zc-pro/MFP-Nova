"""Live Rich terminal dashboard for multi-arm MFP-Nova experiments."""

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


def discover_arms(run_dir: Path):
    """Find all experiment arms under run_dir containing model/history.jsonl or history.jsonl."""
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

    planned = None
    patience_lim = None
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
            "tr_loss": None, "val_loss": None,
            "val_acc": None, "val_f1": None, "val_ap": None,
            "best_acc": None, "best_ep": None,
            "stale": 0, "patience_lim": patience_lim, "mem_gb": None,
            "ep_sec": None, "elapsed_sec": init_elapsed, "eta_sec": None,
        }

    last = lines[-1]
    val = last.get("validation", {})
    losses = last.get("losses", {})
    tr_loss = losses.get("total") if isinstance(losses, dict) else None
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

    return {
        "name": name,
        "status": "DONE" if done else "RUN",
        "epoch": cur_ep,
        "planned": planned,
        "lr": last.get("lr"),
        "tr_loss": tr_loss,
        "val_loss": last.get("validation_loss"),
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
    mins, s = divmod(sec, 60)
    if mins < 60:
        return f"{mins}m{s:02d}s" if mins < 10 else f"{mins}m"
    hrs, m = divmod(mins, 60)
    return f"{hrs}h{m:02d}m"


def build_rich_view(run_dir: Path, rows: list):
    table = Table(box=box.ROUNDED, expand=True, header_style="bold cyan")
    table.add_column("Arm", style="bold white", no_wrap=True)
    table.add_column("State", justify="center")
    table.add_column("Epoch", justify="right")
    table.add_column("EpTime", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("ETA", justify="right", style="cyan")
    table.add_column("LR", justify="right")
    table.add_column("TrLoss", justify="right")
    table.add_column("ValLoss", justify="right")
    table.add_column("ValAcc", justify="right")
    table.add_column("BestAcc", justify="right", style="bold green")
    table.add_column("ValF1", justify="right")
    table.add_column("ValAP", justify="right")
    table.add_column("Best@Ep", justify="right")
    table.add_column("Stale", justify="right")
    table.add_column("VRAM", justify="right")

    best_overall = max((r["best_acc"] or -1.0 for r in rows), default=-1.0)
    for r in rows:
        st = r["status"]
        st_text = Text(st, style="bold green" if st == "DONE" else "bold yellow" if st == "RUN" else "dim")
        ep_str = f"{r['epoch']}/{r['planned']}" if r["planned"] else str(r["epoch"])
        stale_str = f"{r['stale']}/{r['patience_lim']}" if r["patience_lim"] else str(r["stale"])
        mem_str = "-" if r["mem_gb"] is None else f"{r['mem_gb']:.1f}G"
        best_acc_str = _fmt_pct(r["best_acc"])
        if r["best_acc"] is not None and abs(r["best_acc"] - best_overall) < 1e-9 and best_overall > 0:
            best_acc_str = f"* {best_acc_str}"
        table.add_row(
            r["name"], st_text, ep_str,
            _fmt_dur(r["ep_sec"]),
            _fmt_dur(r["elapsed_sec"]),
            _fmt_dur(r["eta_sec"]),
            _fmt_num(r["lr"], ".1e"),
            _fmt_num(r["tr_loss"], ".4f"),
            _fmt_num(r["val_loss"], ".4f"),
            _fmt_pct(r["val_acc"]),
            best_acc_str,
            _fmt_pct(r["val_f1"]),
            _fmt_pct(r["val_ap"]),
            "-" if r["best_ep"] is None else f"@{r['best_ep']}",
            stale_str,
            mem_str,
        )

    done_n = sum(1 for r in rows if r["status"] == "DONE")
    title = f"[bold cyan]MFP-Nova Monitor[/] | [white]{run_dir.name}[/] | Done {done_n}/{len(rows)} | {time.strftime('%H:%M:%S')}"
    return Panel(table, title=title, border_style="cyan")


def render_plain(run_dir: Path, rows: list):
    header = f"=== MFP-Nova Monitor | {run_dir.name} | {time.strftime('%H:%M:%S')} (install rich for live color UI) ==="
    cols = (
        f"{'Arm':<20} {'State':<5} {'Epoch':>7} {'EpTime':>7} {'Elapsed':>8} {'ETA':>7} "
        f"{'LR':>8} {'TrLoss':>7} {'ValLoss':>7} {'ValAcc':>7} {'BestAcc':>8} {'ValF1':>7} {'ValAP':>7} {'Best':>5} {'VRAM':>6}"
    )
    out = [header, cols, "-" * len(cols)]
    for r in rows:
        ep = f"{r['epoch']}/{r['planned']}" if r["planned"] else str(r["epoch"])
        mem = "-" if r["mem_gb"] is None else f"{r['mem_gb']:.1f}G"
        best_ep_s = "-" if r["best_ep"] is None else f"@{r['best_ep']}"
        out.append(
            f"{r['name']:<20} {r['status']:<5} {ep:>7} {_fmt_dur(r['ep_sec']):>7} "
            f"{_fmt_dur(r['elapsed_sec']):>8} {_fmt_dur(r['eta_sec']):>7} {_fmt_num(r['lr'], '.1e'):>8} "
            f"{_fmt_num(r['tr_loss']):>7} {_fmt_num(r['val_loss']):>7} {_fmt_pct(r['val_acc']):>7} "
            f"{_fmt_pct(r['best_acc']):>8} {_fmt_pct(r['val_f1']):>7} {_fmt_pct(r['val_ap']):>7} "
            f"{best_ep_s:>5} {mem:>6}"
        )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Live Rich dashboard for MFP-Nova runs")
    parser.add_argument("run_dir", type=Path, help="Experiment directory under runs/ (or single model directory)")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"Directory not found: {args.run_dir}")

    def collect():
        return [parse_arm(name, mdir) for name, mdir in discover_arms(args.run_dir)]

    if not HAS_RICH:
        print(render_plain(args.run_dir, collect()))
        return

    console = Console()
    if args.once:
        console.print(build_rich_view(args.run_dir, collect()))
        return

    with Live(build_rich_view(args.run_dir, collect()), console=console, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                time.sleep(args.interval)
                rows = collect()
                live.update(build_rich_view(args.run_dir, rows))
                if rows and all(r["status"] == "DONE" for r in rows):
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
