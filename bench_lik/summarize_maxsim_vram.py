"""Summarize the per-op MaxSim VRAM sweep: markdown table + log-log plot.

Usage:
    python bench_lik/summarize_maxsim_vram.py --results-dir /tmp/lik_vram_results2 \\
        --plot bench_lik/maxsim_vram.png
"""

import argparse
import glob
import json
from pathlib import Path


def _fmt_mib(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:.0f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    cells: dict[tuple[int, str], dict] = {}
    for path in glob.glob(str(args.results_dir / "maxsim_vram_*.json")):
        data = json.load(open(path))
        summary = data["maxsim_isolated_summary"]
        cells[(data["batch_size"], data["colpali_scores_backend"])] = {
            "saved": summary["max_saved_for_backward_mib"],
            "bwd": summary["max_backward_transient_peak_mib"],
            "step_peak": data["step_peak_alloc_mib"],
            "oom": data["oom"],
            "oom_message": data.get("oom_message"),
        }

    batch_sizes = sorted({batch for batch, _ in cells})

    print(
        "| batch size | vanilla: held | vanilla: bwd spike | vanilla: total "
        "| LIK: held | LIK: bwd spike | LIK: total | step OOM? |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for batch in batch_sizes:
        vanilla, lik = cells[(batch, "torch")], cells[(batch, "auto")]
        oom_note = "vanilla OOMs" if vanilla["oom"] else "both fit"
        print(
            f"| {batch} | {_fmt_mib(vanilla['saved'])} | {_fmt_mib(vanilla['bwd'])} "
            f"| {_fmt_mib(vanilla['saved'] + vanilla['bwd'])} | {_fmt_mib(lik['saved'])} "
            f"| {_fmt_mib(lik['bwd'])} | {_fmt_mib(lik['saved'] + lik['bwd'])} | {oom_note} |"
        )

    for batch in batch_sizes:
        message = cells[(batch, "torch")].get("oom_message")
        if message:
            print(f"\nOOM message (vanilla B={batch}): {message}")

    if args.plot is None:
        return

    # Imported here so the table works in a venv without matplotlib.
    import matplotlib.pyplot as plt

    vanilla_totals = [cells[(b, "torch")]["saved"] + cells[(b, "torch")]["bwd"] for b in batch_sizes]
    lik_totals = [cells[(b, "auto")]["saved"] + cells[(b, "auto")]["bwd"] for b in batch_sizes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(batch_sizes, vanilla_totals, "o-", color="tab:red", label="vanilla (torch einsum)")
    ax.plot(batch_sizes, lik_totals, "s-", color="tab:green", label="LIK (fused kernel)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(batch_sizes, [str(b) for b in batch_sizes])
    tick_values = [8, 32, 128, 512, 2048, 8192]
    ax.set_yticks(tick_values, [_fmt_mib(v) for v in tick_values])
    ax.minorticks_off()
    ax.set_xlabel("per-device batch size (log scale)")
    ax.set_ylabel("MaxSim op VRAM: held + backward spike\n(log scale)")
    ax.set_title("VRAM attributable to the MaxSim op")
    ax.grid(True, which="both", alpha=0.3)
    # Slope annotations: vanilla quadruples per doubling (B² score grid), LIK doubles (grad_D).
    ax.annotate(
        "×4 per doubling (B²)",
        xy=(batch_sizes[-2], vanilla_totals[-2]),
        xytext=(-10, 14),
        textcoords="offset points",
        color="tab:red",
        ha="right",
    )
    ax.annotate(
        "×2 per doubling (grad_D, linear)",
        xy=(batch_sizes[-2], lik_totals[-2]),
        xytext=(-10, 14),
        textcoords="offset points",
        color="tab:green",
        ha="right",
    )
    ratio = vanilla_totals[-1] / lik_totals[-1]
    ax.annotate(
        f"{ratio:.0f}× at B={batch_sizes[-1]}",
        xy=(batch_sizes[-1], vanilla_totals[-1]),
        xytext=(-60, -8),
        textcoords="offset points",
        color="black",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"\nWrote plot to {args.plot}")


if __name__ == "__main__":
    main()
