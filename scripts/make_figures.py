"""Render the README charts from a benchmark results file.

    python scripts/make_figures.py [--results docs/results/benchmarks.json] [--out docs/assets]

Every number drawn comes from the JSON written by ``pagellm bench all --json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Validated reference palette (light mode): categorical slots 1 and 2, text inks, surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY = "#b9b8b2"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titleweight": "bold",
        "axes.titlesize": 12.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)


def _bar_labels(ax, bars, fmt: str) -> None:
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=10,
        )


def _save(fig, out: Path, name: str, note: str) -> None:
    fig.text(0.01, 0.01, note, color=INK_2, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print(f"wrote {out / name}")


def batching(r: dict, out: Path) -> None:
    s, c = r["static"], r["continuous"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    bars = ax.bar(
        ["static", "continuous"],
        [s["steps"], c["steps"]],
        color=[GRAY, BLUE],
        width=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    _bar_labels(ax, bars, "{:.0f}")
    ax.set_title("Engine steps (forward passes)")
    ax.set_ylim(0, s["steps"] * 1.18)
    ax = axes[1]
    bars = ax.bar(
        ["static", "continuous"],
        [s["mean_batch"], c["mean_batch"]],
        color=[GRAY, BLUE],
        width=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    _bar_labels(ax, bars, "{:.1f}")
    ax.set_title("Mean tokens per step")
    ax.set_ylim(0, c["mean_batch"] * 1.18)
    t = r["trace"]
    _save(
        fig,
        out,
        "batching.png",
        f"Same {t['requests']}-request Poisson trace, max {t['max_num_seqs']} concurrent, greedy. "
        f"Continuous batching: {s['steps'] / c['steps']:.2f}x fewer forward passes.",
    )


def memory(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    names = ["contiguous", "paged"]
    used = [100 - r["contiguous_waste_pct"], 100 - r["paged_waste_pct"]]
    waste = [r["contiguous_waste_pct"], r["paged_waste_pct"]]
    ax.barh(
        names,
        used,
        color=BLUE,
        height=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="holds real tokens",
    )
    ax.barh(
        names,
        waste,
        left=used,
        color=GRAY,
        height=0.5,
        edgecolor=SURFACE,
        linewidth=2,
        label="reserved, never used",
    )
    for i, (u, w) in enumerate(zip(used, waste)):
        ax.text(u / 2, i, f"{u:.1f}%", ha="center", va="center", color="white", fontsize=10)
        if w >= 12:
            ax.text(u + w / 2, i, f"{w:.1f}%", ha="center", va="center", color=INK, fontsize=10)
        else:  # too thin to hold its label: put it just past the bar end
            ax.text(101, i, f"{w:.1f}%", ha="left", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("KV cache slots")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax = axes[1]
    vals = [r["contiguous_max_concurrent"], r["paged_max_concurrent"]]
    bars = ax.bar(names, vals, color=[GRAY, BLUE], width=0.55, edgecolor=SURFACE, linewidth=2)
    _bar_labels(ax, bars, "{:.0f}")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("Max concurrent sequences")
    _save(
        fig,
        out,
        "memory.png",
        f"Pool of {r['pool_slots']} token slots, block size {r['block_size']}, "
        f"max_model_len {r['max_model_len']}. Paged: "
        f"{r['paged_max_concurrent'] / r['contiguous_max_concurrent']:.1f}x the concurrency.",
    )


def prefix(r: dict, out: Path) -> None:
    a, b = r["no_cache"], r["prefix_cache"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    names = ["no cache", "prefix cache"]
    ax = axes[0]
    bars = ax.bar(
        names,
        [a["prefill_tokens_computed"], b["prefill_tokens_computed"]],
        color=[GRAY, BLUE],
        width=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    _bar_labels(ax, bars, "{:.0f}")
    ax.set_ylim(0, a["prefill_tokens_computed"] * 1.18)
    ax.set_title("Prompt tokens computed")
    ax = axes[1]
    bars = ax.bar(
        names,
        [a["seconds"], b["seconds"]],
        color=[GRAY, BLUE],
        width=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    _bar_labels(ax, bars, "{:.2f} s")
    ax.set_ylim(0, a["seconds"] * 1.2)
    ax.set_title("Wall time")
    share = 100 * b["prompt_tokens_from_cache"] / a["prefill_tokens_computed"]
    _save(
        fig,
        out,
        "prefix.png",
        f"32 requests sharing a 192-token system prompt. {share:.0f}% of prompt tokens served "
        f"from cache; outputs identical: {r['outputs_identical']}.",
    )


def speculative(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for label, color in (("greedy", BLUE), ("T=0.8", ORANGE)):
        runs = [x for x in r["runs"] if x["sampling"] == label]
        ks = [x["k"] for x in runs]
        for ax, key in ((axes[0], "tokens_per_target_forward"), (axes[1], "acceptance_rate")):
            ys = [x[key] for x in runs]
            ax.plot(
                ks,
                ys,
                color=color,
                linewidth=2,
                marker="o",
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=label,
            )
            ax.annotate(
                f"{ys[-1]:.2f}",
                (ks[-1], ys[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=10,
            )
    axes[0].set_title("Tokens per target forward")
    axes[1].set_title("Acceptance rate")
    for ax in axes:
        ax.set_xlabel("draft tokens per step (k)")
        ax.set_xticks(sorted({x["k"] for x in r["runs"]}))
        ax.legend(loc="best")
    axes[0].set_ylim(1, None)
    axes[1].set_ylim(0, 1)
    _save(
        fig,
        out,
        "speculative.png",
        f"Target {r['target_params'] / 1e6:.2f}M params, draft {r['draft_params'] / 1e6:.3f}M. "
        "Output distribution equals the target's (tested); fewer target forwards per token.",
    )


def scaling(r: dict, out: Path) -> None:
    rows = r["rows"]
    bs = [x["batch"] for x in rows]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    series = {key: [x[key]["ms_per_token"] for x in rows] for key in ("per_sequence", "batched")}
    upper = max(series, key=lambda k: series[k][-1])
    for key, label, color in (
        ("per_sequence", "per-sequence attention", GRAY),
        ("batched", "batched decode attention", BLUE),
    ):
        ys = series[key]
        ax.plot(
            bs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
        )
        dy = 9 if key == upper else -9  # nudge apart so close end points never collide
        ax.annotate(
            label,
            (bs[-1], ys[-1]),
            xytext=(10, dy),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=10,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs, [str(b) for b in bs])
    ax.set_xlim(bs[0] * 0.8, bs[-1] * 4)
    ax.set_ylim(0, None)
    ax.set_xlabel("sequences in the batch")
    ax.set_ylabel("ms per generated token")
    ax.set_title("Decode cost per token vs batch size")
    ax.legend(loc="upper right")
    _save(
        fig,
        out,
        "scaling.png",
        f"Decode-only steps at ~{r['context_tokens']}-token context, best of 3 runs. "
        "Lower is better.",
    )


def chunked(r: dict, out: Path) -> None:
    runs = r["runs"]
    labels = [
        x["label"].replace(" (budget 256)", "\n(budget 256)").replace(", budget", "\nbudget")
        for x in runs
    ]
    colors = [GRAY] + [BLUE] * (len(runs) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, key, title, fmt in (
        (axes[0], "step_p99_ms", "p99 step time (ms)", "{:.1f}"),
        (axes[1], "itl_p99_ms", "p99 inter-token latency (ms)", "{:.1f}"),
        (axes[2], "ttft_mean_s", "Mean time to first token (s)", "{:.2f}"),
    ):
        vals = [x[key] for x in runs]
        bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars, fmt)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=9)
    t = r["trace"]
    _save(
        fig,
        out,
        "chunked.png",
        f"{t['requests']} requests with 96 to 224-token prompts arriving while others decode. "
        "Lower is better on every panel.",
    )


def http(r: dict, out: Path) -> None:
    runs = r["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    pcts = ["p50", "p95", "p99"]
    width = 0.36
    for ax, key, title in (
        (axes[0], "itl_ms", "Inter-token latency (ms)"),
        (axes[1], "ttft_ms", "Time to first token (ms)"),
    ):
        for i, (run, color) in enumerate(zip(runs, (GRAY, BLUE))):
            xs = [j + (i - 0.5) * width for j in range(len(pcts))]
            vals = [run[key][p] for p in pcts]
            bars = ax.bar(
                xs,
                vals,
                width=width,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                label=run["label"],
            )
            _bar_labels(ax, bars, "{:.0f}")
        ax.set_xticks(range(len(pcts)), pcts)
        ax.set_title(title)
        ax.set_ylim(0, max(max(x[key][p] for p in pcts) for x in runs) * 1.22)
        ax.legend(loc="upper left")
    _save(
        fig,
        out,
        "http.png",
        f"{runs[0]['requests']} streaming requests, {runs[0]['concurrency']} concurrent clients, "
        "real HTTP against `pagellm serve` in a subprocess.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "docs/results/benchmarks.json")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/assets")
    a = ap.parse_args()
    data = json.loads(a.results.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("batching", batching),
        ("memory", memory),
        ("prefix", prefix),
        ("spec", speculative),
        ("scaling", scaling),
        ("chunked", chunked),
        ("http", http),
    ):
        if name in data:
            fn(data[name], a.out)


if __name__ == "__main__":
    main()
