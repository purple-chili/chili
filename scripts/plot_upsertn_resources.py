#!/usr/bin/env python3
"""Create before/after upsertn charts from a paired resource benchmark.

Requires matplotlib. Run: python scripts/plot_upsertn_resources.py <results-dir>
Outputs PNG and SVG charts plus comparison.csv in the results directory.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def short(value):
    return f"{value // 1000000}M" if value >= 1000000 else f"{value // 1000}k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.results / "summary.json").read_text())
    metadata = json.loads((args.results / "metadata.json").read_text())
    lookup = {(r["variant"], r["batch"], r["keep"]): r for r in rows if r["method"] == "upsertn"}
    batches = metadata["batches"]
    keeps = metadata["keeps"]
    assert all((variant, b, k) in lookup for variant in ("baseline", "current") for b in batches for k in keeps)
    metrics = [
        ("wall_us_per_update", "Elapsed time per update", "µs / update (log scale)", "elapsed-time", True),
        ("cpu_us_per_update", "CPU time per update", "µs / update (log scale)", "cpu-time", True),
        ("peak_rss_mib", "Peak resident memory", "Peak process RSS (MiB)", "peak-memory", False),
    ]
    variants = [("baseline", "Before: append then trim", "#777777"),
                ("current", "After: selective trim-first", "#087f8c")]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for metric, title, ylabel, filename, logarithmic in metrics:
        fig, axes = plt.subplots(1, len(keeps), figsize=(5 * len(keeps), 4.8), squeeze=False, sharey=True)
        maximum = max(r[metric + "_max"] for r in lookup.values())
        minimum = min(r[metric + "_min"] for r in lookup.values())
        for ax, keep in zip(axes[0], keeps):
            for i, (variant, label, color) in enumerate(variants):
                samples = [lookup[variant, batch, keep] for batch in batches]
                vals = [r[metric] for r in samples]
                error = [[r[metric] - r[metric + "_min"] for r in samples],
                         [r[metric + "_max"] - r[metric] for r in samples]]
                positions = [x + (i - 0.5) * 0.36 for x in range(len(batches))]
                ax.bar(positions, vals, width=0.34, color=color, label=label,
                       yerr=error, capsize=3, error_kw={"linewidth": 0.8}, zorder=3)
                for x, value, sample in zip(positions, vals, samples):
                    upper = sample[metric + "_max"]
                    y = upper * 1.16 if logarithmic else upper + maximum * 0.025
                    ax.text(x, y, f"{value:,.1f}", ha="center", va="bottom", fontsize=8, rotation=45 if logarithmic else 0)
            if logarithmic:
                ax.set_yscale("log")
                ax.set_ylim(max(0.1, minimum * 0.5), maximum * 3)
            else:
                ax.set_ylim(0, maximum * 1.2)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_title(f"Retain {keep:,} rows", fontsize=11)
            ax.set_xticks(range(len(batches)), [short(b) for b in batches])
            ax.set_xlabel("Incoming rows per update")
            ax.grid(axis="y", alpha=0.2, zorder=0)
        axes[0, 0].set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"upsertn — {title}", fontsize=15, y=0.98)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
        period = f"{metadata['started_at'][:10]} {metadata['started_at'][11:19]}–{metadata['finished_at'][11:19]} UTC"
        footer = (f"Source: local Chili benchmark · {period} · {metadata['cpu_model']}\n"
                  f"Median of {metadata['repetitions']} fresh processes; {metadata['seconds_per_sample']:g}s/sample after 250ms warmup. "
                  "Whiskers: sample min–max. Four columns; same resident input reused. Lower is better.")
        fig.text(0.02, 0.025, footer, fontsize=8, va="bottom")
        fig.subplots_adjust(top=0.77, bottom=0.24, left=0.07, right=0.98, wspace=0.15)
        fig.savefig(args.results / f"{filename}.png", dpi=180)
        fig.savefig(args.results / f"{filename}.svg")
        plt.close(fig)
    comparisons = []
    for batch in batches:
        for keep in keeps:
            old, new = lookup["baseline", batch, keep], lookup["current", batch, keep]
            comparisons.append({
                "batch": batch, "keep": keep,
                "before_wall_us": old["wall_us_per_update"], "after_wall_us": new["wall_us_per_update"],
                "elapsed_speedup": old["wall_us_per_update"] / new["wall_us_per_update"],
                "before_cpu_us": old["cpu_us_per_update"], "after_cpu_us": new["cpu_us_per_update"],
                "before_peak_mib": old["peak_rss_mib"], "after_peak_mib": new["peak_rss_mib"],
                "peak_reduction_percent": 100 * (1 - new["peak_rss_mib"] / old["peak_rss_mib"]),
            })
    with (args.results / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
        writer.writeheader(); writer.writerows(comparisons)
    (args.results / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")


if __name__ == "__main__":
    main()
