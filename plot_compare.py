"""Overlay several experiments on one figure (micro F1, mean over seeds).

    python plot_compare.py --x cer --out results/compare_edit_types.png \\
        results/en_deletion_any.csv results/en_substitution_any.csv results/en_insertion_any.csv

    python plot_compare.py --x words_changed --out results/compare_positions.png \\
        results/en_substitution_first.csv results/en_substitution_inner.csv

The legend names each line with the edit type and position stored in its CSV.
Use --x cer to compare at equal character error rate, or --x words_changed to
compare at equal share of corrupted words (the fair axis for first vs inner).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

from plot_results import GRID, INK, MUTED, summarise  # noqa: E402

# Fixed categorical order, one slot per file in the order given.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D"), ("#e87ba4", "v")]
X_LABELS = {"cer": "Measured character error rate (CER)", "words_changed": "Share of words corrupted"}


def label_for(df, path):
    if {"edit_type", "position"} <= set(df.columns):
        edit_type, position = df["edit_type"].iloc[0], df["position"].iloc[0]
        return edit_type if position == "any" else f"{edit_type}, {position}"
    return Path(path).stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--x", choices=sorted(X_LABELS), default="cer")
    parser.add_argument("--out", default="results/compare.png")
    parser.add_argument("--title", default="Micro F1 by noise type")
    args = parser.parse_args()
    if len(args.csvs) > len(SERIES):
        parser.error(f"at most {len(SERIES)} files per figure")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x_max = 0.0
    for (color, marker), path in zip(SERIES, args.csvs):
        df = pd.read_csv(path)
        summary = summarise(df)
        label = label_for(df, path)
        x, y = summary[args.x], summary["micro_f1"]
        ax.plot(x, y, color=color, marker=marker, linewidth=2, markersize=6, label=label)
        x_max = max(x_max, x.max())

    ax.set_title(args.title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(X_LABELS[args.x], color=MUTED)
    ax.set_ylabel("Entity-level F1 (micro)", color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_xlim(right=x_max * 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Figure written to {args.out}")


if __name__ == "__main__":
    main()
