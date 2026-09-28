"""Figures and a summary table from a results CSV.

    python plot_results.py results/en_mixed_any.csv

Writes, next to the CSV:
    <name>_f1.png         micro F1 against measured CER, mean and spread over seeds
    <name>_by_type.png    F1 per entity type against measured CER
    <name>_summary.md     a Markdown table to paste into the README
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order: the same entity type always gets the same colour and marker.
TYPE_STYLE = {
    "PER": ("#2a78d6", "o"),
    "LOC": ("#eb6834", "s"),
    "ORG": ("#1baf7a", "^"),
    "MISC": ("#eda100", "D"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def summarise(df):
    """Mean and standard deviation over seeds, one row per noise rate."""
    grouped = df.groupby("rate")
    summary = grouped.mean(numeric_only=True).drop(columns="seed")
    spread = grouped.std(numeric_only=True).fillna(0.0)
    summary["micro_f1_std"] = spread["micro_f1"]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


def entity_types(df):
    types = [c[:-3] for c in df.columns if c.endswith("_f1") and c != "micro_f1"]
    return [t for t in TYPE_STYLE if t in types] + [t for t in types if t not in TYPE_STYLE]


def style_axes(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel("Measured character error rate (CER)", color=MUTED)
    ax.set_ylabel("Entity-level F1", color=MUTED)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_micro(summary, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x, y, s = summary["cer"], summary["micro_f1"], summary["micro_f1_std"]
    ax.fill_between(x, y - s, y + s, color=TYPE_STYLE["PER"][0], alpha=0.15, linewidth=0)
    ax.plot(x, y, color=TYPE_STYLE["PER"][0], linewidth=2, marker="o", markersize=6)
    for point in (summary.iloc[0], summary.iloc[-1]):
        ax.annotate(f"{point['micro_f1']:.2f}", (point["cer"], point["micro_f1"]),
                    textcoords="offset points", xytext=(9, 4), ha="left", color=INK, fontsize=9)
    style_axes(ax, "Micro F1 under synthetic OCR noise (mean and sd over seeds)")
    ax.set_xlim(right=summary["cer"].max() * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_by_type(summary, types, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for t in types:
        color, marker = TYPE_STYLE.get(t, ("#6b6a64", "x"))
        ax.plot(summary["cer"], summary[f"{t}_f1"], color=color, marker=marker,
                linewidth=2, markersize=6, label=t)
    # Direct labels at the line ends, pushed apart so they never overlap.
    last = summary.iloc[-1]
    ends = sorted((last[f"{t}_f1"], t) for t in types)
    placed = []
    for value, t in ends:
        y = max(value, placed[-1] + 0.06) if placed else value
        placed.append(y)
        ax.text(last["cer"] + summary["cer"].max() * 0.025, y, t, va="center", color=INK, fontsize=9)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    style_axes(ax, "F1 by entity type under synthetic OCR noise")
    ax.set_xlim(right=summary["cer"].max() * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def markdown_table(summary, types):
    header = ["Edit rate", "Measured CER", "Words changed", "Micro F1 (mean ± sd)"] + [f"{t} F1" for t in types]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, r in summary.iterrows():
        cells = [f"{r['rate']:.2f}", f"{r['cer']:.1%}", f"{r['words_changed']:.1%}",
                 f"{r['micro_f1']:.3f} ± {r['micro_f1_std']:.3f}"] + [f"{r[f'{t}_f1']:.3f}" for t in types]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    summary = summarise(df)
    types = entity_types(df)
    base = csv_path.with_suffix("")
    plot_micro(summary, f"{base}_f1.png")
    plot_by_type(summary, types, f"{base}_by_type.png")
    table = markdown_table(summary, types)
    Path(f"{base}_summary.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/en_mixed_any.csv")
