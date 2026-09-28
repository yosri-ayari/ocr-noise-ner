"""Measure how NER performance degrades as synthetic OCR noise increases.

Examples:
    python run_experiment.py --limit 200                  # quick check, a few minutes
    python run_experiment.py                              # full run, mixed noise
    python run_experiment.py --edit-type deletion         # only deletions
    python run_experiment.py --position first --rates 0,0.25,0.5,1
    python run_experiment.py --preset fr

Unless --out is given, results go to results/<preset>_<edit type>_<position>.csv
"""

import argparse
import csv
import time
from pathlib import Path

from labels import score, to_iob2
from noise import EDIT_MIXES, POSITIONS, corrupt_corpus

DEFAULT_RATES = "0,0.02,0.05,0.10,0.15,0.20,0.30"


def run(sentences, gold, tagger, rates, n_seeds, keep_types, out_csv, examples_path=None,
        edit_type="mixed", position="any"):
    """Sweep noise rates and seeds, write one CSV row per (rate, seed) and return the rows."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keep = set(keep_types)
    rows, examples = [], []
    fieldnames = None

    for rate in rates:
        seeds = [0] if rate == 0 else list(range(n_seeds))  # rate 0 is deterministic
        for seed in seeds:
            t0 = time.time()
            noisy, cer, words_changed = corrupt_corpus(sentences, rate, seed, edit_type, position)
            predictions = [to_iob2(p, keep) for p in tagger.predict(noisy)]
            metrics = score(gold, predictions)

            row = {"edit_type": edit_type, "position": position,
                   "rate": rate, "seed": seed, "cer": round(cer, 4),
                   "words_changed": round(words_changed, 4)}
            for name, values in metrics.items():
                for key in ("precision", "recall", "f1"):
                    row[f"{name}_{key}"] = round(values[key], 4)
            rows.append(row)

            if fieldnames is None:
                fieldnames = list(row)
                with out_csv.open("w", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
            with out_csv.open("a", newline="") as f:  # written as we go, safe if interrupted
                csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

            print(f"rate={rate:.2f} seed={seed} CER={cer:.3f} "
                  f"micro F1={metrics['micro']['f1']:.3f} ({time.time() - t0:.0f}s)")
            if seed == 0:
                examples.append((rate, cer, noisy[:3]))

    if examples_path:
        with Path(examples_path).open("w", encoding="utf-8") as f:
            for rate, cer, sample in examples:
                f.write(f"=== rate {rate:.2f} (measured CER {cer:.3f}) ===\n")
                for tokens in sample:
                    f.write(" ".join(tokens) + "\n")
                f.write("\n")
    return rows


def main():
    from ner_eval import PRESETS, Tagger, load_corpus

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="en")
    parser.add_argument("--dataset", help="overrides the preset")
    parser.add_argument("--config", help="overrides the preset")
    parser.add_argument("--split", help="overrides the preset")
    parser.add_argument("--model", help="overrides the preset")
    parser.add_argument("--rates", default=DEFAULT_RATES, help="comma-separated edit rates")
    parser.add_argument("--seeds", type=int, default=3, help="noise seeds per rate")
    parser.add_argument("--edit-type", choices=sorted(EDIT_MIXES), default="mixed")
    parser.add_argument("--position", choices=POSITIONS, default="any")
    parser.add_argument("--limit", type=int, help="only use the first N sentences")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", help="CSV path (default: results/<preset>_<edit type>_<position>.csv)")
    args = parser.parse_args()

    preset = dict(PRESETS[args.preset])
    for key in ("dataset", "config", "split", "model"):
        if getattr(args, key):
            preset[key] = getattr(args, key)

    print(f"Corpus: {preset['dataset']} ({preset['config']}, {preset['split']})  Model: {preset['model']}")
    sentences, gold = load_corpus(preset["dataset"], preset["config"], preset["split"],
                                  preset["keep_types"], args.limit)
    print(f"{len(sentences)} sentences loaded")
    tagger = Tagger(preset["model"], batch_size=args.batch_size)
    rates = [float(r) for r in args.rates.split(",")]
    out = Path(args.out or f"results/{args.preset}_{args.edit_type}_{args.position}.csv")
    print(f"Noise: edit type {args.edit_type}, position {args.position}")
    run(sentences, gold, tagger, rates, args.seeds, preset["keep_types"], out,
        examples_path=out.with_name(out.stem + "_examples.txt"),
        edit_type=args.edit_type, position=args.position)
    print(f"Results written to {out}. Next: python plot_results.py {out}")


if __name__ == "__main__":
    main()
