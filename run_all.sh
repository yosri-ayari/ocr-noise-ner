#!/usr/bin/env bash
# Runs every experiment of the README, then draws all the figures.
# Usage:  bash run_all.sh            (full test split)
#         bash run_all.sh 200        (only the first 200 sentences, for a quick check)
set -euo pipefail

PRESET=en
LIMIT=${1:+--limit $1}
ANY_RATES=0,0.02,0.05,0.10,0.15,0.20,0.30   # probability that each character is edited
WORD_RATES=0,0.10,0.25,0.50,0.75,1.0        # probability that each word gets one edit

# Q1. Overall degradation, all edit types mixed
python run_experiment.py --preset $PRESET $LIMIT --rates $ANY_RATES

# Q2. Which operation hurts most: deletion, substitution or insertion
for EDIT in deletion substitution insertion; do
    python run_experiment.py --preset $PRESET $LIMIT --rates $ANY_RATES --edit-type $EDIT
done

# Q3. Does the position of the error matter: first character or elsewhere
for POSITION in first inner; do
    python run_experiment.py --preset $PRESET $LIMIT --rates $WORD_RATES --edit-type substitution --position $POSITION
done
python run_experiment.py --preset $PRESET $LIMIT --rates $WORD_RATES --edit-type case --position first

# Figures
python plot_results.py results/${PRESET}_mixed_any.csv
python plot_compare.py --x cer --out results/compare_edit_types.png \
    --title "Micro F1 by edit operation" \
    results/${PRESET}_deletion_any.csv results/${PRESET}_substitution_any.csv results/${PRESET}_insertion_any.csv
python plot_compare.py --x words_changed --out results/compare_positions.png \
    --title "Micro F1 by error position (one edit per corrupted word)" \
    results/${PRESET}_substitution_first.csv results/${PRESET}_substitution_inner.csv results/${PRESET}_case_first.csv
