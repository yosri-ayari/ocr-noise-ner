# How robust is transformer-based NER to OCR errors?

Controlled experiments measuring how character-level OCR errors degrade a
pre-trained transformer model for named entity recognition (NER), and which
kinds of errors matter most.

## Background

Digitised historical archives are read through OCR and HTR, and their output is
noisy. Named entities (people, places, organisations) are the anchor points for
searching and linking these documents, so any downstream analysis depends on how
robust entity extraction is to that noise.

Prior work has shown that OCR errors substantially degrade NER. On CoNLL-2003
[4] rendered as degraded images and read back with an OCR engine, a
BiLSTM-CNN-CRF model [2] trained on clean text drops from **90.90** micro F1 on
clean text to **87.45** at a character error rate (CER) of 1.7%, **70.82** at
6.9% and **60.31** at 41.3% [1]. Analysing the errors produced by the OCR
engine, the same study reports that:

- among entities hit by a single error, **deletions are handled well** (79.5%
  still correctly recognised), unlike substitutions and insertions;
- **errors on the first character are very damaging**: only 30.41% of the
  affected entities are still correctly recognised;
- segmentation errors (merged or split words) have a very strong impact.

These observations come from the errors a real OCR engine happened to make. This
project asks whether they hold for a transformer model [3], and tests them under
controlled noise: each factor (edit operation, error position) is varied while
everything else stays fixed.

## Research questions

1. **Overall robustness.** How much does a pre-trained transformer
   (`dslim/bert-base-NER`) degrade as the CER increases, compared with the
   figures reported for a BiLSTM-CNN-CRF model [1]?
2. **Edit operations.** At equal CER, do deletions hurt less than substitutions
   and insertions?
3. **Error position.** With one edit per corrupted word, does an error on the
   first character hurt more than the same error elsewhere in the word? Does a
   simple case change of the first letter (Paris to paris) suffice?
4. **Entity types.** Are persons, locations and organisations equally fragile?

## Method

**Corpus and model.** CoNLL-2003 English test split [4], with
`dslim/bert-base-NER` used as is (inference only, trained on clean text). A
French preset (WikiANN, `Jean-Baptiste/camembert-ner`) is also available with
`--preset fr`.

**Noise model** (`noise.py`), controlled by two options:

| Option | Values | Meaning |
|---|---|---|
| `--edit-type` | `mixed`, `substitution`, `deletion`, `insertion`, `case` | which operations are applied. `mixed` combines visually plausible confusions (`rn` to `m`, `l` to `1`, `e` to `c`, long s), random substitutions, deletions and insertions |
| `--position` | `any`, `first`, `inner` | `any`: every character can be edited. `first` / `inner`: a corrupted word receives exactly one edit, on its first character or on another one |

With the same seed, `first` and `inner` corrupt exactly the same words, so the
two conditions form a paired comparison where only the position changes.
Punctuation-only tokens are never corrupted. The reported CER is measured with
the Levenshtein distance.

**Design choice.** Noise never creates or removes a word boundary, so gold
labels stay aligned with the noisy words. Segmentation errors are therefore not
simulated (see Limitations).

**Evaluation** (`labels.py`). Entity-level precision, recall and F1 with exact
match on span and type, as in CoNLL. Gold and predicted tags are normalised to
IOB2 first. Each word takes the prediction of its first sub-token. Every noise
level is run with 3 seeds; figures show means (and standard deviations for the
main curve).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover tests   # checks, no download needed
bash run_all.sh 200                 # every experiment on 200 sentences: a quick check
bash run_all.sh                     # every experiment on the full test split
```

`run_all.sh` runs each experiment with `run_experiment.py` and draws the
figures with `plot_results.py` and `plot_compare.py`. All outputs go to
`results/`: one CSV per experiment (one row per noise level and seed), figures,
a Markdown summary table and noisy example sentences.

## Results

### Q1. Overall degradation

<!-- TODO: paste results/en_mixed_any_summary.md here, and add a line comparing
your F1 at CER around 2% and 7% with the 87.45 and 70.82 reported in [1]. -->

![Micro F1 against CER](results/en_mixed_any_f1.png)

### Q2. Edit operations

![Micro F1 by edit operation](results/compare_edit_types.png)

### Q3. Error position

![Micro F1 by error position](results/compare_positions.png)

### Q4. Entity types

![F1 per entity type](results/en_mixed_any_by_type.png)

## Discussion

<!-- TODO, in your own words:
- Q1: does the transformer hold up better than the BiLSTM-CNN-CRF? Careful: the
  models, the noise and the pipeline all differ, so this is an indication, not
  a controlled comparison.
- Q2 and Q3: are the earlier observations reproduced under controlled noise? If
  not, what could explain the difference (observational vs controlled design,
  sub-word tokenisation of transformers, ...)?
- Q4: which types are the most fragile, and a hypothesis for why.
- Read the example files and a few model errors by hand: what goes wrong? -->

## Limitations

- Synthetic noise is not real OCR noise: real errors are correlated (a degraded
  line, a font, a damaged page) whereas here they are independent.
- No segmentation errors, by design, although they strongly affect NER [1].
- The comparison with previously reported figures is indirect: different model,
  different noise and a different pipeline.
- Modern newswire text, not historical documents: language drift, spelling
  variation and period-specific entities are not covered.

## Possible extensions

- Evaluate the transformer on the noisy OCRed version of CoNLL-2003 released
  with [1] (https://zenodo.org/record/3877554), for a direct comparison.
- Add segmentation errors, which requires re-aligning labels.
- Compare with a model adapted to noisy text.

## References

[1] A. Hamdi, E. Linhares Pontes, N. Sidere, M. Coustaty, A. Doucet. 2023.
*In-depth analysis of the impact of OCR errors on named entity recognition and
linking.* Natural Language Engineering 29(2), 425-448.

[2] X. Ma, E. Hovy. 2016. *End-to-end Sequence Labeling via Bi-directional
LSTM-CNNs-CRF.* Proceedings of ACL 2016.

[3] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. 2019. *BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding.* Proceedings of
NAACL-HLT 2019.

[4] E. F. Tjong Kim Sang, F. De Meulder. 2003. *Introduction to the CoNLL-2003
Shared Task: Language-Independent Named Entity Recognition.* Proceedings of
CoNLL 2003.
