"""Tests that need no model and no download.

    python -m unittest discover tests
"""

import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labels import align_predictions, entity_spans, score, to_iob2  # noqa: E402
from noise import corrupt_corpus, corrupt_word, levenshtein  # noqa: E402

SENTENCES = [
    ["Victor", "Hugo", "est", "né", "à", "Besançon", "en", "1802", "."],
    ["La", "Société", "Générale", "ouvre", "une", "agence", "à", "Lyon", "."],
    ["Marie", "Curie", "travaille", "à", "Paris", "avec", "Pierre", "Curie", "."],
] * 40
GOLD = [
    ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "O", "O", "O"],
    ["O", "B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC", "O"],
    ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "B-PER", "I-PER", "O"],
] * 40


class NoiseTests(unittest.TestCase):
    def test_rate_zero_is_identity(self):
        noisy, cer, changed = corrupt_corpus(SENTENCES, 0.0, seed=1)
        self.assertEqual(noisy, SENTENCES)
        self.assertEqual((cer, changed), (0.0, 0.0))

    def test_word_boundaries_are_preserved(self):
        noisy, _, _ = corrupt_corpus(SENTENCES, 0.3, seed=1)
        for clean, dirty in zip(SENTENCES, noisy):
            self.assertEqual(len(clean), len(dirty))
            for word in dirty:
                self.assertTrue(word)
                self.assertNotIn(" ", word)

    def test_punctuation_is_untouched(self):
        noisy, _, _ = corrupt_corpus(SENTENCES, 0.5, seed=3)
        self.assertTrue(all(s[-1] == "." for s in noisy))

    def test_same_seed_same_noise(self):
        self.assertEqual(corrupt_corpus(SENTENCES, 0.1, seed=7)[0],
                         corrupt_corpus(SENTENCES, 0.1, seed=7)[0])

    def test_cer_grows_with_rate(self):
        cers = [corrupt_corpus(SENTENCES, r, seed=0)[1] for r in (0.02, 0.1, 0.3)]
        self.assertLess(cers[0], cers[1])
        self.assertLess(cers[1], cers[2])

    def test_never_empty(self):
        rng = random.Random(0)
        for _ in range(2000):
            self.assertTrue(corrupt_word("a", 1.0, rng))

    def test_deletion_only_shortens(self):
        noisy, _, _ = corrupt_corpus(SENTENCES, 0.3, seed=2, edit_type="deletion")
        for clean, dirty in zip(SENTENCES, noisy):
            for a, b in zip(clean, dirty):
                self.assertLessEqual(len(b), len(a))

    def test_insertion_only_lengthens(self):
        noisy, _, _ = corrupt_corpus(SENTENCES, 0.3, seed=2, edit_type="insertion")
        for clean, dirty in zip(SENTENCES, noisy):
            for a, b in zip(clean, dirty):
                self.assertGreaterEqual(len(b), len(a))

    def test_case_only_changes_case(self):
        noisy, _, _ = corrupt_corpus(SENTENCES, 0.5, seed=2, edit_type="case")
        for clean, dirty in zip(SENTENCES, noisy):
            for a, b in zip(clean, dirty):
                self.assertEqual(a.lower(), b.lower())

    def test_first_position_touches_only_first_character(self):
        rng = random.Random(0)
        for _ in range(500):
            self.assertEqual(corrupt_word("Besançon", 1.0, rng, "deletion", "first"), "esançon")
            self.assertEqual(corrupt_word("Paris", 1.0, rng, "case", "first"), "paris")

    def test_inner_position_never_touches_first_character(self):
        rng = random.Random(0)
        for _ in range(500):
            noisy = corrupt_word("Besançon", 1.0, rng, "substitution", "inner")
            self.assertEqual(noisy[0], "B")
            self.assertEqual(levenshtein("Besançon", noisy) <= 2, True)

    def test_first_and_inner_corrupt_the_same_share_of_words(self):
        _, _, first = corrupt_corpus(SENTENCES, 0.5, seed=4, edit_type="deletion", position="first")
        _, _, inner = corrupt_corpus(SENTENCES, 0.5, seed=4, edit_type="deletion", position="inner")
        self.assertAlmostEqual(first, inner, places=6)

    def test_levenshtein(self):
        self.assertEqual(levenshtein("modern", "modem"), 2)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("Paris", "Paris"), 0)
        self.assertEqual(levenshtein("chat", "chut"), 1)


class LabelTests(unittest.TestCase):
    def test_io_scheme_becomes_iob2(self):
        self.assertEqual(to_iob2(["I-PER", "I-PER", "O", "I-LOC"]),
                         ["B-PER", "I-PER", "O", "B-LOC"])

    def test_iob1_new_entity_after_type_change(self):
        self.assertEqual(to_iob2(["I-PER", "I-LOC"]), ["B-PER", "B-LOC"])

    def test_bioes(self):
        self.assertEqual(to_iob2(["S-PER", "S-PER", "B-ORG", "E-ORG"]),
                         ["B-PER", "B-PER", "B-ORG", "I-ORG"])

    def test_filtered_types(self):
        self.assertEqual(to_iob2(["B-MISC", "I-MISC", "B-PER"], {"PER"}),
                         ["O", "O", "B-PER"])

    def test_first_subtoken_wins(self):
        id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        # [CLS] Vic ##tor Hugo [SEP] -> words 0, 0, 1
        word_ids = [None, 0, 0, 1, None]
        labels = [0, 1, 0, 2, 0]
        self.assertEqual(align_predictions(word_ids, labels, 2, id2label), ["B-PER", "I-PER"])

    def test_truncated_words_are_o(self):
        self.assertEqual(align_predictions([None, 0, None], [0, 1, 0], 3, {0: "O", 1: "B-LOC"}),
                         ["B-LOC", "O", "O"])

    def test_spans(self):
        self.assertEqual(entity_spans(["B-PER", "I-PER", "O", "B-LOC", "B-LOC"]),
                         [("PER", 0, 2), ("LOC", 3, 4), ("LOC", 4, 5)])

    def test_score_exact_match_only(self):
        gold = [["B-PER", "I-PER", "O", "B-LOC"]]
        pred = [["B-PER", "O", "O", "B-LOC"]]  # PER boundary wrong, LOC right
        result = score(gold, pred)
        self.assertAlmostEqual(result["micro"]["precision"], 0.5)
        self.assertAlmostEqual(result["micro"]["recall"], 0.5)
        self.assertEqual(result["PER"]["f1"], 0.0)
        self.assertEqual(result["LOC"]["f1"], 1.0)

    def test_perfect_score(self):
        self.assertEqual(score(GOLD, GOLD)["micro"]["f1"], 1.0)


class LexiconTagger:
    """Stand-in for a real model: knows the clean words, fails on corrupted ones."""

    def __init__(self, sentences, gold):
        self.lexicon = {w: t.replace("B-", "I-") for s, g in zip(sentences, gold) for w, t in zip(s, g)}

    def predict(self, sentences):
        return [[self.lexicon.get(w, "O") for w in s] for s in sentences]


class PipelineTest(unittest.TestCase):
    def test_end_to_end_without_a_model(self):
        from plot_results import main as plot_main
        from run_experiment import run

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.csv"
            rows = run(SENTENCES, GOLD, LexiconTagger(SENTENCES, GOLD), [0, 0.05, 0.2], 2,
                       ["PER", "LOC", "ORG"], out, examples_path=Path(tmp) / "examples.txt")
            self.assertEqual(len(rows), 1 + 2 + 2)
            self.assertEqual(rows[0]["micro_f1"], 1.0)
            self.assertGreater(rows[0]["micro_f1"], rows[-1]["micro_f1"])
            plot_main(out)
            for suffix in ("_f1.png", "_by_type.png", "_summary.md"):
                self.assertTrue((Path(tmp) / f"results{suffix}").exists())

    def test_edit_type_and_position_are_recorded(self):
        from run_experiment import run

        with tempfile.TemporaryDirectory() as tmp:
            rows = run(SENTENCES, GOLD, LexiconTagger(SENTENCES, GOLD), [0, 0.5], 1,
                       ["PER", "LOC", "ORG"], Path(tmp) / "r.csv",
                       edit_type="case", position="first")
            self.assertEqual((rows[-1]["edit_type"], rows[-1]["position"]), ("case", "first"))
            self.assertLess(rows[-1]["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
