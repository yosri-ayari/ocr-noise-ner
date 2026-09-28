"""Synthetic OCR noise injection at the character level.

Design choice: noise is applied inside each word and never adds or removes a
word boundary. Word-level NER labels therefore stay aligned one-to-one with the
noisy words, and no label re-alignment is needed. The cost of this choice is
that segmentation errors (merged or split words), which are frequent in real
OCR output and strongly hurt NER according to Hamdi et al. (2023), are not
simulated. This is stated as a limitation in the README.

Two settings control the noise:

* `edit_type` chooses which edits are made (see EDIT_MIXES). "mixed" combines
  all of them; the others isolate one operation, to test which kind of error
  hurts the model most.
* `position` chooses where edits happen.
    - "any": every character can start an edit, with probability `rate`.
    - "first": a word gets exactly one edit, on its first character, with
      probability `rate`.
    - "inner": a word gets exactly one edit, on a random character other than
      the first, with probability `rate`.
  "first" and "inner" corrupt the same number of words with the same number of
  edits, so comparing them isolates the effect of the error position. Both only
  apply to words of two characters or more, so that they cover the same words.

With position "any", `rate` is close to, but not exactly, the character error
rate (CER), because some confusions replace two characters by one (rn -> m).
The CER that is reported is always measured afterwards with the Levenshtein
distance.
"""

import random
import string

# Visually confusable sequences, inspired by common OCR and HTR errors.
# Two-character keys are tried before single characters.
CONFUSIONS = {
    "rn": ["m"], "cl": ["d"], "li": ["h"], "vv": ["w"], "ri": ["n"],
    "m": ["rn", "nn"], "d": ["cl"], "h": ["li", "b"], "w": ["vv"],
    "l": ["1", "I", "i"], "I": ["l", "1"], "i": ["l", "1"], "1": ["l", "I"],
    "O": ["0", "Q"], "o": ["0", "c", "e"], "0": ["O", "o"],
    "e": ["c", "o"], "c": ["e", "o"], "a": ["o", "e"],
    "S": ["5", "s"], "s": ["5", "ſ"], "5": ["S"],  # ſ is the long s
    "B": ["8"], "8": ["B"], "u": ["n", "ii"], "n": ["u", "ri"],
    "f": ["t", "ſ"], "t": ["f"], "g": ["q", "9"], "q": ["g"],
    "E": ["F"], "F": ["E"], "C": ["G"], "G": ["C"],
}

# Characters used for random substitutions and insertions.
ALPHABET = string.ascii_letters + string.digits + "éèàç"

# Share of each operation within an edit type.
EDIT_MIXES = {
    "mixed": {"confusion": 0.60, "random_sub": 0.15, "deletion": 0.15, "insertion": 0.10},
    "substitution": {"confusion": 0.80, "random_sub": 0.20},
    "deletion": {"deletion": 1.0},
    "insertion": {"insertion": 1.0},
    "case": {"case": 1.0},  # upper case <-> lower case, as when OCR misreads a capital
}
POSITIONS = ("any", "first", "inner")


def is_wordlike(token):
    """Only tokens with at least one letter or digit are corrupted."""
    return any(ch.isalnum() for ch in token)


def _pick(mix, rng):
    r = rng.random()
    total = 0.0
    for operation, share in mix.items():
        total += share
        if r < total:
            return operation
    return operation


def _edit(word, i, operation, rng):
    """Apply one operation at position i. Returns (replacement, characters consumed)."""
    if operation == "confusion":
        pair = word[i:i + 2]
        if len(pair) == 2 and pair in CONFUSIONS:
            return rng.choice(CONFUSIONS[pair]), 2
        if word[i] in CONFUSIONS:
            return rng.choice(CONFUSIONS[word[i]]), 1
        operation = "random_sub"  # no plausible confusion for this character
    if operation == "random_sub":
        return rng.choice(ALPHABET), 1
    if operation == "deletion":
        return "", 1
    if operation == "insertion":
        return word[i] + rng.choice(ALPHABET), 1
    if operation == "case":
        return word[i].swapcase(), 1
    raise ValueError(f"unknown operation {operation!r}")


def corrupt_word(word, rate, rng, edit_type="mixed", position="any"):
    """Return a noisy copy of `word`. Never returns an empty string or a space."""
    mix = EDIT_MIXES[edit_type]
    if position == "any":
        out, i = [], 0
        while i < len(word):
            if rng.random() < rate:
                replacement, consumed = _edit(word, i, _pick(mix, rng), rng)
                out.append(replacement)
                i += consumed
            else:
                out.append(word[i])
                i += 1
        noisy = "".join(out)
    elif position in ("first", "inner"):
        if len(word) < 2:
            return word
        # The same three draws are made in both modes, so that with the same seed
        # "first" and "inner" corrupt exactly the same words: a paired comparison.
        hit = rng.random() < rate
        inner_index = rng.randrange(1, len(word))
        local_rng = random.Random(rng.random())
        if not hit:
            return word
        i = 0 if position == "first" else inner_index
        replacement, consumed = _edit(word, i, _pick(mix, local_rng), local_rng)
        noisy = word[:i] + replacement + word[i + consumed:]
    else:
        raise ValueError(f"position must be one of {POSITIONS}")
    return noisy if noisy else word[:1]


def levenshtein(a, b):
    """Edit distance between two strings (insertions, deletions, substitutions)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def corrupt_corpus(sentences, rate, seed, edit_type="mixed", position="any"):
    """Corrupt a list of tokenised sentences.

    Returns (noisy_sentences, measured_cer, share_of_words_changed).
    The CER is computed over all characters of all tokens, spaces excluded.
    """
    rng = random.Random(seed)
    noisy_sentences = []
    edits = chars = changed = words = 0
    for tokens in sentences:
        noisy = [corrupt_word(t, rate, rng, edit_type, position)
                 if rate > 0 and is_wordlike(t) else t for t in tokens]
        for clean, dirty in zip(tokens, noisy):
            edits += levenshtein(clean, dirty)
            chars += len(clean)
            changed += clean != dirty
            words += 1
        noisy_sentences.append(noisy)
    return noisy_sentences, edits / max(chars, 1), changed / max(words, 1)
