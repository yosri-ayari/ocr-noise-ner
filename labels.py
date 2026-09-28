"""Label handling: tag normalisation, sub-token alignment and entity-level scoring.

Everything here is plain Python, with no dependency on a model, so it can be
unit tested without downloading anything.
"""

from collections import Counter


def to_iob2(tags, keep_types=None):
    """Normalise tags written in IOB1, IOB2, IO, BIOES or BILOU into IOB2.

    Entity types absent from `keep_types` become "O". This lets a model that
    predicts MISC be scored against a corpus that has no MISC annotations.
    """
    out = []
    previous_type = None
    for tag in tags:
        if tag is None or tag == "O":
            out.append("O")
            previous_type = None
            continue
        prefix, _, entity_type = tag.partition("-")
        if not entity_type:  # a bare type such as "PER"
            prefix, entity_type = "I", tag
        if keep_types is not None and entity_type not in keep_types:
            out.append("O")
            previous_type = None
            continue
        if prefix in ("B", "S", "U") or entity_type != previous_type:
            out.append("B-" + entity_type)
        else:
            out.append("I-" + entity_type)
        previous_type = entity_type
    return out


def align_predictions(word_ids, token_label_ids, n_words, id2label):
    """Map sub-token predictions back to words.

    Each word takes the prediction of its first sub-token. A word that received
    no sub-token (for instance because the sentence was truncated) is labelled
    "O". `word_ids` comes from a fast tokenizer (`encoding.word_ids(i)`).
    """
    labels = ["O"] * n_words
    seen = set()
    for word_id, label_id in zip(word_ids, token_label_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        labels[word_id] = id2label[label_id]
    return labels


def entity_spans(tags):
    """Extract (type, start, end) spans from a valid IOB2 sequence."""
    spans = []
    start = entity_type = None
    for i, tag in enumerate(list(tags) + ["O"]):
        if start is not None and not (tag.startswith("I-") and tag[2:] == entity_type):
            spans.append((entity_type, start, i))
            start = entity_type = None
        if tag.startswith("B-"):
            start, entity_type = i, tag[2:]
    return spans


def score(gold_sentences, pred_sentences):
    """Entity-level precision, recall and F1 with exact span and type match.

    This is the CoNLL evaluation: an entity counts as correct only if both its
    boundaries and its type are right. Returns a dict with a "micro" entry and
    one entry per entity type.
    """
    tp, n_gold, n_pred = Counter(), Counter(), Counter()
    for gold, pred in zip(gold_sentences, pred_sentences):
        gold_spans = set(entity_spans(gold))
        pred_spans = set(entity_spans(pred))
        for span in gold_spans:
            n_gold[span[0]] += 1
        for span in pred_spans:
            n_pred[span[0]] += 1
        for span in gold_spans & pred_spans:
            tp[span[0]] += 1

    def prf(t, g, p):
        precision = t / p if p else 0.0
        recall = t / g if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "support": g}

    results = {"micro": prf(sum(tp.values()), sum(n_gold.values()), sum(n_pred.values()))}
    for entity_type in sorted(n_gold):
        results[entity_type] = prf(tp[entity_type], n_gold[entity_type], n_pred[entity_type])
    return results
