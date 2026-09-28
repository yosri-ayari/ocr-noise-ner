"""Loading a corpus and a pre-trained NER model from the Hugging Face Hub."""

from labels import align_predictions, to_iob2

# Two ready-made configurations. Any token-classification model and any
# dataset with "tokens" and "ner_tags" columns can be passed on the command line.
PRESETS = {
    "en": {
        "dataset": "eriktks/conll2003",
        "config": None,
        "split": "test",
        "model": "dslim/bert-base-NER",
        "keep_types": ["PER", "LOC", "ORG", "MISC"],
    },
    "fr": {
        "dataset": "unimelb-nlp/wikiann",
        "config": "fr",
        "split": "test",
        "model": "Jean-Baptiste/camembert-ner",
        "keep_types": ["PER", "LOC", "ORG"],
    },
}


def load_corpus(dataset, config, split, keep_types, limit=None):
    """Return (sentences, gold_tags) with gold tags normalised to IOB2."""
    from datasets import load_dataset

    try:
        data = load_dataset(dataset, config, split=split)
    except Exception as first_error:  # datasets that still rely on a loading script
        print(f"Direct loading failed ({first_error}); trying the parquet export.")
        data = load_dataset(dataset, config, split=split, revision="refs/convert/parquet")

    names = data.features["ner_tags"].feature.names
    if limit:
        data = data.select(range(min(limit, len(data))))
    sentences, gold = [], []
    for row in data:
        if not row["tokens"]:
            continue
        sentences.append(list(row["tokens"]))
        gold.append(to_iob2([names[i] for i in row["ner_tags"]], set(keep_types)))
    return sentences, gold


class Tagger:
    """Wraps a Hugging Face token-classification model for pre-tokenised input."""

    def __init__(self, model_name, batch_size=32, max_length=512, device=None):
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        config = AutoConfig.from_pretrained(model_name)
        extra = {"add_prefix_space": True} if config.model_type in {"roberta", "longformer", "bart"} else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **extra)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required to map sub-tokens back to words.")
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size = batch_size
        self.max_length = max_length

    def predict(self, sentences):
        """Return one raw label per word, in the model's own tagging scheme."""
        predictions = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start:start + self.batch_size]
            encoding = self.tokenizer(batch, is_split_into_words=True, truncation=True,
                                      max_length=self.max_length, padding=True,
                                      return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            with self.torch.no_grad():
                label_ids = self.model(**inputs).logits.argmax(-1).cpu().tolist()
            for b, words in enumerate(batch):
                predictions.append(align_predictions(encoding.word_ids(b), label_ids[b],
                                                     len(words), self.id2label))
        return predictions
