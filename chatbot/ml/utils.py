"""
Text preprocessing utilities shared by training and inference.
No external NLP dependency is required: a lightweight regex tokenizer
is used so the project only needs torch + Django to run.
"""
import re
import json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(sentence: str):
    """Lowercase and split a sentence into word tokens."""
    sentence = sentence.lower()
    tokens = re.findall(r"[a-zA-Z']+", sentence)
    return tokens


def build_vocab(all_patterns):
    """Build a word -> index vocabulary from a list of sentences."""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for sentence in all_patterns:
        for token in tokenize(sentence):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_sentence(sentence: str, vocab: dict, max_len: int):
    """Convert a sentence into a fixed-length list of word indices."""
    tokens = tokenize(sentence)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens][:max_len]
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def load_intents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
