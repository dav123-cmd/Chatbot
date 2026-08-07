"""
Train the Bidirectional LSTM intent classifier on chatbot/ml/intents.json
and save the trained weights + vocabulary + tags to chatbot_model.pth.

Run from the project root with:
    python chatbot/ml/train.py
"""
import os
import sys
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(__file__))
from utils import build_vocab, encode_sentence, load_intents  # noqa: E402
from model import BiLSTMChatbotModel  # noqa: E402

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(__file__)
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "chatbot_model.pth")

EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.3
MAX_LEN = 12
BATCH_SIZE = 8
EPOCHS = 300
LEARNING_RATE = 0.001


class IntentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def main():
    data = load_intents(INTENTS_PATH)

    all_patterns = []
    all_labels = []
    tags = []

    for intent in data["intents"]:
        tag = intent["tag"]
        if tag not in tags:
            tags.append(tag)
        for pattern in intent["patterns"]:
            all_patterns.append(pattern)
            all_labels.append(tags.index(tag))

    if len(all_patterns) == 0:
        raise RuntimeError("No training patterns found in intents.json")

    vocab = build_vocab(all_patterns)
    sequences = [encode_sentence(p, vocab, MAX_LEN) for p in all_patterns]

    dataset = IntentDataset(sequences, all_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMChatbotModel(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=len(tags),
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        if epoch % 25 == 0 or epoch == 1:
            avg_loss = total_loss / total
            acc = correct / total
            print(f"Epoch {epoch:4d}/{EPOCHS} | loss={avg_loss:.4f} | acc={acc:.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "tags": tags,
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }, MODEL_SAVE_PATH)

    print(f"\nTraining complete. Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
