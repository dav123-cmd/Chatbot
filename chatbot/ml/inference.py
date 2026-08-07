"""
Loads the trained BiLSTM checkpoint once (singleton pattern) and exposes
a simple predict(text) -> (response, tag, confidence) function used by
the Django view.
"""
import os
import random
import threading

import torch
import torch.nn.functional as F

from .model import BiLSTMChatbotModel
from .utils import encode_sentence, load_intents


class ChatbotEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path: str, intents_path: str, confidence_threshold: float = 0.55):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python chatbot/ml/train.py` first."
            )

        checkpoint = torch.load(model_path, map_location="cpu")

        self.vocab = checkpoint["vocab"]
        self.tags = checkpoint["tags"]
        self.max_len = checkpoint["max_len"]
        self.confidence_threshold = confidence_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMChatbotModel(
            vocab_size=len(self.vocab),
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=len(self.tags),
            num_layers=checkpoint.get("num_layers", 1),
            dropout=checkpoint.get("dropout", 0.3),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        intents_data = load_intents(intents_path)
        self.responses_by_tag = {
            intent["tag"]: intent["responses"] for intent in intents_data["intents"]
        }

    @classmethod
    def get_instance(cls, model_path: str, intents_path: str, confidence_threshold: float = 0.55):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path, intents_path, confidence_threshold)
        return cls._instance

    def predict(self, text: str):
        ids = encode_sentence(text, self.vocab, self.max_len)
        tensor = torch.tensor([ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            confidence, predicted_idx = torch.max(probs, dim=0)

        confidence = confidence.item()
        tag = self.tags[predicted_idx.item()]

        if confidence < self.confidence_threshold:
            tag = "fallback"

        responses = self.responses_by_tag.get(tag) or self.responses_by_tag.get("fallback")
        response = random.choice(responses)
        return response, tag, confidence
