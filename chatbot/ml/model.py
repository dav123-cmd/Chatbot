"""
Bidirectional LSTM intent-classification model.

Architecture:
    Embedding -> Bidirectional LSTM -> concat(final forward, final backward
    hidden states) -> Linear -> ReLU -> Dropout -> Linear -> logits
"""
import torch
import torch.nn as nn


class BiLSTMChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)                 # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)          # hidden: (2*num_layers, batch, hidden_dim)

        # Concatenate the last forward and backward hidden states
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)

        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits
