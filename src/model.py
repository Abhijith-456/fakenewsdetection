import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file


class DistilBertBiLSTMAttention(nn.Module):
    """
    Hybrid fake news detection model used in the MTech research project.

    Architecture:
        DistilBERT
            ↓
        BiLSTM
            ↓
        Attention
            ↓
        Dropout
            ↓
        Binary Classifier
    """

    def __init__(self, lstm_hidden=128, dropout=0.3):
        super().__init__()

        # Pretrained DistilBERT encoder
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Freeze embeddings
        for param in self.distilbert.embeddings.parameters():
            param.requires_grad = False

        # Freeze DistilBERT layers 0–3
        # Train layers 4–5
        for i, layer in enumerate(self.distilbert.transformer.layer):
            for param in layer.parameters():
                param.requires_grad = i >= 4

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        # DistilBERT contextual representations
        sequence_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)

        # Attention scores
        attention_scores = self.attention_layer(lstm_output)

        # Ignore padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0,
            -1e9
        )

        # Attention weights + weighted pooling
        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled_output = (
            lstm_output * attention_weights
        ).sum(dim=1)

        # Classification
        logits = self.classifier(
            self.dropout(pooled_output)
        )

        # Training loss
        loss = None

        if labels is not None:
            loss = nn.CrossEntropyLoss(
                label_smoothing=0.1
            )(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


def load_model(model_path, device=None):
    """
    Load the trained hybrid model from a safetensors checkpoint.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = DistilBertBiLSTMAttention()

    checkpoint_path = f"{model_path}/model.safetensors"

    state_dict = load_file(
        checkpoint_path,
        device=str(device)
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model