import torch
from transformers import DistilBertTokenizerFast

from src.model import load_model


class FakeNewsPredictor:
    """
    Production inference wrapper for the trained
    DistilBERT + BiLSTM + Attention model.
    """

    def __init__(self, model_path="models/hybrid_model"):
        self.model_path = model_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading model on: {self.device}")

        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.model_path
        )

        # Load trained model
        self.model = load_model(
            self.model_path,
            self.device
        )

        print("Model loaded successfully.")

    def predict(self, text: str) -> dict:
        """
        Predict whether a news article is fake or real.

        Parameters
        ----------
        text : str
            News article text.

        Returns
        -------
        dict
            Prediction and class probabilities.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        # Move tensors to the selected device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Inference
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

        fake_probability = probabilities[0].item()
        real_probability = probabilities[1].item()

        label = "FAKE" if prediction == 0 else "REAL"

        confidence = max(
            fake_probability,
            real_probability
        )

        return {
            "prediction": label,
            "confidence": confidence,
            "probabilities": {
                "fake": fake_probability,
                "real": real_probability
            }
        }