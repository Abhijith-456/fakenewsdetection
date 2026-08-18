import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import FakeNewsPredictor


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Fake News Detection API",
    description=(
        "API for detecting fake news using a "
        "DistilBERT + BiLSTM + Attention model."
    ),
    version="1.0.0",
)


# Load the model once when the API starts
predictor = FakeNewsPredictor()


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="News article text to classify.",
        examples=[
            "Scientists have confirmed that regular exercise "
            "can improve cardiovascular health."
        ],
    )


class ProbabilityResponse(BaseModel):
    fake: float
    real: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: ProbabilityResponse


@app.get("/")
def root():
    return {
        "message": "Fake News Detection API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    try:
        result = predictor.predict(request.text)

        return result

    except Exception:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        )