import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": (
                "Scientists have confirmed that regular exercise "
                "can improve cardiovascular health and reduce "
                "the risk of several chronic diseases."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data

    assert data["prediction"] in ["FAKE", "REAL"]

    assert 0 <= data["confidence"] <= 1

    assert "fake" in data["probabilities"]
    assert "real" in data["probabilities"]

    assert 0 <= data["probabilities"]["fake"] <= 1
    assert 0 <= data["probabilities"]["real"] <= 1

def test_predict_rejects_short_text():
    response = client.post(
        "/predict",
        json={
            "text": "hello"
        },
    )

    assert response.status_code == 422

def test_predict_rejects_long_text():
    response = client.post(
        "/predict",
        json={
            "text": "a" * 10001
        },
    )

    assert response.status_code == 422


def test_predict_rejects_missing_text():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422
def test_predict_handles_inference_error(monkeypatch):
    def failing_predict(text):
        raise RuntimeError("internal model failure")

    monkeypatch.setattr(
        "api.main.predictor.predict",
        failing_predict,
    )

    response = client.post(
        "/predict",
        json={
            "text": (
                "This is a valid article text that should "
                "normally be processed by the prediction model."
            )
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == "Prediction failed. Please try again."