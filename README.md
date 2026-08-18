# Fake News Detection

End-to-end fake news detection system built using a hybrid
**DistilBERT + BiLSTM + Attention** architecture.

The project combines a trained NLP classification model with a
production-style inference pipeline consisting of FastAPI, Streamlit,
Docker, automated testing, and GitHub Actions CI.

---

## Project Status

**Completed — Production-style local deployment**

The complete application has been tested locally using Docker Compose.

Current validation includes:

- 7 automated API tests passing
- FastAPI inference verified
- Streamlit frontend verified
- Docker API verified
- Docker Compose verified
- API health check verified
- Container-to-container communication verified
- GitHub Actions CI passing
- Model loading and CPU inference verified

---

## Key Features

- Hybrid **DistilBERT + BiLSTM + Attention** architecture
- Selective fine-tuning of DistilBERT layers
- Binary fake/real news classification
- Token-level attention mechanism
- FastAPI inference API
- Streamlit interactive frontend
- Input validation
- Prediction confidence and class probabilities
- API health monitoring
- Dockerized backend and frontend
- Docker Compose orchestration
- Automated API testing with PyTest
- GitHub Actions continuous integration
- Git LFS model management

---

## Architecture

```text
                         User
                           |
                           v
                  +----------------+
                  |    Streamlit   |
                  |      :8501     |
                  +-------+--------+
                          |
                          | HTTP
                          v
                  +----------------+
                  |     FastAPI    |
                  |      :8000     |
                  +-------+--------+
                          |
                          v
                +----------------------+
                |   FakeNewsPredictor  |
                +----------+-----------+
                           |
                           v
                    +-------------+
                    |  DistilBERT |
                    +------+------+
                           |
                           v
                     +---------+
                     |  BiLSTM |
                     +----+----+
                          |
                          v
                    +-----------+
                    | Attention |
                    +-----+-----+
                          |
                          v
                   +-------------+
                   |   Dropout   |
                   +------+------+ 
                          |
                          v
                   +-------------+
                   | Classifier  |
                   +------+------+
                          |
                          v
                      FAKE / REAL
```

---

## Model Architecture

The model uses the following pipeline:

**DistilBERT → BiLSTM → Attention → Dropout → Binary Classifier**

### DistilBERT

A pretrained `distilbert-base-uncased` model is used as the contextual text encoder.

The model processes tokenized news articles and produces contextualized token representations.

### Selective Fine-Tuning

To reduce the number of trainable parameters:

- DistilBERT embeddings are frozen
- Transformer layers 0–3 are frozen
- Transformer layers 4–5 are trainable
- BiLSTM is trainable
- Attention mechanism is trainable
- Classification head is trainable

### BiLSTM

The DistilBERT sequence representations are passed through a bidirectional LSTM.

Configuration:

- Input size: 768
- Hidden size: 128
- Bidirectional: Yes
- Output representation: 256 dimensions

### Attention

An attention mechanism assigns different weights to the BiLSTM sequence representations.

Architecture:

```text
256 → 64 → 1
```

Padding tokens are masked before calculating the attention weights.

The weighted sequence representation is then passed to the classification head.

### Classification

The final classifier produces two logits:

```text
0 → FAKE
1 → REAL
```

A softmax function converts the logits into class probabilities.

---

## Model Configuration

| Component | Configuration |
|---|---|
| Base model | `distilbert-base-uncased` |
| Maximum sequence length | 256 tokens |
| DistilBERT embeddings | Frozen |
| DistilBERT layers 0–3 | Frozen |
| DistilBERT layers 4–5 | Trainable |
| BiLSTM hidden size | 128 |
| BiLSTM | Bidirectional |
| Attention hidden size | 64 |
| Dropout | 0.3 |
| Output classes | 2 |
| Loss | Cross Entropy |
| Label smoothing | 0.1 |

---

## Dataset

The primary research dataset used for training and evaluation is:

**WELFake**

The research experiment used a stratified:

```text
80% Training
10% Validation
10% Test
```

split.

The project also includes cross-dataset evaluation using the **ISOT** dataset.

The datasets used during research are not included in the production repository's `data/` directory.

---

## Model Performance

The following results are the **reported results from the research evaluation notebooks** associated with the trained model.

### WELFake Evaluation

| Metric | Result |
|---|---:|
| Accuracy | **99.38%** |
| F1 Score | **0.9938** |
| Trainable Parameters | **3.15M** |
| Total Parameters | **~66.4M** |

The reported experiment used:

- WELFake dataset
- Maximum sequence length of 256
- 4 training epochs
- Selective DistilBERT fine-tuning
- DistilBERT + BiLSTM + Attention architecture

### Cross-Dataset Evaluation

The trained model was additionally evaluated on the **ISOT** dataset without retraining or domain adaptation.

| Metric | Result |
|---|---:|
| Dataset | ISOT |
| Test Articles | **8,980** |
| F1 Score | **0.9998** |

### Baseline Comparison

A vanilla DistilBERT baseline achieved a reported F1 score of **0.9946** on the WELFake evaluation.

The hybrid model achieved **0.9938 F1**.

A reported McNemar's test produced:

```text
p-value = 0.3768
```

indicating that the observed difference between the two systems was not statistically significant under that reported comparison.

> These metrics represent the research evaluation associated with the trained model. The original research evaluation outputs are not stored in the production repository.

---

## Inference Pipeline

The production inference pipeline is:

```text
News Article
     |
     v
Input Validation
     |
     v
DistilBERT Tokenizer
     |
     v
DistilBERT
     |
     v
BiLSTM
     |
     v
Attention
     |
     v
Classifier
     |
     v
Softmax Probabilities
     |
     v
FAKE / REAL
```

The model is loaded once when the FastAPI application starts and reused for subsequent prediction requests.

---

## API

The project exposes a FastAPI backend.

### Root Endpoint

```http
GET /
```

Example response:

```json
{
  "message": "Fake News Detection API",
  "status": "running"
}
```

### Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Prediction Endpoint

```http
POST /predict
```

Request:

```json
{
  "text": "Scientists have confirmed that regular exercise can improve cardiovascular health."
}
```

Example response:

```json
{
  "prediction": "FAKE",
  "confidence": 0.6445227265357971,
  "probabilities": {
    "fake": 0.6445227265357971,
    "real": 0.3554772734642029
  }
}
```

---

## Input Validation

The API validates incoming requests using Pydantic.

### Limits

- Minimum article length: **10 characters**
- Maximum article length: **10,000 characters**

Invalid input is rejected with HTTP `422`.

The Streamlit frontend performs the same validation before sending requests to the API.

---

## Error Handling

The Streamlit application handles:

- Empty input
- Input below the minimum length
- Input above the maximum length
- API connection errors
- API timeouts
- HTTP status errors
- Unexpected API responses

The FastAPI application also validates the incoming request schema before inference.

---

## Streamlit Interface

The project includes an interactive Streamlit frontend.

The interface allows users to:

1. Enter or paste a news article
2. Validate the article length
3. Submit the article for prediction
4. View the predicted class
5. View prediction confidence
6. View fake probability
7. View real probability

The frontend communicates with the FastAPI backend using HTTP.

---

## Docker

The application is containerized using Docker.

Two services are defined:

```text
Streamlit
   |
   | HTTP
   v
FastAPI
   |
   v
ML Model
```

### Services

| Service | Port | Purpose |
|---|---:|---|
| API | 8000 | FastAPI inference service |
| Streamlit | 8501 | Web frontend |

---

## Docker Compose

The application can be started using:

```powershell
docker compose up -d
```

Check the running services:

```powershell
docker compose ps
```

The API includes a Docker health check using:

```http
GET /health
```

The Streamlit service depends on the API becoming healthy before starting.

### Build

Build both services:

```powershell
docker compose build
```

Build only the API:

```powershell
docker compose build api
```

Build only Streamlit:

```powershell
docker compose build streamlit
```

### Stop

```powershell
docker compose down
```

### View API Logs

```powershell
docker compose logs --tail=30 api
```

### View Streamlit Logs

```powershell
docker compose logs --tail=30 streamlit
```

---

## Docker Architecture

```text
                    Docker Compose
                         |
             +-----------+-----------+
             |                       |
             v                       v
   +-------------------+   +-------------------+
   | fakenews-streamlit|   |   fakenews-api    |
   |       :8501       |-->|       :8000       |
   +-------------------+   +---------+---------+
                                     |
                                     v
                              Hybrid ML Model
```

The Streamlit container communicates with the API using the Docker Compose service name:

```text
http://api:8000
```

The API is exposed to the host at:

```text
http://localhost:8000
```

The Streamlit application is exposed at:

```text
http://localhost:8501
```

---

## Local Development

### 1. Clone the repository

```powershell
git clone <your-github-repository-url>
cd <repository-directory>
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
API_URL=http://127.0.0.1:8000
```

A `.env.example` file is included as a template.

The local `.env` file should not be committed to Git.

### 5. Start FastAPI

```powershell
uvicorn api.main:app --reload --port 8000
```

### 6. Start Streamlit

In another terminal:

```powershell
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## API Example

Using PowerShell:

```powershell
$body = @{
    text = "Scientists have confirmed that regular exercise can improve cardiovascular health and reduce the risk of several chronic diseases."
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

---

## Testing

The project uses **PyTest** for automated API testing.

Run:

```powershell
python -m pytest
```

Current test suite:

```text
7 passed
```

The tests cover:

- Root endpoint
- Health endpoint
- Successful prediction
- Short input rejection
- Long input rejection
- Missing/invalid input handling
- Prediction response validation

The tests verify:

- HTTP status codes
- API response structure
- Prediction labels
- Confidence range
- Fake probability range
- Real probability range
- Request validation

---

## Continuous Integration

GitHub Actions automatically runs the test suite for changes pushed to `main` and pull requests targeting `main`.

The workflow is located at:

```text
.github/workflows/tests.yml
```

The CI pipeline performs:

1. Repository checkout
2. Python environment setup
3. Dependency installation
4. Automated test execution

The GitHub Actions workflow has been successfully verified.

---

## Project Structure

```text
FND PROJECT/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│
├── data/
│
├── models/
│   └── hybrid_model/
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── vocab.txt
│
├── notebooks/
│   ├── 01_model_training.ipynb
│   └── 02_cross_dataset_evaluation.ipynb
│
├── scripts/
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── predictor.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── .gitattributes
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── README.md
├── requirements.txt
├── requirements-docker.txt
└── requirements-streamlit.txt
```

---

## Technologies

### Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- BiLSTM
- Attention Mechanism
- Scikit-learn

### API

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- Streamlit
- HTTPX

### Testing

- PyTest
- FastAPI TestClient

### Deployment and Infrastructure

- Docker
- Docker Compose
- GitHub Actions
- Git LFS

### Development

- Git
- GitHub
- Jupyter Notebook

---

## Research Components

The research implementation includes additional experimental components such as:

- Selective transformer fine-tuning
- Attention-based representation pooling
- Cross-dataset evaluation
- Statistical comparison against a vanilla DistilBERT baseline
- SHAP-based token-level explainability
- Attention visualization
- Threshold sensitivity analysis
- Calibration analysis
- Error analysis
- Bootstrap evaluation

These research components are primarily documented in the project notebooks.

---

## Limitations

The system is intended as a research and portfolio project for fake-news classification.

Important limitations include:

- Model predictions should not be treated as definitive fact verification.
- Model confidence does not guarantee factual correctness.
- Dataset characteristics can influence model behavior.
- Domain shifts may affect real-world performance.
- The production repository does not contain the complete research datasets.
- Reported research metrics were generated during the research evaluation phase.

---

## Future Improvements

Potential future improvements include:

- Public cloud deployment
- Model monitoring
- Request logging
- Model versioning
- Automated model evaluation in CI
- Additional domain-specific datasets
- Improved calibration
- Adversarial robustness evaluation
- Explainability exposed directly through the API/UI
- Automated retraining pipelines

---

## Research Context

This project was developed as part of an MTech research project focused on transformer-based fake news detection.

The implementation combines transformer-based contextual representations with recurrent sequence modeling and attention to create a hybrid NLP classification architecture.

The production layer extends the research model into an end-to-end machine learning application with API serving, frontend integration, containerization, automated testing, and continuous integration.

---

## License

This project is intended for academic, research, and portfolio purposes.