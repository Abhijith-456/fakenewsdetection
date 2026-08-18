# Fake News Detection

End-to-end fake news detection system built using a hybrid
**DistilBERT + BiLSTM + Attention** architecture.

The project provides:

- A trained NLP classification model
- Production-style inference wrapper
- FastAPI prediction API
- Streamlit web interface
- Dockerized API and frontend
- Docker Compose orchestration
- Automated API tests
- GitHub Actions CI
- Git LFS model management

---

## Project Status

**Production-ready local deployment**

The complete application has been tested locally using Docker Compose.

Current validation includes:

- 7 automated API tests passing
- FastAPI inference verified
- Streamlit UI verified
- Docker API verified
- Docker Compose verified
- API health check verified
- Container-to-container communication verified
- GitHub Actions CI passing

---

## Architecture

```text
                    User
                     |
                     v
              +-------------+
              |  Streamlit  |
              |    :8501    |
              +------+------+
                     |
                     | HTTP
                     v
              +-------------+
              |   FastAPI   |
              |    :8000    |
              +------+------+
                     |
                     v
          +-----------------------+
          |   FakeNewsPredictor   |
          +-----------+-----------+
                      |
                      v
               +-------------+
               |  DistilBERT |
               +------+------+
                      |
                      v
                +---------+
                | BiLSTM  |
                +----+----+
                     |
                     v
                +---------+
                |Attention|
                +----+----+
                     |
                     v
                +---------+
                |Classifier|
                +---------+
                     |
                     v
              FAKE / REAL
