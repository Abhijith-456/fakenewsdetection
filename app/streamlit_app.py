import os

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)


st.title("📰 Fake News Detection")

st.markdown(
    """
    Enter a news article below and use the trained
    **DistilBERT + BiLSTM + Attention** model to
    classify it as **FAKE** or **REAL**.
    """
)


article_text = st.text_area(
    "News Article",
    height=300,
    placeholder="Paste the complete news article here...",
)


analyze = st.button(
    "🔍 Analyze Article",
    use_container_width=True,
)


if analyze:

    text = article_text.strip()

    if not text:
        st.warning(
            "Please enter a news article before analyzing."
        )

    elif len(text) < 10:
        st.warning(
            "Please enter at least 10 characters."
        )

    elif len(text) > 10000:
        st.warning(
            "Article is too long. Please keep it below 10,000 characters."
        )

    else:

        with st.spinner("Analyzing article..."):

            try:

                response = httpx.post(
                    f"{API_URL}/predict",
                    json={"text": text},
                    timeout=60.0,
                )

                response.raise_for_status()

                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                fake_probability = result["probabilities"]["fake"]
                real_probability = result["probabilities"]["real"]

                st.divider()

                st.subheader("Prediction")

                if prediction == "FAKE":

                    st.error(
                        f"🔴 FAKE — {confidence:.2%} confidence"
                    )

                else:

                    st.success(
                        f"🟢 REAL — {confidence:.2%} confidence"
                    )

                st.subheader("Model Probabilities")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Fake",
                        f"{fake_probability:.2%}",
                    )

                with col2:
                    st.metric(
                        "Real",
                        f"{real_probability:.2%}",
                    )

                st.progress(
                    fake_probability,
                    text=f"Fake probability: {fake_probability:.2%}",
                )

                st.progress(
                    real_probability,
                    text=f"Real probability: {real_probability:.2%}",
                )

            except httpx.TimeoutException:

                st.error(
                    "The prediction API took too long to respond."
                )

            except httpx.ConnectError:

                st.error(
                    "Unable to connect to the prediction API."
                )

            except httpx.HTTPStatusError as error:

                st.error(
                    "The prediction API returned an error."
                )

                st.caption(
                    f"API status: {error.response.status_code}"
                )

            except httpx.HTTPError as error:

                st.error(
                    "An HTTP error occurred while contacting the API."
                )

                st.caption(
                    f"API error: {error}"
                )

            except (KeyError, TypeError, ValueError):

                st.error(
                    "The prediction API returned an unexpected response."
                )