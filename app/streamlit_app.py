import httpx
import streamlit as st


API_URL = "http://127.0.0.1:8000"


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

    if not article_text.strip():

        st.warning(
            "Please enter a news article before analyzing."
        )

    else:

        with st.spinner("Analyzing article..."):

            try:

                response = httpx.post(
                    f"{API_URL}/predict",
                    json={
                        "text": article_text
                    },
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
                        f"{fake_probability:.2%}"
                    )

                with col2:

                    st.metric(
                        "Real",
                        f"{real_probability:.2%}"
                    )

                st.progress(
                    fake_probability,
                    text=f"Fake probability: {fake_probability:.2%}"
                )

                st.progress(
                    real_probability,
                    text=f"Real probability: {real_probability:.2%}"
                )

            except httpx.HTTPError as error:

                st.error(
                    "Unable to connect to the prediction API."
                )

                st.caption(
                    f"API error: {error}"
                )