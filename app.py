import streamlit as st
import joblib
import numpy as np

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="AI Review Analyzer", layout="wide")

st.title("🧠 AI Review Intelligence System")
st.markdown("Analyze reviews using **Spam + Sentiment + Aspect Extraction**")

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():
    spam_model = joblib.load("spam_model.pkl")
    sentiment_model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # load embeddings (split)
    parts = [np.load(f"review_part_{i}.npy") for i in range(10)]
    review_embeddings = np.vstack(parts)

    features_list = joblib.load("features_list.pkl")

    return spam_model, sentiment_model, vectorizer, review_embeddings, features_list

spam_model, sentiment_model, vectorizer, review_embeddings, features_list = load_models()

from aspect_model import predict_aspects

# ======================
# SESSION STATE (history)
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# INPUT
# ======================
review = st.text_area("✍️ Enter a review:")

if st.button("🔍 Analyze"):

    if review.strip():

        # ===== Spam =====
        spam_prob = spam_model.predict_proba([review])[0][1]
        spam_pred = 1 if spam_prob > 0.5 else 0

        # ===== Sentiment =====
        tfidf = vectorizer.transform([review])
        sent_pred = sentiment_model.predict(tfidf)[0]
        sent_prob = sentiment_model.predict_proba(tfidf)[0]

        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        sentiment = label_map[sent_pred]
        confidence = int(max(sent_prob) * 100)

        # ===== Aspects =====
        aspects = predict_aspects(
            review,
            review_embeddings,
            features_list,
            vectorizer
        )

        # Save to history
        st.session_state.history.append({
            "review": review,
            "spam": spam_pred,
            "spam_prob": spam_prob,
            "sentiment": sentiment,
            "confidence": confidence,
            "aspects": aspects
        })

# ======================
# DISPLAY CARDS
# ======================
st.markdown("---")
st.subheader("📋 Review Analysis")

for item in reversed(st.session_state.history):

    with st.container():
        st.markdown("""
        <div style="
            border:1px solid #ddd;
            padding:15px;
            border-radius:10px;
            margin-bottom:10px;
            background-color:#fafafa;
        ">
        """, unsafe_allow_html=True)

        # Review
        st.markdown(f"**📝 Review:** {item['review']}")

        # Spam
        if item["spam"] == 1:
            st.markdown(f"🚨 **Spam** ({item['spam_prob']:.2f})")
        else:
            st.markdown(f"✅ **Genuine**")

        # Sentiment
        st.markdown(f"😊 **Sentiment:** {item['sentiment']} ({item['confidence']}%)")

        # Aspects
        st.markdown("🏷️ **Aspects:**")
        if item["aspects"]:
            cols = st.columns(len(item["aspects"]))
            for i, a in enumerate(item["aspects"]):
                cols[i].info(a)
        else:
            st.write("No aspects found")

        st.markdown("</div>", unsafe_allow_html=True)
