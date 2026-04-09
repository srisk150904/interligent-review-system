import streamlit as st
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="AI Review Analyzer", layout="wide")

st.title("🧠 AI Review Intelligence System")
st.markdown("Analyze reviews using **Spam Detection + Sentiment Analysis**")

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():

    spam_model = joblib.load("spam_lightgbm_model (1).pkl")
    sentiment_model = joblib.load("sentiment_lg_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return spam_model, sentiment_model, vectorizer


spam_model, sentiment_model, vectorizer = load_models()

# ======================
# SESSION STATE
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

# ======================
# INPUT
# ======================
review = st.text_area("✍️ Enter a review:")

if st.button("🔍 Analyze"):

    if review.strip():

        # ======================
        # SPAM MODEL
        # ======================
        try:
            spam_prob = spam_model.predict_proba([review])[0][1]
        except:
            spam_prob = 0.0

        spam_pred = 1 if spam_prob > 0.5 else 0

        # ======================
        # SENTIMENT MODEL
        # ======================
        tfidf = vectorizer.transform([review])

        sent_pred = sentiment_model.predict(tfidf)[0]
        sent_prob = sentiment_model.predict_proba(tfidf)[0]

        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        sentiment = label_map.get(sent_pred, "Unknown")

        confidence = int(max(sent_prob) * 100)

        # ======================
        # STORE RESULT
        # ======================
        st.session_state.history.append({
            "review": review,
            "spam": spam_pred,
            "spam_prob": spam_prob,
            "sentiment": sentiment,
            "confidence": confidence
        })

# ======================
# DISPLAY RESULTS
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
            st.markdown("✅ **Genuine Review**")

        # Sentiment
        st.markdown(f"😊 **Sentiment:** {item['sentiment']} ({item['confidence']}%)")

        st.markdown("</div>", unsafe_allow_html=True)

















# import streamlit as st
# import joblib
# import numpy as np

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(page_title="AI Review Analyzer", layout="wide")

# st.title("🧠 AI Review Intelligence System")
# st.markdown("Analyze reviews using **Spam + Sentiment + Aspect Extraction**")

# # ======================
# # LOAD MODELS
# # ======================
# @st.cache_resource
# def load_models():

#     # ✅ USE YOUR EXACT FILE NAMES
#     spam_model = joblib.load("spam_lightgbm_model (1).pkl")
#     sentiment_model = joblib.load("sentiment_lg_model.pkl")
#     vectorizer = joblib.load("tfidf_vectorizer.pkl")

#     # ✅ FIXED: only 5 parts
#     parts = [np.load(f"review_part_{i}.npy", mmap_mode='r') for i in range(10)]
#     review_embeddings = np.concatenate(parts, axis=0)

#     features_list = joblib.load("features_list.pkl")

#     return spam_model, sentiment_model, vectorizer, review_embeddings, features_list


# spam_model, sentiment_model, vectorizer, review_embeddings, features_list = load_models()

# from aspect_model import predict_aspects

# # ======================
# # SESSION STATE
# # ======================
# if "history" not in st.session_state:
#     st.session_state.history = []

# # ======================
# # INPUT
# # ======================
# review = st.text_area("✍️ Enter a review:")

# if st.button("🔍 Analyze"):

#     if review.strip():

#         # ======================
#         # SPAM MODEL
#         # ======================
#         try:
#             spam_prob = spam_model.predict_proba([review])[0][1]
#         except:
#             # fallback if model expects numeric features
#             spam_prob = 0.0

#         spam_pred = 1 if spam_prob > 0.5 else 0

#         # ======================
#         # SENTIMENT MODEL
#         # ======================
#         tfidf = vectorizer.transform([review])

#         sent_pred = sentiment_model.predict(tfidf)[0]
#         sent_prob = sentiment_model.predict_proba(tfidf)[0]

#         label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
#         sentiment = label_map.get(sent_pred, "Unknown")

#         confidence = int(max(sent_prob) * 100)

#         # ======================
#         # ASPECT MODEL
#         # ======================
#         aspects = predict_aspects(
#             review,
#             review_embeddings,
#             features_list,
#             vectorizer
#         )

#         # ======================
#         # STORE RESULT
#         # ======================
#         st.session_state.history.append({
#             "review": review,
#             "spam": spam_pred,
#             "spam_prob": spam_prob,
#             "sentiment": sentiment,
#             "confidence": confidence,
#             "aspects": aspects
#         })

# # ======================
# # DISPLAY RESULTS
# # ======================
# st.markdown("---")
# st.subheader("📋 Review Analysis")

# for item in reversed(st.session_state.history):

#     with st.container():

#         st.markdown("""
#         <div style="
#             border:1px solid #ddd;
#             padding:15px;
#             border-radius:10px;
#             margin-bottom:10px;
#             background-color:#fafafa;
#         ">
#         """, unsafe_allow_html=True)

#         # Review
#         st.markdown(f"**📝 Review:** {item['review']}")

#         # Spam
#         if item["spam"] == 1:
#             st.markdown(f"🚨 **Spam** ({item['spam_prob']:.2f})")
#         else:
#             st.markdown("✅ **Genuine Review**")

#         # Sentiment
#         st.markdown(f"😊 **Sentiment:** {item['sentiment']} ({item['confidence']}%)")

#         # Aspects
#         st.markdown("🏷️ **Aspects:**")
#         if item["aspects"]:
#             cols = st.columns(len(item["aspects"]))
#             for i, a in enumerate(item["aspects"]):
#                 cols[i].info(a)
#         else:
#             st.write("No aspects found")

#         st.markdown("</div>", unsafe_allow_html=True)
