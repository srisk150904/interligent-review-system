import streamlit as st
import joblib
import numpy as np

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(layout="wide")

st.title("🧠 Trust-Aware Review Intelligence System")

st.markdown("""
Analyze reviews using:
- 🛡️ Spam Detection  
- 😊 Sentiment Analysis  
- ⭐ Rating Consistency  

👉 Combined into a **Trust Score (0–5 scale)**
""")

# ======================
# CLEAN UI STYLES
# ======================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.card {
    padding:16px;
    border-radius:13px;
    background:#1E222B;
    border:1px solid #2A2F3A;
}
.title {
    font-size:13px;
    color:#9AA0A6;
}
.value {
    font-size:20px;
    font-weight:600;
}
.caption {
    font-size:15px;
    color:#9AA0A6;
}
.review-box {
    padding:16px;
    border-radius:10px;
    background:#111;
    border:1px solid #222;
    font-size:20px;
}
/* Text area label */
div[data-testid="stTextArea"] label {
    font-size:20px !important;
    font-weight:600;
}

/* Slider label */
div[data-testid="stSlider"] label {
    font-size:19px !important;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# st.title("Trust-Aware Review Intelligence")

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():
    spam_model = joblib.load("spam_logreg_model.pkl")
    spam_vectorizer = joblib.load("tfidf_vectorizer_spam.pkl")

    sentiment_model = joblib.load("sentiment_lg_model.pkl")
    sentiment_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return spam_model, spam_vectorizer, sentiment_model, sentiment_vectorizer


spam_model, spam_vectorizer, sentiment_model, sentiment_vectorizer = load_models()

# ======================
# HELPERS (UNCHANGED)
# ======================
label_reverse_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}

def convert_to_5_scale(score):
    scaled = score ** 0.6
    return round(scaled * 5, 2)

def get_spam_label(spam_prob, trust_score):
    if spam_prob < 0.4 or trust_score >= 4:
        return "✅ Genuine"
    elif spam_prob < 0.7 or trust_score > 3.2:
        return "🟡 Possibly Genuine"
    elif spam_prob < 0.8:
        return "⚠️ Suspicious (Review Needed)"
    elif spam_prob < 0.9:
        return "🚨 Likely Spam"
    else:
        return "🚨🚨 Very Likely Spam"

def sentiment_emoji_and_label(pred_class, percent, neutral_percent):
    if pred_class != 0:
        diff = percent - neutral_percent
        if diff <= 10:
            return "😐", "Neutral", neutral_percent
        percent = diff

    percent = max(0, min(99, percent))

    if pred_class == 0:
        return "😐", "Neutral", percent

    if pred_class == 1:
        if percent >= 96:
            return "🤩", "Extremely Positive", percent
        elif percent >= 87:
            return "😄", "Very Positive", percent
        elif percent >= 70:
            return "🙂", "Positive", percent
        else:
            return "😊", "Slightly Positive", percent

    if pred_class == -1:
        if percent >= 95:
            return "🤬", "Extremely Negative", percent
        elif percent >= 85:
            return "😠", "Very Negative", percent
        elif percent >= 70:
            return "😞", "Negative", percent
        else:
            return "😕", "Slightly Negative", percent

def check_rating_sentiment_mismatch(rating, pred_class):
    if rating <= 2:
        expected = -1
    elif rating == 3:
        expected = 0
    else:
        expected = 1

    if expected == pred_class:
        return "match", "✅ Rating and review are consistent"

    if abs(expected - pred_class) == 1:
        return "slight", "⚠️ Slight mismatch between rating and review"

    return "strong", "🚨 Strong mismatch: rating contradicts review"

def explain_spam(review):
    reasons = []
    if review.count("!") > 3:
        reasons.append("Excessive exclamation marks")
    words = review.lower().split()
    if len(words) > 0 and len(set(words)) < len(words) * 0.6:
        reasons.append("Repetitive words")
    if any(word in review.lower() for word in ["buy", "offer", "click", "free"]):
        reasons.append("Promotional language")
    if len(words) < 5:
        reasons.append("Very short / low information")
    return reasons

# ======================
# SESSION STATE
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Clear History"):
    st.session_state.history = []

# ======================
# INPUT
# ======================
review = st.text_area("✍️ Enter a review:")
rating = st.slider("⭐ Rating", 1, 5, 4)

if st.button("🔍 Analyze"):

    if review.strip():

        tfidf = sentiment_vectorizer.transform([review.lower()])
        sent_pred = sentiment_model.predict(tfidf)[0]
        sent_prob = sentiment_model.predict_proba(tfidf)[0]

        class_indices = {c: i for i, c in enumerate(sentiment_model.classes_)}
        neg_prob = sent_prob[class_indices[-1]]
        neu_prob = sent_prob[class_indices[0]]

        pred_prob = sent_prob[class_indices[sent_pred]]
        percent = min(99, int(round(pred_prob * 100)))
        neutral_percent = int(round(neu_prob * 100))

        emoji, intensity_label, adjusted_percent = sentiment_emoji_and_label(
            sent_pred, percent, neutral_percent
        )

        # ======================
        # SPAM
        # ======================
        spam_tfidf = spam_vectorizer.transform([review])
        spam_prob = float(spam_model.predict_proba(spam_tfidf)[0][1])

        # ======================
        # MISMATCH
        # ======================
        mismatch_type, mismatch_msg = check_rating_sentiment_mismatch(
            rating, sent_pred
        )

        # ======================
        # TRUST (UPDATED LOGIC)
        # ======================
        sentiment_conf = adjusted_percent / 100

        # Base trust from spam ONLY
        raw_trust = (1 - spam_prob)

        # Penalize mismatch strongly
        if mismatch_type == "strong":
            raw_trust *= 0.6
        elif mismatch_type == "slight":
            raw_trust *= 0.85

        # Small confidence adjustment (not dominance)
        raw_trust *= (0.8 + 0.2 * sentiment_conf)

        # Convert to 0–5 scale (safe clamp)
        trust_score = round(min(5, convert_to_5_scale(raw_trust) + 0.73), 2)

        # ======================
        # SPAM LABEL
        # ======================
        spam_label = get_spam_label(spam_prob, trust_score)

        # ======================
        # STORE
        # ======================
        st.session_state.history.append({
            "review": review,
            "spam_label": spam_label,
            "sentiment": adjusted_percent,
            "emoji": emoji,
            "intensity": intensity_label,
            "trust_score": trust_score,
            "mismatch_type": mismatch_type,
            "mismatch_msg": mismatch_msg,
            "probs": sent_prob
        })

# ======================
# DISPLAY (FINAL UI)
# ======================
st.markdown("---")
st.subheader("📋 Review Analysis")
for item in reversed(st.session_state.history):

    # Review box
    st.markdown(f"""
    <div class="review-box">
    📝 {item['review']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # FLOW: Sentiment → Spam → Trust
    col1, col2, col3 = st.columns(3)

    # SENTIMENT (UPDATED AS REQUESTED)
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="value">
                {item['emoji']} {item['intensity']} ({item['sentiment']}%)
            </div>
            <div class="caption">😊 Sentiment</div>
        </div>
        """, unsafe_allow_html=True)

    # SPAM
    with col2:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="value">{item['spam_label']}</div>
            <div class="caption">🛡️ Spam Detection</div>
        </div>
        """, unsafe_allow_html=True)

    # TRUST
    with col3:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="value">{item['trust_score']} / 5</div>
            <div class="caption">🧠 Trust Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if item["trust_score"] < 2.2:
        st.markdown("🚨 Low Trust / Suspicious Review")
    elif item["trust_score"] < 3:
        st.markdown("🟡 Moderately Trustable Review")
    elif item["trust_score"] < 4:
        st.markdown("✅ Highly Trustworthy Review")
    else:
        st.markdown("✅✅ Highly Trustworthy Review")
    
    # Consistency
    if item["mismatch_type"] == "match":
        st.success(item["mismatch_msg"])
    elif item["mismatch_type"] == "slight":
        st.warning(item["mismatch_msg"])
    else:
        st.error(item["mismatch_msg"])

    # Spam reasons
    reasons = explain_spam(item["review"])
    if reasons:
        with st.expander("Details"):
            for r in reasons:
                st.write(f"- {r}")

    # Sentiment breakdown
    with st.expander("📊 Sentiment Breakdown"):
            for c, p in zip(sentiment_model.classes_, item["probs"]):
                st.progress(float(p))
                st.write(f"{label_reverse_map[c]}: {p*100:.1f}%")

    st.markdown("---")





# import streamlit as st
# import joblib
# import numpy as np

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(page_title="Trust-Aware Review Intelligence", layout="wide")

# st.title("🧠 Trust-Aware Review Intelligence System")

# st.markdown("""
# Analyze reviews using:
# - 🛡️ Spam Detection  
# - 😊 Sentiment Analysis  
# - ⭐ Rating Consistency  

# 👉 Combined into a **Trust Score (0–5 scale)**
# """)

# # ======================
# # LOAD MODELS
# # ======================
# @st.cache_resource
# def load_models():
#     spam_model = joblib.load("spam_logreg_model.pkl")
#     spam_vectorizer = joblib.load("tfidf_vectorizer_spam.pkl")

#     sentiment_model = joblib.load("sentiment_lg_model.pkl")
#     sentiment_vectorizer = joblib.load("tfidf_vectorizer.pkl")

#     return spam_model, spam_vectorizer, sentiment_model, sentiment_vectorizer


# spam_model, spam_vectorizer, sentiment_model, sentiment_vectorizer = load_models()

# # ======================
# # HELPERS
# # ======================
# label_reverse_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}


# # ✅ TRUST SCORE CALIBRATION (IMPORTANT)
# def convert_to_5_scale(score):
#     """
#     Converts raw 0–1 trust score → 0–5 human scale
#     Non-linear mapping for better spread
#     """
#     scaled = score ** 0.6   # boost mid values
#     return round(scaled * 5, 2)


# def get_spam_label(spam_prob):

#     if spam_prob < 0.4:
#         return "✅ Genuine"
#     elif spam_prob < 0.6:
#         return "🟡 Possibly Genuine"
#     elif spam_prob < 0.75:
#         return "⚠️ Suspicious (Review Needed)"
#     elif spam_prob < 0.9:
#         return "🚨 Likely Spam"
#     else:
#         return "🚨🚨 Very Likely Spam"


# def sentiment_emoji_and_label(pred_class, percent, neutral_percent):

#     if pred_class != 0:
#         diff = percent - neutral_percent

#         if diff <= 10:
#             return "😐", "Neutral", neutral_percent

#         percent = diff

#     percent = max(0, min(99, percent))

#     if pred_class == 0:
#         return "😐", "Neutral", percent

#     if pred_class == 1:
#         if percent >= 96:
#             return "🤩", "Extremely Positive", percent
#         elif percent >= 87:
#             return "😄", "Very Positive", percent
#         elif percent >= 70:
#             return "🙂", "Positive", percent
#         else:
#             return "😊", "Slightly Positive", percent

#     if pred_class == -1:
#         if percent >= 95:
#             return "🤬", "Extremely Negative", percent
#         elif percent >= 85:
#             return "😠", "Very Negative", percent
#         elif percent >= 70:
#             return "😞", "Negative", percent
#         else:
#             return "😕", "Slightly Negative", percent


# def check_rating_sentiment_mismatch(rating, pred_class):

#     if rating <= 2:
#         expected = -1
#     elif rating == 3:
#         expected = 0
#     else:
#         expected = 1

#     if expected == pred_class:
#         return "match", "✅ Rating and review are consistent"

#     if abs(expected - pred_class) == 1:
#         return "slight", "⚠️ Slight mismatch between rating and review"

#     return "strong", "🚨 Strong mismatch: rating contradicts review"


# def explain_spam(review):
#     reasons = []

#     if review.count("!") > 3:
#         reasons.append("Excessive exclamation marks")

#     words = review.lower().split()
#     if len(words) > 0 and len(set(words)) < len(words) * 0.6:
#         reasons.append("Repetitive words")

#     if any(word in review.lower() for word in ["buy", "offer", "click", "free"]):
#         reasons.append("Promotional language")

#     if len(words) < 5:
#         reasons.append("Very short / low information")

#     return reasons


# # ======================
# # SESSION STATE
# # ======================
# if "history" not in st.session_state:
#     st.session_state.history = []

# if st.button("🗑️ Clear History"):
#     st.session_state.history = []

# # ======================
# # INPUT
# # ======================
# review = st.text_area("✍️ Enter a review:")
# rating = st.slider("⭐ Rating", 1, 5, 4)

# if st.button("🔍 Analyze"):

#     if review.strip():

#         # ======================
#         # SENTIMENT
#         # ======================
#         tfidf = sentiment_vectorizer.transform([review.lower()])

#         sent_pred = sentiment_model.predict(tfidf)[0]
#         sent_prob = sentiment_model.predict_proba(tfidf)[0]

#         class_indices = {c: i for i, c in enumerate(sentiment_model.classes_)}

#         neg_prob = sent_prob[class_indices[-1]]
#         neu_prob = sent_prob[class_indices[0]]

#         pred_prob = sent_prob[class_indices[sent_pred]]
#         percent = min(99, int(round(pred_prob * 100)))
#         neutral_percent = int(round(neu_prob * 100))

#         emoji, intensity_label, adjusted_percent = sentiment_emoji_and_label(
#             sent_pred, percent, neutral_percent
#         )

#         # ======================
#         # SPAM
#         # ======================
#         spam_tfidf = spam_vectorizer.transform([review])
#         spam_prob = float(spam_model.predict_proba(spam_tfidf)[0][1])

#         spam_label = get_spam_label(spam_prob)

#         # ======================
#         # CONSISTENCY
#         # ======================
#         mismatch_type, mismatch_msg = check_rating_sentiment_mismatch(
#             rating, sent_pred
#         )

#         # ======================
#         # TRUST SCORE
#         # ======================
#         sentiment_conf = adjusted_percent / 100
#         raw_trust = (1 - spam_prob) * sentiment_conf

#         if mismatch_type == "strong":
#             raw_trust *= 0.7
#         elif mismatch_type == "slight":
#             raw_trust *= 0.85

#         trust_score = convert_to_5_scale(raw_trust)

#         # ======================
#         # STORE
#         # ======================
#         st.session_state.history.append({
#             "review": review,
#             "spam_label": spam_label,
#             "sentiment": adjusted_percent,
#             "emoji": emoji,
#             "intensity": intensity_label,
#             "trust_score": trust_score,
#             "mismatch_type": mismatch_type,
#             "mismatch_msg": mismatch_msg,
#             "probs": sent_prob
#         })

# # ======================
# # DISPLAY
# # ======================
# st.markdown("---")
# st.subheader("📋 Review Analysis")

# for item in reversed(st.session_state.history):

#     with st.container():

#         st.markdown(f"### 📝 {item['review']}")

#         col1, col2 = st.columns(2)

#         with col1:
#             spam_label = item.get("spam_label", "⚠️ Unknown (Old Data)")
#             st.markdown(f"### 🛡️ {spam_label}")

#         with col2:
#             st.metric("🧠 Trust Score (0–5)", f"{item['trust_score']}")

#         # Trust interpretation
#         if item["trust_score"] < 2.5:
#             st.error("🚨 Low Trust / Suspicious Review")
#         elif item["trust_score"] <= 3.5:
#             st.warning("🟡 Moderately Trustable Review")
#         else:
#             st.success("✅ Highly Trustworthy Review")

#         st.markdown(f"## {item['emoji']} {item['intensity']} ({item['sentiment']}%)")

#         if item["mismatch_type"] == "match":
#             st.success(item["mismatch_msg"])
#         elif item["mismatch_type"] == "slight":
#             st.warning(item["mismatch_msg"])
#         else:
#             st.error(item["mismatch_msg"])

#         reasons = explain_spam(item["review"])
#         if reasons:
#             st.warning("⚠️ Possible spam indicators:")
#             for r in reasons:
#                 st.write(f"- {r}")

#         with st.expander("📊 Sentiment Breakdown"):
#             for c, p in zip(sentiment_model.classes_, item["probs"]):
#                 st.progress(float(p))
#                 st.write(f"{label_reverse_map[c]}: {p*100:.1f}%")

#         st.markdown("---")

# import streamlit as st
# import joblib
# import numpy as np

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(page_title="AI Review Analyzer", layout="wide")

# st.title("🧠 AI Review Intelligence System")
# st.markdown("Analyze reviews using **Spam Detection + Sentiment Analysis**")

# # ======================
# # LOAD MODELS
# # ======================
# @st.cache_resource
# def load_models():
#     spam_model = joblib.load("spam_lightgbm_model.pkl")
#     sentiment_model = joblib.load("sentiment_lg_model.pkl")
#     vectorizer = joblib.load("tfidf_vectorizer.pkl")
#     return spam_model, sentiment_model, vectorizer

# spam_model, sentiment_model, vectorizer = load_models()

# # ======================
# # SENTIMENT HELPERS
# # ======================
# label_reverse_map = {
#     -1: "Negative",
#      0: "Neutral",
#      1: "Positive"
# }

# def sentiment_emoji_and_label(pred_class, percent):

#     if pred_class == 0:
#         return "😐", "Neutral"

#     if pred_class == 1:
#         if percent >= 96:
#             return "🤩", "Extremely Positive"
#         elif percent >= 87:
#             return "😄", "Very Positive"
#         elif percent >= 70:
#             return "🙂", "Positive"
#         else:
#             return "😊", "Slightly Positive"

#     if pred_class == -1:
#         if percent >= 95:
#             return "🤬", "Extremely Negative"
#         elif percent >= 85:
#             return "😠", "Very Negative"
#         elif percent >= 70:
#             return "😞", "Negative"
#         else:
#             return "😕", "Slightly Negative"

# # ======================
# # RATING vs SENTIMENT CHECK
# # ======================
# def check_rating_sentiment_mismatch(rating, pred_class):

#     if rating <= 2:
#         expected = -1
#     elif rating == 3:
#         expected = 0
#     else:
#         expected = 1

#     if expected == pred_class:
#         return "match", "✅ Rating and review are consistent"

#     if abs(expected - pred_class) == 1:
#         return "slight", "⚠️ Slight mismatch between rating and review"

#     return "strong", "🚨 Strong mismatch: rating contradicts review"

# # ======================
# # SPAM FEATURE ENGINEERING
# # ======================
# def build_spam_features(review, rating=4):

#     text = str(review)

#     tfidf_vec = vectorizer.transform([text])
#     tfidf_nonzero_ratio = tfidf_vec.getnnz() / tfidf_vec.shape[1]

#     features = np.array([[
#         rating,
#         1.0,
#         rating,
#         0.0,
#         len(text),
#         1.0,
#         tfidf_nonzero_ratio
#     ]])

#     return features

# # ======================
# # SESSION STATE
# # ======================
# if "history" not in st.session_state:
#     st.session_state.history = []

# # OPTIONAL: Clear button (good UX)
# if st.button("🗑️ Clear History"):
#     st.session_state.history = []

# # ======================
# # INPUT
# # ======================
# review = st.text_area("✍️ Enter a review:")
# rating = st.slider("⭐ Rating", 1, 5, 4)

# if st.button("🔍 Analyze"):

#     if review.strip():

#         # ======================
#         # SPAM MODEL
#         # ======================
#         try:
#             features = build_spam_features(review, rating)
#             spam_prob = float(spam_model.predict_proba(features)[0][1])
#         except:
#             spam_prob = 0.0

#         spam_pred = 1 if spam_prob > 0.2 else 0

#         # ======================
#         # SENTIMENT MODEL
#         # ======================
#         tfidf = vectorizer.transform([review.lower()])

#         sent_pred = sentiment_model.predict(tfidf)[0]
#         sent_prob = sentiment_model.predict_proba(tfidf)[0]

#         pred_prob = sent_prob[list(sentiment_model.classes_).index(sent_pred)]
#         percent = min(99, int(round(pred_prob * 100)))

#         emoji, intensity_label = sentiment_emoji_and_label(sent_pred, percent)

#         # ======================
#         # RATING vs SENTIMENT
#         # ======================
#         mismatch_type, mismatch_msg = check_rating_sentiment_mismatch(
#             rating,
#             sent_pred
#         )

#         # ======================
#         # STORE RESULT
#         # ======================
#         st.session_state.history.append({
#             "review": review,
#             "spam": spam_pred,
#             "spam_prob": spam_prob,
#             "emoji": emoji,
#             "intensity": intensity_label,
#             "percent": percent,
#             "probs": sent_prob,
#             "mismatch_type": mismatch_type,
#             "mismatch_msg": mismatch_msg
#         })

# # ======================
# # DISPLAY RESULTS
# # ======================
# st.markdown("---")
# st.subheader("📋 Review Analysis")

# for item in reversed(st.session_state.history):

#     with st.container():

#         st.markdown(f"### 📝 {item['review']}")

#         # Spam
#         if item["spam"] == 1:
#             st.error(f"🚨 Spam Detected ({item['spam_prob']:.2f})")
#         else:
#             st.success("✅ Genuine Review")

#         # Sentiment
#         st.markdown(f"## {item['emoji']} {item['intensity']} ({item['percent']}%)")

#         # SAFE mismatch handling (fix for KeyError)
#         mismatch_type = item.get("mismatch_type")
#         mismatch_msg = item.get("mismatch_msg")

#         if mismatch_type:
#             if mismatch_type == "match":
#                 st.success(mismatch_msg)
#             elif mismatch_type == "slight":
#                 st.warning(mismatch_msg)
#             else:
#                 st.error(mismatch_msg)

#         # Breakdown
#         with st.expander("📊 Sentiment Breakdown"):
#             for c, p in zip(sentiment_model.classes_, item["probs"]):
#                 st.progress(float(p))
#                 st.write(f"{label_reverse_map[c]}: {p*100:.1f}%")

#         st.markdown("---")







# import streamlit as st
# import joblib
# import numpy as np

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(page_title="AI Review Analyzer", layout="wide")

# st.title("🧠 AI Review Intelligence System")
# st.markdown("Analyze reviews using **Spam Detection + Sentiment Analysis**")

# # ======================
# # LOAD MODELS
# # ======================
# @st.cache_resource
# def load_models():
#     spam_model = joblib.load("spam_lightgbm_model.pkl")
#     sentiment_model = joblib.load("sentiment_lg_model.pkl")
#     vectorizer = joblib.load("tfidf_vectorizer.pkl")
#     return spam_model, sentiment_model, vectorizer

# spam_model, sentiment_model, vectorizer = load_models()

# # ======================
# # FEATURE ENGINEERING (MATCH TRAINING)
# # ======================
# def build_spam_features(review, rating=4):
    
#     text = str(review)

#     # basic
#     word_count = len(text.split())

#     # TF-IDF stats (IMPORTANT)
#     tfidf_vec = vectorizer.transform([text])
#     tfidf_nonzero_ratio = tfidf_vec.getnnz() / tfidf_vec.shape[1]

#     # ---- USER FEATURES (DEFAULT APPROXIMATION) ----
#     num_reviews_by_user = 1.0
#     avg_rating_by_user = rating
#     rating_std_by_user = 0.0
#     review_length_avg_user = len(text)
#     reviews_per_day_user = 1.0

#     # Final feature vector (ORDER MUST MATCH TRAINING)
#     features = np.array([[
#         rating,
#         num_reviews_by_user,
#         avg_rating_by_user,
#         rating_std_by_user,
#         review_length_avg_user,
#         reviews_per_day_user,
#         tfidf_nonzero_ratio
#     ]])

#     return features


# # ======================
# # SESSION STATE
# # ======================
# if "history" not in st.session_state:
#     st.session_state.history = []

# # ======================
# # INPUT
# # ======================
# review = st.text_area("✍️ Enter a review:")
# rating = st.slider("⭐ Rating", 1, 5, 4)

# if st.button("🔍 Analyze"):

#     if review.strip():

#         # ======================
#         # SPAM MODEL (FIXED PIPELINE)
#         # ======================
#         try:
#             features = build_spam_features(review, rating)
#             spam_prob = float(spam_model.predict_proba(features)[0][1])
#         except:
#             spam_prob = 0.0

#         spam_pred = 1 if spam_prob > 0.2 else 0   # lower threshold

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
#         # STORE RESULT
#         # ======================
#         st.session_state.history.append({
#             "review": review,
#             "spam": spam_pred,
#             "spam_prob": spam_prob,
#             "sentiment": sentiment,
#             "confidence": confidence
#         })

# # ======================
# # DISPLAY RESULTS
# # ======================
# st.markdown("---")
# st.subheader("📋 Review Analysis")

# for item in reversed(st.session_state.history):

#     with st.container():

#         st.markdown(f"**📝 Review:** {item['review']}")

#         if item["spam"] == 1:
#             st.markdown(f"🚨 **Spam** ({item['spam_prob']:.2f})")
#         else:
#             st.markdown("✅ **Genuine Review**")

#         st.markdown(f"😊 **Sentiment:** {item['sentiment']} ({item['confidence']}%)")

#         st.markdown("---")




# import streamlit as st
# import joblib

# # ======================
# # PAGE CONFIG
# # ======================
# st.set_page_config(page_title="AI Review Analyzer", layout="wide")

# st.title("🧠 AI Review Intelligence System")
# st.markdown("Analyze reviews using **Spam Detection + Sentiment Analysis**")

# # ======================
# # LOAD MODELS
# # ======================
# @st.cache_resource
# def load_models():

#     spam_model = joblib.load("spam_lightgbm_model.pkl")
#     sentiment_model = joblib.load("sentiment_lg_model.pkl")
#     vectorizer = joblib.load("tfidf_vectorizer.pkl")

#     return spam_model, sentiment_model, vectorizer


# spam_model, sentiment_model, vectorizer = load_models()

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
#         # STORE RESULT
#         # ======================
#         st.session_state.history.append({
#             "review": review,
#             "spam": spam_pred,
#             "spam_prob": spam_prob,
#             "sentiment": sentiment,
#             "confidence": confidence
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

#         st.markdown("</div>", unsafe_allow_html=True)











# import streamlit as st
# import joblib
# import numpy as np
# import re
# import os

# # ─── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Intelligent Review Analyzer",
#     page_icon="🔍",
#     layout="centered",
# )

# # ─── Custom CSS ────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

# /* ── Global reset ── */
# html, body, [class*="css"] {
#     font-family: 'DM Sans', sans-serif;
# }

# /* ── Background ── */
# .stApp {
#     background: #0a0a0f;
#     color: #e8e4dc;
# }

# /* ── Header ── */
# .hero-title {
#     font-family: 'Syne', sans-serif;
#     font-size: 2.6rem;
#     font-weight: 800;
#     letter-spacing: -0.03em;
#     line-height: 1.1;
#     background: linear-gradient(135deg, #f5e6c8 0%, #e8b86d 50%, #c97b3a 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     margin-bottom: 0.2rem;
# }
# .hero-sub {
#     font-size: 0.95rem;
#     font-weight: 300;
#     color: #7a7468;
#     letter-spacing: 0.06em;
#     text-transform: uppercase;
#     margin-bottom: 2.5rem;
# }

# /* ── Textarea ── */
# .stTextArea > div > div > textarea {
#     background: #12121a !important;
#     border: 1px solid #2a2830 !important;
#     border-radius: 12px !important;
#     color: #e8e4dc !important;
#     font-family: 'DM Sans', sans-serif !important;
#     font-size: 1rem !important;
#     padding: 16px !important;
#     caret-color: #e8b86d;
#     transition: border-color 0.2s;
# }
# .stTextArea > div > div > textarea:focus {
#     border-color: #e8b86d !important;
#     box-shadow: 0 0 0 2px rgba(232,184,109,0.15) !important;
# }

# /* ── Button ── */
# .stButton > button {
#     background: linear-gradient(135deg, #e8b86d, #c97b3a) !important;
#     color: #0a0a0f !important;
#     font-family: 'Syne', sans-serif !important;
#     font-weight: 700 !important;
#     font-size: 0.95rem !important;
#     letter-spacing: 0.04em !important;
#     border: none !important;
#     border-radius: 10px !important;
#     padding: 0.6rem 2.4rem !important;
#     cursor: pointer !important;
#     transition: opacity 0.2s, transform 0.15s !important;
# }
# .stButton > button:hover {
#     opacity: 0.88 !important;
#     transform: translateY(-1px) !important;
# }

# /* ── Result cards ── */
# .result-card {
#     background: #12121a;
#     border: 1px solid #2a2830;
#     border-radius: 14px;
#     padding: 1.4rem 1.6rem;
#     margin-bottom: 1rem;
#     position: relative;
#     overflow: hidden;
# }
# .result-card::before {
#     content: '';
#     position: absolute;
#     top: 0; left: 0; right: 0;
#     height: 3px;
# }
# .card-spam::before   { background: linear-gradient(90deg, #ff4545, #ff8c42); }
# .card-legit::before  { background: linear-gradient(90deg, #38d96a, #00c9a7); }
# .card-pos::before    { background: linear-gradient(90deg, #38d96a, #00c9a7); }
# .card-neg::before    { background: linear-gradient(90deg, #ff4545, #e84393); }
# .card-neu::before    { background: linear-gradient(90deg, #e8b86d, #c97b3a); }

# .card-label {
#     font-family: 'Syne', sans-serif;
#     font-size: 0.72rem;
#     font-weight: 700;
#     letter-spacing: 0.12em;
#     text-transform: uppercase;
#     color: #7a7468;
#     margin-bottom: 0.4rem;
# }
# .card-verdict {
#     font-family: 'Syne', sans-serif;
#     font-size: 1.7rem;
#     font-weight: 800;
#     letter-spacing: -0.02em;
#     line-height: 1;
#     margin-bottom: 0.5rem;
# }
# .card-sub {
#     font-size: 0.88rem;
#     color: #7a7468;
#     font-weight: 300;
# }

# /* ── Confidence bar ── */
# .conf-wrap { margin-top: 0.8rem; }
# .conf-label {
#     display: flex;
#     justify-content: space-between;
#     font-size: 0.78rem;
#     color: #7a7468;
#     margin-bottom: 0.3rem;
# }
# .conf-bar-bg {
#     background: #1e1e28;
#     border-radius: 99px;
#     height: 6px;
#     overflow: hidden;
# }
# .conf-bar-fill {
#     height: 100%;
#     border-radius: 99px;
#     transition: width 0.6s ease;
# }

# /* ── Divider ── */
# .divider {
#     border: none;
#     border-top: 1px solid #1e1e28;
#     margin: 1.8rem 0;
# }

# /* ── Tip box ── */
# .tip-box {
#     background: #12121a;
#     border: 1px solid #2a2830;
#     border-left: 3px solid #e8b86d;
#     border-radius: 8px;
#     padding: 0.9rem 1.2rem;
#     font-size: 0.85rem;
#     color: #7a7468;
#     margin-top: 1.5rem;
# }

# /* ── Hide Streamlit branding ── */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 680px; }
# </style>
# """, unsafe_allow_html=True)


# # ─── Load models ───────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_models():
#     base = os.path.dirname(__file__)

#     # Sentiment
#     sent_model = joblib.load(os.path.join(base, "sentiment_lg_model.pkl"))
#     tfidf      = joblib.load(os.path.join(base, "tfidf_vectorizer.pkl"))

#     # Spam (LightGBM + feature list)
#     spam_model    = joblib.load(os.path.join(base, "spam_lightgbm_model.pkl"))
#     features_list = joblib.load(os.path.join(base, "features_list.pkl"))

#     return sent_model, tfidf, spam_model, features_list


# # ─── Feature engineering helpers (mirror your training pipeline) ───────────────
# def count_exclamations(text: str) -> int:
#     return text.count("!")

# def uppercase_ratio(text: str) -> float:
#     letters = [c for c in text if c.isalpha()]
#     if not letters:
#         return 0.0
#     return sum(1 for c in letters if c.isupper()) / len(letters)

# def count_emojis(text: str) -> int:
#     emoji_pattern = re.compile(
#         "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
#         "\U0001F680-\U0001F9FF\U00002702-\U000027B0]+",
#         flags=re.UNICODE,
#     )
#     return len(emoji_pattern.findall(text))

# def extract_spam_features(review: str, rating: float, features_list: list) -> np.ndarray:
#     """
#     Build the feature vector the spam model expects.
#     We use sensible defaults for user-aggregated features since we have one review.
#     TF-IDF / PCA embedding features are zeroed out (model degrades gracefully).
#     """
#     text = str(review)
#     word_count   = len(text.split())
#     review_len   = len(text)
#     excl         = count_exclamations(text)
#     upper        = uppercase_ratio(text)
#     emoji        = count_emojis(text)

#     # Build a dict with every feature the model was trained on
#     feat_map = {
#         "Rating":                 rating,
#         "num_reviews_by_user":    1.0,          # single review → default 1
#         "avg_rating_by_user":     rating,
#         "rating_std_by_user":     0.0,
#         "review_length_avg_user": review_len,
#         "reviews_per_day_user":   1.0,
#         "tfidf_nonzero_ratio":    min(word_count / 100.0, 1.0),
#         "exclamation_count":      excl,
#         "uppercase_ratio":        upper,
#         "emoji_count":            emoji,
#     }

#     row = [feat_map.get(f, 0.0) for f in features_list]
#     return np.array(row, dtype=np.float32).reshape(1, -1)


# # ─── Sentiment helpers ─────────────────────────────────────────────────────────
# SENTIMENT_CONFIG = {
#     1:  {"label": "Positive",  "emoji": "😊", "color": "#38d96a", "card": "card-pos"},
#     0:  {"label": "Neutral",   "emoji": "😐", "color": "#e8b86d", "card": "card-neu"},
#     -1: {"label": "Negative",  "emoji": "😟", "color": "#ff4545", "card": "card-neg"},
# }

# def get_sentiment_intensity(prob: float) -> str:
#     if prob > 0.90: return "Extremely"
#     if prob > 0.75: return "Very"
#     if prob > 0.55: return "Moderately"
#     return "Slightly"


# # ─── UI ────────────────────────────────────────────────────────────────────────
# st.markdown('<p class="hero-title">Intelligent Review Analyzer</p>', unsafe_allow_html=True)
# st.markdown('<p class="hero-sub">Spam detection · Sentiment analysis</p>', unsafe_allow_html=True)

# review_text = st.text_area(
#     "Enter a product review",
#     placeholder="e.g. This product is absolutely amazing! Totally worth it...",
#     height=140,
#     label_visibility="collapsed",
# )

# rating = st.slider("Rating given by user", min_value=1, max_value=5, value=4, step=1)

# run = st.button("Analyze Review →")

# if run:
#     if not review_text.strip():
#         st.warning("Please enter a review to analyze.")
#     else:
#         try:
#             sent_model, tfidf, spam_model, features_list = load_models()
#         except Exception as e:
#             st.error(f"Could not load models: {e}")
#             st.stop()

#         # ── Sentiment ──────────────────────────────────────────────────────────
#         vec = tfidf.transform([review_text])
#         sent_pred  = sent_model.predict(vec)[0]
#         sent_proba = sent_model.predict_proba(vec)[0]
#         sent_conf  = float(np.max(sent_proba))

#         cfg = SENTIMENT_CONFIG.get(sent_pred, SENTIMENT_CONFIG[1])
#         intensity = get_sentiment_intensity(sent_conf)

#         # ── Spam ───────────────────────────────────────────────────────────────
#         spam_feats = extract_spam_features(review_text, rating, features_list)
#         spam_prob  = float(spam_model.predict_proba(spam_feats)[0][1])
#         SPAM_THRESHOLD = 0.184
#         is_spam    = spam_prob >= SPAM_THRESHOLD

#         # ── Render ─────────────────────────────────────────────────────────────
#         st.markdown('<hr class="divider">', unsafe_allow_html=True)

#         col1, col2 = st.columns(2)

#         with col1:
#             spam_card  = "card-spam" if is_spam else "card-legit"
#             spam_label = "Spam Review" if is_spam else "Genuine Review"
#             spam_emoji = "🚨" if is_spam else "✅"
#             spam_sub   = f"Spam probability: {spam_prob:.0%}"
#             bar_color  = "#ff4545" if is_spam else "#38d96a"
#             st.markdown(f"""
#             <div class="result-card {spam_card}">
#                 <div class="card-label">Spam Detection</div>
#                 <div class="card-verdict">{spam_emoji} {spam_label}</div>
#                 <div class="card-sub">{spam_sub}</div>
#                 <div class="conf-wrap">
#                     <div class="conf-label">
#                         <span>Confidence</span>
#                         <span>{spam_prob:.0%}</span>
#                     </div>
#                     <div class="conf-bar-bg">
#                         <div class="conf-bar-fill" style="width:{spam_prob*100:.1f}%;background:{bar_color};"></div>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#         with col2:
#             st.markdown(f"""
#             <div class="result-card {cfg['card']}">
#                 <div class="card-label">Sentiment Analysis</div>
#                 <div class="card-verdict">{cfg['emoji']} {cfg['label']}</div>
#                 <div class="card-sub">{intensity} {cfg['label'].lower()} · {sent_conf:.0%} confident</div>
#                 <div class="conf-wrap">
#                     <div class="conf-label">
#                         <span>Confidence</span>
#                         <span>{sent_conf:.0%}</span>
#                     </div>
#                     <div class="conf-bar-bg">
#                         <div class="conf-bar-fill" style="width:{sent_conf*100:.1f}%;background:{cfg['color']};"></div>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#         # ── Probability breakdown ───────────────────────────────────────────────
#         with st.expander("Sentiment class probabilities"):
#             classes = sent_model.classes_
#             for cls, p in sorted(zip(classes, sent_proba), key=lambda x: -x[1]):
#                 c = SENTIMENT_CONFIG.get(cls, SENTIMENT_CONFIG[0])
#                 st.markdown(f"""
#                 <div style="margin-bottom:0.6rem;">
#                     <div class="conf-label">
#                         <span>{c['emoji']} {c['label']}</span>
#                         <span>{p:.1%}</span>
#                     </div>
#                     <div class="conf-bar-bg">
#                         <div class="conf-bar-fill" style="width:{p*100:.1f}%;background:{c['color']};"></div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)

#         # ── Contextual tip ─────────────────────────────────────────────────────
#         if is_spam and sent_pred == 1:
#             tip = "⚠️ High spam probability with positive sentiment — pattern consistent with fake positive reviews."
#         elif is_spam and sent_pred == -1:
#             tip = "⚠️ Spam detected with negative sentiment — could be a competitor attack or bot review."
#         elif not is_spam and sent_pred == 1:
#             tip = "✅ Genuine positive review — this looks authentic and trustworthy."
#         elif not is_spam and sent_pred == -1:
#             tip = "✅ Genuine negative review — real user frustration, worth acting on."
#         else:
#             tip = "ℹ️ Neutral review classified as genuine. Low sentiment signal."

#         st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
