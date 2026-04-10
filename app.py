import streamlit as st
import joblib
import numpy as np
import re
import os
from typing import List

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Review Analyzer",
    page_icon="🔍",
    layout="centered",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0a0f;
    color: #e8e4dc;
}

/* ── Header ── */
@@ -166,168 +167,224 @@ html, body, [class*="css"] {
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #7a7468;
    margin-top: 1.5rem;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 680px; }
</style>
""", unsafe_allow_html=True)


# ─── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)

    # Sentiment
    sent_model = joblib.load(os.path.join(base, "sentiment_lg_model.pkl"))
    tfidf      = joblib.load(os.path.join(base, "tfidf_vectorizer.pkl"))

    # Spam (LightGBM + feature list)
    spam_model    = joblib.load(os.path.join(base, "spam_lightgbm_model.pkl"))
    features_list = joblib.load(os.path.join(base, "features_list.pkl"))
    features_path = os.path.join(base, "features_list.pkl")
    features_list = load_feature_list(features_path, spam_model)

    return sent_model, tfidf, spam_model, features_list


def load_feature_list(features_path: str, spam_model) -> List[str]:
    """
    Load feature names robustly.
    If features_list.pkl is missing/corrupted, fall back to model feature names.
    """
    try:
        if os.path.exists(features_path):
            loaded = joblib.load(features_path)
            if isinstance(loaded, (list, tuple)) and len(loaded) > 0:
                return list(loaded)
    except Exception:
        pass

    if hasattr(spam_model, "feature_name_") and len(spam_model.feature_name_) > 0:
        return list(spam_model.feature_name_)

    n_features = int(getattr(spam_model, "n_features_in_", 0))
    return [f"f_{i}" for i in range(n_features)]


# ─── Feature engineering helpers (mirror your training pipeline) ───────────────
def count_exclamations(text: str) -> int:
    return text.count("!")

def uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def count_emojis(text: str) -> int:
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF\U00002702-\U000027B0]+",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text))

def extract_spam_features(review: str, rating: float, features_list: list) -> np.ndarray:
def extract_spam_features(
    review: str,
    rating: float,
    features_list: list,
    num_reviews_by_user: float,
    avg_rating_by_user: float,
    rating_std_by_user: float,
    review_length_avg_user: float,
    reviews_per_day_user: float,
) -> np.ndarray:
    """
    Build the feature vector the spam model expects.
    We use sensible defaults for user-aggregated features since we have one review.
    TF-IDF / PCA embedding features are zeroed out (model degrades gracefully).
    """
    text = str(review)
    word_count   = len(text.split())
    review_len   = len(text)
    excl         = count_exclamations(text)
    upper        = uppercase_ratio(text)
    emoji        = count_emojis(text)

    # Build a dict with every feature the model was trained on
    feat_map = {
        "Rating":                 rating,
        "num_reviews_by_user":    1.0,          # single review → default 1
        "avg_rating_by_user":     rating,
        "rating_std_by_user":     0.0,
        "review_length_avg_user": review_len,
        "reviews_per_day_user":   1.0,
        "num_reviews_by_user":    num_reviews_by_user,
        "avg_rating_by_user":     avg_rating_by_user,
        "rating_std_by_user":     rating_std_by_user,
        "review_length_avg_user": review_length_avg_user,
        "reviews_per_day_user":   reviews_per_day_user,
        "tfidf_nonzero_ratio":    min(word_count / 100.0, 1.0),
        "exclamation_count":      excl,
        "uppercase_ratio":        upper,
        "emoji_count":            emoji,
    }

    # If the model contains bert_pca_* features, we keep them as 0.0 at inference.
    # This keeps compatibility with a hybrid-trained model when only raw text is provided.
    row = [feat_map.get(f, 0.0) for f in features_list]
    return np.array(row, dtype=np.float32).reshape(1, -1)


# ─── Sentiment helpers ─────────────────────────────────────────────────────────
SENTIMENT_CONFIG = {
    1:  {"label": "Positive",  "emoji": "😊", "color": "#38d96a", "card": "card-pos"},
    0:  {"label": "Neutral",   "emoji": "😐", "color": "#e8b86d", "card": "card-neu"},
    -1: {"label": "Negative",  "emoji": "😟", "color": "#ff4545", "card": "card-neg"},
}

def get_sentiment_intensity(prob: float) -> str:
    if prob > 0.90: return "Extremely"
    if prob > 0.75: return "Very"
    if prob > 0.55: return "Moderately"
    return "Slightly"


# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Intelligent Review Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Spam detection · Sentiment analysis</p>', unsafe_allow_html=True)

review_text = st.text_area(
    "Enter a product review",
    placeholder="e.g. This product is absolutely amazing! Totally worth it...",
    height=140,
    label_visibility="collapsed",
)

rating = st.slider("Rating given by user", min_value=1, max_value=5, value=4, step=1)
spam_threshold = st.slider(
    "Spam decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.184,
    step=0.01,
    help="Lower = higher recall (catch more spam), Higher = higher precision (fewer false positives).",
)

with st.expander("Optional user behavior inputs (improves spam score quality)"):
    st.caption("If unknown, keep defaults. These features are used by your spam model.")
    num_reviews_by_user = st.number_input("num_reviews_by_user", min_value=1.0, value=1.0, step=1.0)
    avg_rating_by_user = st.slider("avg_rating_by_user", min_value=1.0, max_value=5.0, value=float(rating), step=0.1)
    rating_std_by_user = st.number_input("rating_std_by_user", min_value=0.0, value=0.0, step=0.1)
    review_length_avg_user = st.number_input("review_length_avg_user", min_value=1.0, value=80.0, step=1.0)
    reviews_per_day_user = st.number_input("reviews_per_day_user", min_value=0.0, value=1.0, step=0.1)

run = st.button("Analyze Review →")

if run:
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
    else:
        try:
            sent_model, tfidf, spam_model, features_list = load_models()
        except Exception as e:
            st.error(f"Could not load models: {e}")
            st.stop()

        # ── Sentiment ──────────────────────────────────────────────────────────
        vec = tfidf.transform([review_text])
        sent_pred  = sent_model.predict(vec)[0]
        sent_proba = sent_model.predict_proba(vec)[0]
        sent_conf  = float(np.max(sent_proba))

        cfg = SENTIMENT_CONFIG.get(sent_pred, SENTIMENT_CONFIG[1])
        intensity = get_sentiment_intensity(sent_conf)

        # ── Spam ───────────────────────────────────────────────────────────────
        spam_feats = extract_spam_features(review_text, rating, features_list)
        spam_feats = extract_spam_features(
            review=review_text,
            rating=rating,
            features_list=features_list,
            num_reviews_by_user=num_reviews_by_user,
            avg_rating_by_user=avg_rating_by_user,
            rating_std_by_user=rating_std_by_user,
            review_length_avg_user=review_length_avg_user,
            reviews_per_day_user=reviews_per_day_user,
        )
        spam_prob  = float(spam_model.predict_proba(spam_feats)[0][1])
        SPAM_THRESHOLD = 0.184
        is_spam    = spam_prob >= SPAM_THRESHOLD
        is_spam    = spam_prob >= spam_threshold

        # ── Render ─────────────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            spam_card  = "card-spam" if is_spam else "card-legit"
            spam_label = "Spam Review" if is_spam else "Genuine Review"
            spam_emoji = "🚨" if is_spam else "✅"
            spam_sub   = f"Spam probability: {spam_prob:.0%}"
            spam_sub   = f"Spam probability: {spam_prob:.0%} (threshold {spam_threshold:.2f})"
            bar_color  = "#ff4545" if is_spam else "#38d96a"
            st.markdown(f"""
            <div class="result-card {spam_card}">
                <div class="card-label">Spam Detection</div>
                <div class="card-verdict">{spam_emoji} {spam_label}</div>
                <div class="card-sub">{spam_sub}</div>
                <div class="conf-wrap">
                    <div class="conf-label">
                        <span>Confidence</span>
                        <span>{spam_prob:.0%}</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{spam_prob*100:.1f}%;background:{bar_color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card {cfg['card']}">
                <div class="card-label">Sentiment Analysis</div>
                <div class="card-verdict">{cfg['emoji']} {cfg['label']}</div>
                <div class="card-sub">{intensity} {cfg['label'].lower()} · {sent_conf:.0%} confident</div>
                <div class="conf-wrap">
