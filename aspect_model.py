
import re
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub

nlp = spacy.load("en_core_web_sm")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def extract_noun_aspects(text):
    doc = nlp(text)
    aspects = []
    for chunk in doc.noun_chunks:
        tokens = [token.text.lower() for token in chunk if token.pos_ in ["NOUN", "PROPN"]]
        if len(tokens) == 0:
            continue
        phrase = " ".join(tokens)
        if len(phrase) < 3:
            continue
        aspects.append(phrase)
    return list(set(aspects))

def clean_noun_features(features):
    cleaned = []
    for f in features:
        f = f.lower().strip()
        f = f.replace("couldn t", "couldnt")
        f = re.sub(r"[^a-z\s]", "", f)

        doc = nlp(f)
        tokens = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

        if len(tokens) == 0:
            continue

        phrase = " ".join(tokens)

        if len(tokens) == 1:
            continue

        if phrase in ["food", "place", "thing", "price"]:
            continue

        if phrase in ["soo", "breakie", "ca"]:
            continue

        if len(phrase) < 4:
            continue

        cleaned.append(phrase)

    return list(set(cleaned))

def normalize_aspects(features):
    mapping = {
        "staffs": "staff",
        "services": "service",
        "foods": "food"
    }
    return [mapping.get(f, f) for f in features]

def predict_aspects(review, review_embeddings, features_list, top_k=5):

    if not review or len(review.split()) < 3:
        return []

    input_aspects = extract_noun_aspects(review)

    emb = embed([review]).numpy()
    sims = cosine_similarity(emb, review_embeddings)[0]

    top_indices = np.argsort(sims)[-top_k:]

    collected = []
    for idx in top_indices:
        collected.extend(features_list[idx])

    retrieved = clean_noun_features(collected)

    combined = input_aspects + retrieved
    combined = normalize_aspects(combined)

    final = sorted(set(combined), key=lambda x: -len(x))

    return final[:3]
