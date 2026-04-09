import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize, pos_tag

# ======================
# DOWNLOAD NLTK DATA (runs once)
# ======================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# ======================
# EXTRACT NOUN ASPECTS
# ======================
def extract_noun_aspects(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)

    # Extract nouns
    nouns = [word.lower() for word, tag in tagged if tag.startswith('NN')]

    aspects = []
    for i in range(len(nouns) - 1):
        phrase = nouns[i] + " " + nouns[i + 1]

        if len(phrase) < 3:
            continue

        aspects.append(phrase)

    return list(set(aspects))


# ======================
# CLEAN FEATURES
# ======================
def clean_noun_features(features):
    cleaned = []

    for f in features:
        f = f.lower().strip()
        f = f.replace("couldn t", "couldnt")
        f = re.sub(r"[^a-z\s]", "", f)

        words = f.split()

        # Remove single words
        if len(words) < 2:
            continue

        phrase = " ".join(words)

        # Remove generic words
        if phrase in ["food", "place", "thing", "price"]:
            continue

        # Remove noisy tokens
        if phrase in ["soo", "breakie", "ca"]:
            continue

        if len(phrase) < 4:
            continue

        cleaned.append(phrase)

    return list(set(cleaned))


# ======================
# NORMALIZE TERMS
# ======================
def normalize_aspects(features):
    mapping = {
        "staffs": "staff",
        "services": "service",
        "foods": "food"
    }

    return [mapping.get(f, f) for f in features]


# ======================
# MAIN ASPECT PREDICTION
# ======================
def predict_aspects(review, review_embeddings, features_list, vectorizer, top_k=5):

    if not review or len(review.split()) < 3:
        return []

    # Step 1: Extract aspects from input
    input_aspects = extract_noun_aspects(review)

    # Step 2: Convert review to embedding
    emb = vectorizer.transform([review]).toarray()

    # Step 3: Similarity search
    sims = cosine_similarity(emb, review_embeddings)[0]
    top_indices = np.argsort(sims)[-top_k:]

    # Step 4: Retrieve similar features
    collected = []
    for idx in top_indices:
        collected.extend(features_list[idx])

    # Step 5: Clean retrieved features
    retrieved = clean_noun_features(collected)

    # Step 6: Combine + normalize
    combined = input_aspects + retrieved
    combined = normalize_aspects(combined)

    # Step 7: Final ranking
    final = sorted(set(combined), key=lambda x: -len(x))

    return final[:3]









# import re
# import numpy as np
# import spacy
# from sklearn.metrics.pairwise import cosine_similarity

# import spacy

# import spacy
# nlp = spacy.load("en_core_web_sm")

# def extract_noun_aspects(text):
#     doc = nlp(text)
#     aspects = []
#     for chunk in doc.noun_chunks:
#         tokens = [token.text.lower() for token in chunk if token.pos_ in ["NOUN", "PROPN"]]
#         if len(tokens) == 0:
#             continue
#         phrase = " ".join(tokens)
#         if len(phrase) < 3:
#             continue
#         aspects.append(phrase)
#     return list(set(aspects))

# def clean_noun_features(features):
#     cleaned = []
#     for f in features:
#         f = f.lower().strip()
#         f = f.replace("couldn t", "couldnt")
#         f = re.sub(r"[^a-z\s]", "", f)

#         doc = nlp(f)
#         tokens = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

#         if len(tokens) == 0:
#             continue

#         phrase = " ".join(tokens)

#         if len(tokens) == 1:
#             continue

#         if phrase in ["food", "place", "thing", "price"]:
#             continue

#         if phrase in ["soo", "breakie", "ca"]:
#             continue

#         if len(phrase) < 4:
#             continue

#         cleaned.append(phrase)

#     return list(set(cleaned))

# def normalize_aspects(features):
#     mapping = {
#         "staffs": "staff",
#         "services": "service",
#         "foods": "food"
#     }
#     return [mapping.get(f, f) for f in features]

# def predict_aspects(review, review_embeddings, features_list, vectorizer, top_k=5):

#     if not review or len(review.split()) < 3:
#         return []

#     input_aspects = extract_noun_aspects(review)

#     # TF-IDF embedding
#     emb = vectorizer.transform([review]).toarray()

#     sims = cosine_similarity(emb, review_embeddings)[0]
#     top_indices = np.argsort(sims)[-top_k:]

#     collected = []
#     for idx in top_indices:
#         collected.extend(features_list[idx])

#     retrieved = clean_noun_features(collected)

#     combined = input_aspects + retrieved
#     combined = normalize_aspects(combined)

#     final = sorted(set(combined), key=lambda x: -len(x))

#     return final[:3]
