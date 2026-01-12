from sklearn.metrics.pairwise import cosine_similarity
from src.predict_label import predict_primary
from src.vectorizer import build_place_vectors

def recommend_places(csv_path, user_text, limit=5):
    df, tfidf, X = build_place_vectors(csv_path)
    primary_scores = dict(predict_primary(user_text))

    user_vec = tfidf.transform([user_text])
    sims = cosine_similarity(user_vec, X)[0]

    scores = []
    for i, row in df.iterrows():
        s = 0.5 * sims[i]
        if row["primary_type"] in primary_scores:
            s += 0.5 * primary_scores[row["primary_type"]]
        scores.append(s)

    df["final_score"] = scores
    return df.sort_values("final_score", ascending=False).head(limit)
