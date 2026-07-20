import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import preprocess

# dùng cho place similarity
def build_place_vectors(csv_path):
    df = pd.read_csv(csv_path)

    df["text"] = (
        df["place"].fillna("") + " " +
        df["description"].fillna("") + " " +
        (df["activity"].fillna("") + " ") * 2
    ).apply(preprocess)

    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    X = tfidf.fit_transform(df["text"])
    return df, tfidf, X
