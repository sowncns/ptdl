import joblib
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from src.preprocess import preprocess

def train_and_save(csv_path):
    df = pd.read_csv(csv_path)

    X_text = (
        df["description"].fillna("") + " " +
        df["activity"].fillna("") + " " +
        df["place"].fillna("")
    ).apply(preprocess)

    y = df["primary_type"].str.get_dummies(sep=";")

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test)

    print("Train time:", round(train_time,2))
    print("F1 micro:", f1_score(y_test, y_pred, average="micro"))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "model_primary.pkl")
    joblib.dump(vectorizer, "vectorizer_primary.pkl")
    joblib.dump(y.columns.tolist(), "labels.pkl")
train_and_save('data/data.csv')