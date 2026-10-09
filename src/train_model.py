"""
train_model.py
---------------
Offline training + evaluation script for the PhishGuard text classifier.

This is the script to run for the project REPORT: it splits dataset.csv
into train/test sets, trains several models, prints accuracy / precision
/ recall / F1 for each, saves a comparison table (model_comparison.csv)
and a confusion-matrix chart (confusion_matrix.png) for the best model,
then saves that model + its vectorizer for reuse.

Note: the interactive Streamlit app (app.py) trains its own small
in-memory model for live demos and does its own rule-based + URL-based
scoring — it does not depend on the .pkl files this script produces.
Run this script separately whenever you need report-ready numbers/plots
or want to retrain on a bigger dataset.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

DATA_PATH = "dataset.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    return df


def main():
    df = load_data()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }

    results = []
    best_name, best_model, best_f1 = None, None, -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        results.append([name, acc, prec, rec, f1])

        print(f"\n--- {name} ---")
        print(classification_report(y_test, preds, target_names=["Legitimate", "Phishing"], zero_division=0))

        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
    results_df.to_csv("model_comparison.csv", index=False)
    print("\nModel comparison table saved to model_comparison.csv")
    print(results_df.to_string(index=False))

    # Confusion matrix for the best-performing model
    preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Phishing"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix chart saved to confusion_matrix.png")

    joblib.dump(best_model, "phishing_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print(f"\nBest model: {best_name} (F1 = {best_f1:.3f}) saved to phishing_model.pkl")
    print("Model train completed successfully!")


if __name__ == "__main__":
    main()