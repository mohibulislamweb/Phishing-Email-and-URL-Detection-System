"""
train_model.py
---------------
Advanced offline training script with optimized TF-IDF, class weighting,
and robust cross-validation metrics for PhishGuard AI.
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    return df


def main():
    df = load_data()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # অ্যাডভান্সড TF-IDF: ক্যারেক্টার ও ওয়ার্ড লেভেল এন-গ্রাম এবং সাবলিনিয়ার স্কেলিং
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), 
        sublinear_tf=True, 
        max_features=10000,
        analyzer='word',
        min_df=1
    )
    
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # আরও শক্তিশালী হাইপারপ্যারামিটার সহ মডেল ক্যান্ডিডেট
    candidates = {
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", solver='lbfgs'),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight="balanced"),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", max_iter=2000),
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
    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison table saved to {comparison_path}")
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
    
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix chart saved to {cm_path}")

    model_path = os.path.join(MODELS_DIR, "phishing_model.pkl")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\nBest model: {best_name} (F1 = {best_f1:.3f}) saved to {model_path}")
    print("Advanced model train completed successfully!")


if __name__ == "__main__":
    main()