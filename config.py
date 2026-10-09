"""
Configuration settings and global constants for PhishGuard AI.
"""

# Risk Thresholds
MEDIUM_RISK_THRESHOLD = 36.0
HIGH_RISK_THRESHOLD = 70.0

# Model Ensemble Weights
ENSEMBLE_WEIGHTS = {
    "Weighted Rule Engine v2": 0.40,
    "Random Forest (NLP)": 0.20,
    "LightGBM": 0.20,
    "Naive Bayes": 0.20,
}

# File Paths
DATASET_PATH = "data/dataset.csv"
MODEL_PATH = "models/phishing_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"