"""
Hybrid ensemble prediction engine.
"""

from typing import Dict

def compute_ensemble_score(rule_score: float, ml_scores: Dict[str, float]) -> float:
    weights = {
        "rule_engine": 0.40,
        "random_forest": 0.20,
        "lightgbm": 0.20,
        "naive_bayes": 0.20,
    }
    
    total = rule_score * weights["rule_engine"]
    for key, score in ml_scores.items():
        total += score * weights.get(key, 0.20)
        
    return round(total, 1)