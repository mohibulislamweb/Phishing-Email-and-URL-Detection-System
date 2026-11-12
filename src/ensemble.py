"""
ensemble.py
-----------
Advanced Hybrid ensemble prediction engine with confidence thresholds 
and calibrated weights to balance heuristic rules and ML probabilities.
"""

from typing import Dict

def compute_ensemble_score(rule_score: float, ml_scores: Dict[str, float]) -> float:
    """
    Combines heuristic rule scores and ML model probabilities with adaptive weighting
    to prevent false positives and erratic scoring.
    """
    # Optimized weights ensuring robust balance between rules and ML classifiers
    weights = {
        "rule_engine": 0.25,
        "random_forest": 0.30,
        "lightgbm": 0.30,
        "naive_bayes": 0.15,
    }
    
    # Base weighted accumulation
    total = rule_score * weights["rule_engine"]
    
    active_ml_weight = 0.0
    for key, score in ml_scores.items():
        w = weights.get(key, 0.20)
        total += score * w
        active_ml_weight += w

    # Normalize if some ML scores are missing
    total_weight = weights["rule_engine"] + active_ml_weight
    if total_weight > 0:
        normalized_total = total / (total_weight / 1.0) # Scale back if needed
    else:
        normalized_total = total

    # Safety checks / Confidence bounding (0 to 100)
    final_score = max(0.0, min(normalized_total, 100.0))
    
    # Strong override logic for extreme indicators to prevent false negatives
    if rule_score >= 85.0 and any(score >= 80.0 for score in ml_scores.values()):
        # If both rules and models agree it's highly suspicious, boost slightly
        final_score = max(final_score, 88.0)
    elif rule_score <= 10.0 and all(score <= 20.0 for score in ml_scores.values()):
        # If both agree it's completely safe, lower it
        final_score = min(final_score, 12.0)

    return round(final_score, 1)