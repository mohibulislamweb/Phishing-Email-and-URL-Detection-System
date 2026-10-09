from src.ensemble import compute_ensemble_score

def test_ensemble_calculation():
    rule_score = 50.0
    ml_scores = {"random_forest": 60.0, "lightgbm": 60.0, "naive_bayes": 40.0}
    final_score = compute_ensemble_score(rule_score, ml_scores)
    assert 0 <= final_score <= 100