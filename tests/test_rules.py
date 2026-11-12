from src.rules import evaluate_rules

def test_urgency_rule():
    score, reasons = evaluate_rules("URGENT: Verify your account immediately")
    assert score > 0
    assert len(reasons) >= 2