"""
Rule-based heuristic detection rules.
"""

from typing import List, Tuple

URGENCY_WORDS = ["urgent", "immediately", "action required", "verify your account", "suspended", "locked"]
CREDENTIAL_WORDS = ["password", "username", "ssn", "credit card", "cvv", "pin number"]

def evaluate_rules(text: str) -> Tuple[float, List[str]]:
    lower = text.lower()
    score = 0.0
    reasons = []

    for word in URGENCY_WORDS:
        if word in lower:
            score += 10
            reasons.append(f"Urgency keyword detected: '{word}'")

    for word in CREDENTIAL_WORDS:
        if word in lower:
            score += 15
            reasons.append(f"Credential request keyword: '{word}'")

    return min(score, 100.0), reasons