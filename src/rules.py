"""
Rule-based heuristic detection rules with Context-Aware Benign Suppression 
to prevent False Positives on safe emails/SMS.
"""

from typing import List, Tuple
import re

URGENCY_WORDS = ["urgent", "immediately", "action required", "verify your account", "suspended", "locked"]
CREDENTIAL_WORDS = ["password", "username", "ssn", "credit card", "cvv", "pin number"]

# Benign/Legitimate Context Patterns to prevent False Positives on safe notifications
BENIGN_PATTERNS = [
    r"if you made this change",
    r"no further action is needed",
    r"no action is required",
    r"weekly team meeting",
    r"order has shipped",
    r"subscription renewal was completed successfully",
    r"your appointment is confirmed",
    r"library books are due",
    r"monthly electricity statement",
    r"github security key was added",
]

def evaluate_rules(text: str) -> Tuple[float, List[str]]:
    lower = text.lower()
    score = 0.0
    reasons = []

    # 1. Benign Context Suppression (False Positive Mitigation)
    has_benign_context = any(re.search(pattern, lower) for pattern in BENIGN_PATTERNS)
    if has_benign_context:
        reasons.append("Verified legitimate transactional/notification context detected (Safe Override)")
        return 0.0, reasons

    # 2. Urgency checks
    for word in URGENCY_WORDS:
        if word in lower:
            score += 10
            reasons.append(f"Urgency keyword detected: '{word}'")

    # 3. Credential keyword checks
    for word in CREDENTIAL_WORDS:
        if word in lower:
            score += 15
            reasons.append(f"Credential request keyword: '{word}'")

    final_score = max(0.0, min(score, 100.0))
    return final_score, reasons