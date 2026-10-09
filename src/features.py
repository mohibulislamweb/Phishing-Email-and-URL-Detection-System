"""
Feature extraction module for machine learning classifiers.
"""

import re
from typing import Dict, Any

def extract_text_features(text: str) -> Dict[str, Any]:
    lower = text.lower()
    return {
        "length": len(text),
        "num_urls": len(re.findall(r"https?://", lower)),
        "num_urgency_words": sum(lower.count(w) for w in ["urgent", "verify", "immediately", "locked"]),
        "has_ip_in_url": int(bool(re.search(r"https?://\d{1,3}\.\d{1,3}\.", lower))),
        "caps_count": len(re.findall(r"\b[A-Z]{4,}\b", text)),
    }