"""
Text cleaning and extraction helpers for email contents.
"""

import re
from typing import List

def extract_urls(text: str) -> List[str]:
    links = re.findall(r"https?://[^\s)>\]]+", text or "")
    return list(dict.fromkeys(links))

def extract_ips(text: str) -> List[str]:
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text or "")
    return list(dict.fromkeys(ips))