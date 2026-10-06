"""
url_analysis.py
----------------
Dedicated URL / link threat-analysis module for PhishGuard.

Extracts a risk score and human-readable reasons for every URL found in
an email, using lexical, structural and brand-impersonation heuristics.
Runs fully offline (standard library only) so it needs no API key and
works even without internet access.

Import this module from app.py:

    from url_analysis import analyze_urls, aggregate_url_risk
"""

import re
import ipaddress
import difflib
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".work", ".gq", ".tk", ".ml", ".cf",
    ".biz", ".info", ".click", ".loan", ".zip", ".country",
]
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "buff.ly",
]
KNOWN_BRANDS = [
    "paypal", "microsoft", "apple", "amazon", "netflix", "google",
    "facebook", "instagram", "irs", "dhl", "fedex", "bankofamerica",
    "chase", "wellsfargo", "outlook", "dropbox", "linkedin", "whatsapp",
]


@dataclass
class UrlVerdict:
    url: str
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def tier(self) -> str:
        if self.score >= 60:
            return "high"
        if self.score >= 30:
            return "medium"
        return "low"


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _closest_brand(host: str):
    """Return (brand, similarity_ratio) for the most similar known brand.
    Used to flag typosquatted domains such as 'paypa1-secure-login.xyz'.
    Checks the full leading label plus each hyphen/digit-separated token,
    since attackers often bury the brand inside a longer domain."""
    if not host:
        return None, 0.0
    leading_label = host.split(".")[0]
    candidates = {leading_label}
    candidates.update(re.split(r"[-_0-9]+", leading_label))
    candidates.discard("")

    best_brand, best_ratio = None, 0.0
    for token in candidates:
        for brand in KNOWN_BRANDS:
            ratio = difflib.SequenceMatcher(None, token, brand).ratio()
            if ratio > best_ratio:
                best_brand, best_ratio = brand, ratio
    return best_brand, best_ratio


def analyze_url(url: str) -> UrlVerdict:
    """Score a single URL from 0 (clean) to 100 (highly suspicious)."""
    reasons = []
    score = 0.0

    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
    except Exception:
        return UrlVerdict(url=url, score=40.0, reasons=["Malformed / unparsable URL"])

    if not host:
        return UrlVerdict(url=url, score=30.0, reasons=["Could not extract a host from this URL"])

    # 1. Raw IP address instead of a domain name
    if _is_ip_host(host):
        score += 25
        reasons.append("Uses a raw IP address instead of a domain name")

    # 2. URL-shortening service (hides the real destination)
    if any(host == s or host.endswith("." + s) for s in URL_SHORTENERS):
        score += 15
        reasons.append("Uses a link-shortening service (real destination is hidden)")

    # 3. High-risk / low-cost top-level domain
    if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
        score += 12
        reasons.append(f"High-risk top-level domain (.{host.rsplit('.', 1)[-1]})")

    # 4. Excessive subdomain depth, e.g. paypal.com.verify-login.xyz
    subdomain_count = host.count(".")
    if subdomain_count >= 3:
        score += 10
        reasons.append(f"Unusually deep subdomain structure ({subdomain_count} dots)")

    # 5. '@' trick — browsers ignore everything before '@'
    if "@" in url:
        score += 20
        reasons.append("Contains '@' — everything before it is a decoy; the real host follows '@'")

    # 6. Excessive length / repeated hyphens (common brand-spoofing pattern)
    if len(url) > 75:
        score += 6
        reasons.append("Unusually long URL")
    if host.count("-") >= 2:
        score += 6
        reasons.append("Multiple hyphens in the domain name")

    # 7. Typosquatting — domain closely resembles a known brand but isn't it
    brand, ratio = _closest_brand(host)
    if brand and 0.72 <= ratio < 1.0 and brand not in host:
        score += 25
        reasons.append(f"Domain closely resembles '{brand}' but is not the real domain (possible typosquat)")
    elif brand and brand in host and not host.endswith(f"{brand}.com"):
        score += 15
        reasons.append(f"Brand name '{brand}' appears outside the actual domain (path/subdomain trick)")

    # 8. Credential/verification keywords in the path
    if re.search(r"(login|verify|secure|update|account|confirm|reset)", path.lower()):
        score += 5
        reasons.append("URL path contains credential/verification-related keywords")

    if not reasons:
        reasons.append("No structural red flags detected")

    return UrlVerdict(url=url, score=min(round(score, 1), 100.0), reasons=reasons)


def analyze_urls(urls: List[str]) -> List[UrlVerdict]:
    """Analyze a list of URLs found in one email."""
    return [analyze_url(u) for u in urls]


def aggregate_url_risk(verdicts: List[UrlVerdict]) -> float:
    """Combine multiple URL verdicts into one bounded contribution (0-25)
    for the overall email risk score. Driven by the single worst URL."""
    if not verdicts:
        return 0.0
    worst = max(v.score for v in verdicts)
    return round(min(worst * 0.25, 25.0), 1)