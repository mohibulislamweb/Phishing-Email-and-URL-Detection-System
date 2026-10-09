"""
url_analysis.py
----------------
Advanced & Dedicated URL / Link Threat-Analysis Module for PhishGuard AI.

Features:
- Lexical & Structural Risk Scoring
- Brand Impersonation & Typosquatting Analysis
- Shannon Entropy Calculation for Random Domain Detection
- IDN/Punycode Homograph Attack Detection
- Open Redirect Parameter Checks

Runs fully offline (standard library only) — no external API required.
"""

import math
import re
import ipaddress
import difflib
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

# High-risk Top-Level Domains commonly abused in phishing campaigns
SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".work", ".gq", ".tk", ".ml", ".cf",
    ".biz", ".info", ".click", ".loan", ".zip", ".country", ".icu",
    ".kim", ".buzz", ".monster", ".fit", ".surf", ".rest"
]

# Popular Link-Shortening Services
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "buff.ly", "v.gd", "clck.ru"
]

# High-Value Target Brands for Impersonation / Typosquatting
KNOWN_BRANDS = [
    "paypal", "microsoft", "apple", "amazon", "netflix", "google",
    "facebook", "instagram", "irs", "dhl", "fedex", "bankofamerica",
    "chase", "wellsfargo", "outlook", "dropbox", "linkedin", "whatsapp",
    "binance", "coinbase", "stripe", "steam", "adobe"
]

# Common Suspicious Query Parameters (Open Redirect Targets)
REDIRECT_PARAMS = ["redirect", "url", "next", "link", "goto", "out", "target", "dest"]


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
    """Check if the provided hostname is a raw IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _calculate_entropy(text: str) -> float:
    """Calculate Shannon Entropy of a string to measure randomness/complexity."""
    if not text:
        return 0.0
    entropy = 0.0
    text_length = len(text)
    for x in set(text):
        p_x = float(text.count(x)) / text_length
        entropy -= p_x * math.log2(p_x)
    return round(entropy, 2)


def _closest_brand(host: str) -> Tuple[Optional[str], float]:
    """
    Find closest matching brand using Levenshtein distance sequence matching.
    Checks leading domain labels, subdomains, and hyphenated tokens.
    """
    if not host:
        return None, 0.0
    
    # Extract main domain label before TLD
    labels = host.split(".")
    leading_label = labels[0] if len(labels) > 1 else host
    
    candidates = {leading_label}
    candidates.update(re.split(r"[-_0-9]+", leading_label))
    candidates.discard("")

    best_brand, best_ratio = None, 0.0
    for token in candidates:
        if len(token) < 3:
            continue
        for brand in KNOWN_BRANDS:
            ratio = difflib.SequenceMatcher(None, token, brand).ratio()
            if ratio > best_ratio:
                best_brand, best_ratio = brand, ratio

    return best_brand, best_ratio


def analyze_url(url: str) -> UrlVerdict:
    """
    Comprehensive offline analysis of a URL against multi-vector threat indicators.
    Returns UrlVerdict containing a risk score (0 to 100) and human-readable reasons.
    """
    reasons = []
    score = 0.0

    try:
        # Add http scheme if missing for accurate urlparse execution
        parsed = urlparse(url if "://" in url else f"http://{url}")
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        return UrlVerdict(url=url, score=40.0, reasons=["Malformed / unparsable URL structure"])

    if not host:
        return UrlVerdict(url=url, score=30.0, reasons=["Could not extract a valid host from URL"])

    # 1. Raw IP address usage
    if _is_ip_host(host):
        score += 25
        reasons.append("Uses a raw IP address instead of a standard domain name")

    # 2. Insecure Protocol Check
    if scheme == "http" and not _is_ip_host(host):
        score += 5
        reasons.append("Uses unencrypted HTTP protocol instead of HTTPS")

    # 3. URL-shortening service usage
    if any(host == s or host.endswith("." + s) for s in URL_SHORTENERS):
        score += 15
        reasons.append("Uses a link-shortening service (hides destination URL)")

    # 4. High-risk / low-cost top-level domain (TLD)
    if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
        tld_used = host.rsplit('.', 1)[-1]
        score += 15
        reasons.append(f"High-risk top-level domain (.{tld_used}) commonly associated with abuse")

    # 5. Excessive subdomain depth (Subdomain Sandwiching)
    subdomain_count = host.count(".")
    if subdomain_count >= 3:
        score += 12
        reasons.append(f"Unusually deep subdomain structure ({subdomain_count} dots)")

    # 6. '@' Symbol Userinfo Exploitation
    if "@" in url:
        score += 20
        reasons.append("Contains '@' symbol — browsers ignore preceding text; decoy host trick")

    # 7. IDN / Punycode Homograph Attack
    if host.startswith("xn--") or ".xn--" in host:
        score += 25
        reasons.append("Contains Punycode (xn--) — potential Homograph/international character spoofing")

    # 8. Excessive URL / Host Length & Hyphenation
    if len(url) > 75:
        score += 6
        reasons.append("Unusually long URL (>75 characters)")
    
    hyphen_count = host.count("-")
    if hyphen_count >= 2:
        score += 8
        reasons.append(f"Multiple hyphens in domain name ({hyphen_count} hyphens)")

    # 9. High Shannon Entropy (Random Character Detection)
    domain_label = host.split(".")[0]
    entropy = _calculate_entropy(domain_label)
    if entropy > 3.8 and len(domain_label) > 8:
        score += 15
        reasons.append(f"High domain randomness / entropy score ({entropy}) — likely generated algorithmically")

    # 10. Typosquatting and Brand Impersonation Checks
    brand, ratio = _closest_brand(host)
    if brand:
        if 0.75 <= ratio < 1.0 and brand not in host:
            score += 25
            reasons.append(f"Domain closely resembles '{brand}' (similarity: {int(ratio*100)}%) — possible typosquatting")
        elif brand in host and not (host.endswith(f"{brand}.com") or host.endswith(f"{brand}.org") or host.endswith(f"{brand}.net")):
            score += 18
            reasons.append(f"Brand name '{brand}' used in subdomain/path outside official brand domain")

    # 11. Sensitive/Credential Keywords in Path
    if re.search(r"(login|verify|secure|update|account|confirm|reset|signin|banking|wallet|webmail)", path.lower()):
        score += 8
        reasons.append("URL path contains sensitive credential/verification keywords")

    # 12. Open Redirect Parameter Identification
    if query:
        parsed_query = parse_qs(query.lower())
        for param in REDIRECT_PARAMS:
            if param in parsed_query:
                score += 10
                reasons.append(f"Contains potential Open Redirect parameter ('{param}=') in query string")
                break

    if not reasons:
        reasons.append("No structural or lexical red flags detected")

    return UrlVerdict(url=url, score=min(round(score, 1), 100.0), reasons=reasons)


def analyze_urls(urls: List[str]) -> List[UrlVerdict]:
    """Analyze a batch list of URLs extracted from an email."""
    return [analyze_url(u) for u in urls]


def aggregate_url_risk(verdicts: List[UrlVerdict]) -> float:
    """
    Combine individual URL verdicts into a unified risk contribution (0 to 25 points)
    for the overall email threat scoring engine. Driven primarily by the worst URL found.
    """
    if not verdicts:
        return 0.0
    worst_score = max(v.score for v in verdicts)
    return round(min(worst_score * 0.25, 25.0), 1)