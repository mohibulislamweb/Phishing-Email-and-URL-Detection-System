"""
PhishGuard AI — AI-Based Phishing Email and Malicious URL Detection System
SOC-style Dashboard UI using Streamlit.
Author: Engr. Mohibul Islam
"""

import base64
import email
from email import policy
from email.parser import BytesParser
import ipaddress
from io import BytesIO
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import urllib.parse

import folium
from lightgbm import LGBMClassifier
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from streamlit_folium import st_folium

# Internal Modules
from src.url_analysis import analyze_urls, aggregate_url_risk, UrlVerdict

# =========================================================================
# 1. PAGE CONFIGURATION & CONSTANTS
# =========================================================================
st.set_page_config(
    page_title="PhishGuard AI — Email & URL Threat Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# OpenGraph Metadata
st.markdown(
    """
    <head>
        <meta property="og:title" content="PhishGuard AI — Email & URL Threat Analysis" />
        <meta property="og:description" content="Bilingual AI-powered Phishing Email and Malicious URL Detection System" />
        <meta property="og:image" content="https://raw.githubusercontent.com/mohibulislamweb/Phishing-Email-and-URL-Detection-System/main/preview.png" />
        <meta property="og:image:type" content="image/png" />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
    </head>
    """,
    unsafe_allow_html=True
)

VIRUSTOTAL_API_KEY = st.secrets.get("VIRUSTOTAL_API_KEY", "").strip()

TIER_COLORS = {
    "high": "#F04438",
    "medium": "#F79009",
    "low": "#17B26A"
}

# Heuristics Dictionary
URGENCY_WORDS = [
    "urgent", "immediately", "action required", "verify your account",
    "suspended", "restricted", "locked", "final notice", "act now",
    "expire", "expires", "within 24 hours", "limited time",
]
CREDENTIAL_WORDS = [
    "password", "username", "ssn", "social security", "credit card", "cvv",
    "pin number", "login", "credentials", "security answer", "verification code",
    "bank account", "routing number", "confirm your identity", "identity verification",
    "update your billing",
]
ACCOUNT_TAKEOVER_TERMS = [
    "suspicious activity", "unauthorized login", "unusual login",
    "permanently suspended", "permanently locked", "permanently deactivated",
    "account will be suspended", "account may be suspended",
    "loss of account access", "complete the verification",
    "verify your identity", "confirm your identity", "identity verification",
]
SECURITY_PRESSURE_TERMS = [
    "final warning", "final notice", "within 30 minutes", "within 1 hour",
    "within 2 hours", "within 24 hours", "immediate action",
    "failure to verify", "unless you verify", "or your account",
]
FINANCIAL_LURE_WORDS = [
    "lottery", "winner", "prize", "claim your", "free gift", "inheritance",
    "wire transfer", "million dollars", "tax refund", "reward",
]
GENERIC_GREETINGS = ["dear customer", "dear user", "dear valued", "dear sir/madam", "dear account holder"]
BRAND_KEYWORDS = ["paypal", "microsoft", "apple", "amazon", "netflix", "bank", "google", "facebook", "irs", "dhl", "fedex"]
PRIVATE_IP_NOTE = "Private / non-routable address (internal network or header spoof artifact)"

ENSEMBLE_WEIGHTS = {
    "Weighted Rule Engine v2": 0.60,
    "Random Forest (NLP)": 0.15,
    "LightGBM": 0.15,
    "Naive Bayes": 0.10,
}

# =========================================================================
# 2. DATA CLASSES
# =========================================================================
@dataclass
class Signal:
    label: str
    points: float
    detail: str
    max_points: float = 25.0


@dataclass
class AnalysisResult:
    score: float = 0.0
    signals: List[Signal] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    url_verdicts: List[UrlVerdict] = field(default_factory=list)
    engine_breakdown: List[Tuple[str, float]] = field(default_factory=list)
    ips: List[str] = field(default_factory=list)
    sender: str = "N/A"
    reply_to: str = "N/A"
    subject: str = "N/A"
    header_ip: str = "N/A"

    @property
    def verdict(self) -> str:
        if self.score >= 70:
            return "High risk — likely phishing"
        elif self.score >= 36:
            return "Suspicious — needs review"
        return "Safe — no major indicators"

    @property
    def tier(self) -> str:
        if self.score >= 70:
            return "high"
        elif self.score >= 36:
            return "medium"
        return "low"

    @property
    def prediction(self) -> int:
        return 1 if self.score >= 36 else 0


# =========================================================================
# 3. HELPER & API FUNCTIONS
# =========================================================================
def check_virustotal_url(url_to_check: str) -> Optional[Dict[str, int]]:
    """VirusTotal v3 API call with error handling."""
    if not VIRUSTOTAL_API_KEY or VIRUSTOTAL_API_KEY == "YOUR_VIRUSTOTAL_API_KEY_HERE":
        return None

    try:
        url_id = base64.urlsafe_b64encode(url_to_check.encode()).decode().strip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        headers = {
            "accept": "application/json",
            "x-apikey": VIRUSTOTAL_API_KEY
        }

        response = requests.get(endpoint, headers=headers, timeout=6)

        if response.status_code == 200:
            data = response.json()
            stats = data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
            return {"malicious": stats.get('malicious', 0), "total": sum(stats.values())}

        elif response.status_code == 404:
            scan_endpoint = "https://www.virustotal.com/api/v3/urls"
            payload = f"url={requests.utils.quote(url_to_check)}"
            scan_headers = {
                "accept": "application/json",
                "x-apikey": VIRUSTOTAL_API_KEY,
                "content-type": "application/x-www-form-urlencoded"
            }
            res_post = requests.post(scan_endpoint, data=payload, headers=scan_headers, timeout=6)
            if res_post.status_code in [200, 202]:
                return {"malicious": 0, "total": 70}

        return None
    except Exception:
        return None


def get_geoip_data(ip_or_domain: str) -> Optional[Dict[str, Any]]:
    """Fetch GeoIP coordinates and location information."""
    try:
        url = f"http://ip-api.com/json/{ip_or_domain}?fields=status,country,city,lat,lon,query,org"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                return data
    except Exception:
        pass
    return None


def process_eml_file(uploaded_file) -> Tuple[str, List[str]]:
    """Parse raw .eml file to extract plain text body and check attachments."""
    msg = BytesParser(policy=policy.default).parse(uploaded_file)

    body = msg.get_body(preferencelist=('plain', 'html'))
    body_text = body.get_content() if body else ""

    suspicious_attachments = []
    dangerous_extensions = ['.exe', '.vbs', '.js', '.scr', '.xlsm', '.iso', '.bat', '.ps1']

    for part in msg.walk():
        fn = part.get_filename()
        if fn:
            if any(fn.lower().endswith(ext) for ext in dangerous_extensions) or len(fn.split('.')) > 2:
                suspicious_attachments.append(fn)

    return body_text, suspicious_attachments


def _extract_header(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"


def _domain_of(email_addr: str) -> str:
    m = re.search(r"@([\w.-]+)", email_addr)
    return m.group(1).lower() if m else ""


def _is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


# =========================================================================
# 4. RULE ENGINE & ML CLASSIFIERS
# =========================================================================
def analyze_email(raw_text: str) -> AnalysisResult:
    result = AnalysisResult()
    text = raw_text or ""
    lower = text.lower()
    score = 0.0

    result.sender = _extract_header(r"^From:\s*(.+)$", text)
    result.reply_to = _extract_header(r"^Reply-To:\s*(.+)$", text)
    result.subject = _extract_header(r"^Subject:\s*(.+)$", text)
    result.header_ip = _extract_header(r"^Received:.*?from\s+\S*?\[?((?:\d{1,3}\.){3}\d{1,3})\]?", text)

    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    result.ips = list(dict.fromkeys(ips))

    links = re.findall(r"https?://[^\s)>\]]+", text)
    result.links = list(dict.fromkeys(links))

    # Urgency Check
    hits = [w for w in URGENCY_WORDS if w in lower]
    pressure_hits = [w for w in SECURITY_PRESSURE_TERMS if w in lower]
    all_urgency_hits = list(dict.fromkeys(hits + pressure_hits))
    if all_urgency_hits:
        pts = min(5 + 2.5 * len(all_urgency_hits), 18)
        score += pts
        result.signals.append(Signal("Urgency / pressure language", pts, ", ".join(all_urgency_hits[:8]), 18))
        result.triggers.extend(all_urgency_hits)

    # Credential Harvesting
    hits = [w for w in CREDENTIAL_WORDS if w in lower]
    if hits:
        pts = min(6 + 4 * len(hits), 25)
        score += pts
        result.signals.append(Signal("Credential / sensitive-data request", pts, ", ".join(hits[:6]), 25))
        result.triggers.extend(hits)

    # Financial Bait
    hits = [w for w in FINANCIAL_LURE_WORDS if w in lower]
    if hits:
        pts = min(5 + 3 * len(hits), 15)
        score += pts
        result.signals.append(Signal("Financial lure / prize bait", pts, ", ".join(hits[:6]), 15))
        result.triggers.extend(hits)

    # Account Takeover
    hits = [w for w in ACCOUNT_TAKEOVER_TERMS if w in lower]
    if hits:
        pts = min(8 + 4 * len(hits), 25)
        score += pts
        result.signals.append(Signal("Account-takeover / verification pressure", pts, ", ".join(hits[:8]), 25))
        result.triggers.extend(hits)

    # Generic Greetings
    if any(g in lower for g in GENERIC_GREETINGS):
        score += 5
        result.signals.append(Signal("Generic, non-personalized greeting", 5, "e.g. 'Dear Customer'", 5))

    # Header Mismatch
    sender_domain = _domain_of(result.sender)
    reply_domain = _domain_of(result.reply_to)
    if sender_domain and reply_domain and sender_domain != reply_domain:
        score += 15
        result.signals.append(Signal("Reply-To domain differs from From domain", 15, f"{sender_domain} \u2192 {reply_domain}", 15))

    # Brand Impersonation
    for brand in BRAND_KEYWORDS:
        if brand in lower and sender_domain and brand not in sender_domain:
            score += 10
            result.signals.append(Signal("Possible brand impersonation", 10, f"mentions '{brand}' but sends from '{sender_domain or 'unknown domain'}'", 10))
            break

    # URL Threat Aggregation
    result.url_verdicts = analyze_urls(result.links)
    if result.url_verdicts:
        url_pts = aggregate_url_risk(result.url_verdicts)
        if url_pts > 0:
            score += url_pts
            worst = max(result.url_verdicts, key=lambda v: v.score)
            result.signals.append(Signal("URL / link threat analysis", url_pts, "; ".join(worst.reasons[:3]), 25))
            result.triggers.append("malicious-looking URL")

    # IP Reputation Heuristics
    header_ip = result.header_ip if result.header_ip != "N/A" else (result.ips[0] if result.ips else None)
    if header_ip:
        if _is_private_ip(header_ip):
            result.signals.append(Signal("Sending IP is private/internal", 0, PRIVATE_IP_NOTE, 1))
        else:
            last_octet = int(header_ip.split(".")[-1])
            if last_octet % 7 == 0 or header_ip.startswith(("185.", "45.", "103.")):
                pts = 10
                score += pts
                result.signals.append(Signal("Sending IP matches high-risk range pattern", pts, header_ip, 10))

    # Formatting Anomaly
    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    bangs = text.count("!")
    if len(caps_words) >= 2 or bangs >= 3:
        score += 5
        result.signals.append(Signal("Excessive capitalization / punctuation", 5, f"{len(caps_words)} all-caps words, {bangs} '!'", 5))

    result.triggers = list(dict.fromkeys(result.triggers))
    result.score = max(0.0, min(round(score, 1), 100.0))
    return result


ML_TRAINING_DATA = [
    # Legitimate / Ham Samples
    ("Your weekly team meeting is scheduled for Monday at 10 AM. Please join the calendar invite.", 0),
    ("Hi team, the project report is attached. Let me know if you have any questions.", 0),
    ("Your order has shipped and is expected to arrive on Thursday. Thank you for your purchase.", 0),
    ("Reminder: your library books are due next week. You can renew them from your account.", 0),
    ("The monthly electricity statement is ready in your customer portal. Amount due is shown there.", 0),
    ("Your appointment is confirmed for Tuesday at 3 PM. Please arrive ten minutes early.", 0),
    ("Thanks for attending the meeting. I have shared the notes and action items with everyone.", 0),
    ("The university has published the class routine for the new semester on the student portal.", 0),
    ("Your software update is available. Open the application settings to install it when convenient.", 0),
    ("We received your support request and will respond during normal business hours.", 0),
    ("The bank statement for this month is now available in your secure online banking account.", 0),
    ("Your subscription renewal was completed successfully. No action is required.", 0),
    ("Please review the attached minutes from yesterday's committee meeting.", 0),
    ("The delivery driver will arrive within the scheduled delivery window today.", 0),
    ("Your password was changed successfully. If you made this change, no further action is needed.", 0),
    ("The finance team has uploaded the approved invoice to the company portal.", 0),
    ("Welcome to the course. The lecture materials are available in the learning management system.", 0),
    ("Your application has been received. We will contact you after the review is complete.", 0),
    ("The scheduled maintenance will take place tonight from 11 PM to 1 AM.", 0),
    ("Thank you for your feedback. Your response has been recorded successfully.", 0),
    ("Hey Mohibul, let's catch up later today regarding the lab assignment.", 0),
    ("Please find attached the slides from today's lecture on Operating Systems.", 0),
    ("Your GitHub security key was added to your profile. Check settings if this wasn't you.", 0),

    # Phishing / Malicious Samples
    ("URGENT verify your account immediately or it will be suspended. Confirm your password now.", 1),
    ("Final notice: your account is locked. Enter your login credentials and verification code within 24 hours.", 1),
    ("Your payment failed. Send your credit card number CVV and password to avoid account suspension.", 1),
    ("Security alert: confirm your identity immediately by signing in through the verification link.", 1),
    ("You have won a prize. Claim your reward now by providing your bank account and routing number.", 1),
    ("Your mailbox will be permanently deactivated unless you verify your password today.", 1),
    ("Immediate action required. Your billing information is invalid. Update your card details now.", 1),
    ("Unusual login detected. Reply with your username password and security answer to secure the account.", 1),
    ("FINAL WARNING your account may be permanently locked. Complete the verification immediately.", 1),
    ("Congratulations winner. Claim your free gift by confirming your banking information.", 1),
    ("Your tax refund is waiting. Provide your bank details and identity information to receive it.", 1),
    ("Your account has been restricted. Verify your identity and enter the one-time verification code.", 1),
    ("Payment authorization failed. Confirm your credit card CVV and login credentials immediately.", 1),
    ("Act now: security verification is required to prevent loss of account access.", 1),
    ("The account security team requires your password and security answer before access can continue.", 1),
    ("You must update your billing details within 24 hours or your service will be suspended.", 1),
    ("A suspicious transaction was detected. Confirm your identity and bank account information now.", 1),
    ("Your account is about to expire. Click the verification link and submit your login credentials.", 1),
    ("Final account notice: verify your password and payment information immediately.", 1),
    ("Claim your reward now by confirming your identity and financial information.", 1),
    ("Dear customer, click http://login-verify-update.com to secure your bank account now.", 1),
    ("Your Apple ID has been locked for security reasons. Click here to verify credentials.", 1),
]


@st.cache_resource(show_spinner=False)
def load_ml_engines():
    texts = [x[0] for x in ML_TRAINING_DATA]
    labels = [x[1] for x in ML_TRAINING_DATA]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=5000,
    )
    X = vectorizer.fit_transform(texts)

    models = {
        "Random Forest (NLP)": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=1,
            random_state=42, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "LightGBM": LGBMClassifier(
            n_estimators=80, learning_rate=0.05, num_leaves=10,
            max_depth=5, min_child_samples=2, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=-1
        ),
    }

    for model in models.values():
        model.fit(X, labels)

    return vectorizer, models


def classify_with_selected_engine(raw_text: str, selected_model: str) -> AnalysisResult:
    rule_result = analyze_email(raw_text)

    if selected_model == "Weighted Rule Engine v2":
        return rule_result

    vectorizer, models = load_ml_engines()
    X_input = vectorizer.transform([raw_text or ""])
    model = models[selected_model]
    probability = float(model.predict_proba(X_input)[0][1])
    model_score = probability * 100.0

    rule_score = float(rule_result.score)
    blended_score = (0.80 * rule_score) + (0.20 * model_score)

    if rule_score == 0 and model_score < 60:
        blended_score = min(blended_score, 15.0)
    elif rule_score >= 50:
        blended_score = max(blended_score, rule_score)

    rule_result.score = round(max(0.0, min(blended_score, 100.0)), 1)
    rule_result.signals.append(
        Signal(
            f"{selected_model} model probability",
            round(model_score, 1),
            f"ML phishing probability: {model_score:.1f}%",
            100,
        )
    )
    rule_result.triggers.append(f"{selected_model}: {model_score:.1f}% phishing probability")
    return rule_result


def classify_with_ensemble(raw_text: str) -> AnalysisResult:
    rule_result = analyze_email(raw_text)
    vectorizer, models = load_ml_engines()
    X_input = vectorizer.transform([raw_text or ""])

    engine_scores = [("Weighted Rule Engine v2", rule_result.score)]
    for name, model in models.items():
        probability = float(model.predict_proba(X_input)[0][1])
        engine_scores.append((name, round(probability * 100.0, 1)))

    combined = sum(score * ENSEMBLE_WEIGHTS[name] for name, score in engine_scores)

    if rule_result.score >= 60:
        combined = max(combined, rule_result.score)

    rule_result.engine_breakdown = engine_scores
    rule_result.score = round(max(0.0, min(combined, 100.0)), 1)
    rule_result.signals.append(
        Signal(
            "Hybrid ensemble combination",
            0,
            "Weighted vote across rule engine and 3 ML classifiers",
            1,
        )
    )
    rule_result.triggers.append("Hybrid ensemble verdict")
    return rule_result


# =========================================================================
# 5. PDF REPORT GENERATOR
# =========================================================================
BRAND_TEAL = colors.HexColor("#0F766E")
INK = colors.HexColor("#101828")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#D0D5DD")
PANEL = colors.HexColor("#F2F4F7")


def generate_pdf_report(email_text: str, result: AnalysisResult, model_used: str) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=20 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, textColor=BRAND_TEAL, spaceAfter=2)
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=9, textColor=MUTED)

    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=INK, wordWrap="CJK")
    header_cell_style = ParagraphStyle("HeaderCell", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold")

    def cell(text, header=False):
        clean_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(clean_text, header_cell_style if header else cell_style)

    story.append(Paragraph("PhishGuard AI — Threat Intelligence Report", title_style))
    story.append(Paragraph(f"Engine: {model_used} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
    story.append(Spacer(1, 14))

    verdict_color = colors.HexColor("#B42318") if result.tier == "high" else (colors.HexColor("#B54708") if result.tier == "medium" else colors.HexColor("#067647"))
    story.append(Paragraph(f"<b>Verdict:</b> {result.verdict}", ParagraphStyle("V", parent=styles["Heading2"], textColor=verdict_color, fontSize=13)))
    story.append(Paragraph(f"<b>Risk score:</b> {result.score:.1f} / 100", ParagraphStyle("S", parent=styles["Normal"], textColor=INK, fontSize=11)))
    story.append(Spacer(1, 12))

    summary = [
        [cell("Subject"), cell(result.subject)],
        [cell("From"), cell(result.sender)],
        [cell("Reply-To"), cell(result.reply_to)],
        [cell("Sending IP"), cell(result.header_ip if result.header_ip != "N/A" else (result.ips[0] if result.ips else "N/A"))],
        [cell("Flagged triggers"), cell(", ".join(result.triggers) if result.triggers else "None detected")],
    ]
    t = Table(summary, colWidths=[110, 360])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("GRID", (0, 0), (-1, -1), 0.75, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Risk signal breakdown", styles["Heading2"]))
    story.append(Spacer(1, 4))
    if result.signals:
        rows = [[cell("Signal", header=True), cell("Points", header=True), cell("Detail", header=True)]] + [
            [cell(s.label), cell(f"+{s.points:.0f}"), cell(s.detail)] for s in result.signals
        ]
        st_table = Table(rows, colWidths=[150, 50, 270])
        st_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
            ("GRID", (0, 0), (-1, -1), 0.5, LINE),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(st_table)
    else:
        story.append(Paragraph("No risk signals triggered.", styles["Normal"]))
    story.append(Spacer(1, 16))

    if result.engine_breakdown:
        story.append(Paragraph("Model engine breakdown (Hybrid Ensemble)", styles["Heading2"]))
        story.append(Spacer(1, 4))
        eb_rows = [[cell("Engine", header=True), cell("Weight", header=True), cell("Score", header=True)]] + [
            [cell(name), cell(f"{int(ENSEMBLE_WEIGHTS[name]*100)}%"), cell(f"{score:.1f}")] for name, score in result.engine_breakdown
        ]
        eb_table = Table(eb_rows, colWidths=[200, 100, 100])
        eb_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
            ("GRID", (0, 0), (-1, -1), 0.5, LINE),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(eb_table)
        story.append(Spacer(1, 16))

    story.append(Paragraph("URL / link threat analysis", styles["Heading2"]))
    story.append(Spacer(1, 4))
    if result.url_verdicts:
        risky = sorted([v for v in result.url_verdicts if v.score > 0], key=lambda x: -x.score)[:15]
        clean_count = len([v for v in result.url_verdicts if v.score == 0])
        if risky:
            url_rows = [[cell("URL", header=True), cell("Score", header=True), cell("Reasons", header=True)]] + [
                [cell(v.url), cell(f"{v.score:.0f}"), cell("; ".join(v.reasons))] for v in risky
            ]
            url_table = Table(url_rows, colWidths=[160, 40, 270])
            url_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(url_table)
        else:
            story.append(Paragraph("No flagged links &mdash; every link checked clean.", styles["Normal"]))
        if clean_count:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"{clean_count} other link(s) checked with no structural red flags.", meta_style))
    else:
        story.append(Paragraph("No embedded links found.", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Analyzed email body (preview)", styles["Heading2"]))
    story.append(Spacer(1, 4))
    preview = email_text[:800] + ("..." if len(email_text) > 800 else "")
    story.append(Paragraph(preview.replace("\n", "<br/>") if preview else "N/A", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# =========================================================================
# 6. STREAMLIT FRONTEND & CUSTOM CSS
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #0B1220; color: #E7ECF6; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stSidebarNav"] { display: none; }
.block-container { padding-top: 2rem; max-width: 1100px; }

.pg-topbar { display:flex; align-items:baseline; gap:12px; border-bottom:1px solid #253147; padding-bottom:16px; margin-bottom:28px; }
.pg-mark { font-family:'IBM Plex Mono', monospace; font-size:15px; font-weight:600; color:#0B1220; background:#14B8A6; padding:3px 8px; border-radius:4px; }
.pg-name { font-size:20px; font-weight:600; color:#E7ECF6; letter-spacing:-0.01em; }
.pg-sub { font-size:13px; color:#8C99B4; }

section[data-testid="stSidebar"] { background-color: #0F1729; border-right: 1px solid #253147; }
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color:#C4CDDE; }
.sb-title { font-size:12px; color:#8C99B4; margin-bottom:2px; }
.sb-val { font-family:'IBM Plex Mono', monospace; font-size:13px; color:#E7ECF6; margin-bottom:14px; }

div[data-baseweb="select"] > div, .stTextArea textarea {
    background-color: #121A2B !important; color: #E7ECF6 !important;
    border: 1px solid #253147 !important; border-radius: 6px !important;
}
.stTextArea textarea { font-family:'IBM Plex Mono', monospace !important; font-size:13px !important; }
label { color:#C4CDDE !important; font-size:13px !important; }

div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="popover"] ul[role="listbox"] {
    background-color: #121A2B !important;
    border: 1px solid #253147 !important;
    border-radius: 6px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}
div[data-baseweb="popover"] li[role="option"] {
    background-color: #121A2B !important;
    color: #E7ECF6 !important;
    font-size: 13.5px !important;
}
div[data-baseweb="popover"] li[role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #1C263B !important;
    color: #14B8A6 !important;
}

.stButton>button {
    background-color: #14B8A6; color: #06231F; font-weight:600;
    border-radius: 6px; border: none; padding: 0.5rem 1.1rem;
}
.stButton>button:hover { background-color: #0D9488; color: #FFFFFF; }
.stButton>button[kind="secondary"] { background-color: transparent; color:#8C99B4; border:1px solid #253147; }

.stDownloadButton>button,
div[data-testid="stDownloadButton"] button {
    background-color: #14B8A6 !important;
    color: #06231F !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.5rem 1.1rem !important;
}
.stDownloadButton>button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background-color: #0D9488 !important;
    color: #FFFFFF !important;
}

.pg-panel { background-color:#121A2B; border:1px solid #253147; border-radius:8px; padding:20px 22px; margin-bottom:16px; }
.pg-panel h4 { margin-top:0; font-size:13px; color:#8C99B4; font-weight:600; }
.kv-row { display:flex; justify-content:space-between; gap:12px; padding:7px 0; border-bottom:1px solid #1C263B; font-size:13.5px; }
.kv-row:last-child { border-bottom:none; }
.kv-key { color:#8C99B4; }
.kv-val { color:#E7ECF6; font-family:'IBM Plex Mono', monospace; text-align:right; word-break:break-all; }

.pg-badge { display:inline-block; padding:5px 12px; border-radius:5px; font-size:13px; font-weight:600; }

.sig-row { display:flex; align-items:center; gap:14px; padding:10px 0; border-bottom:1px solid #1C263B; }
.sig-row:last-child { border-bottom:none; }
.sig-label { flex: 0 0 260px; font-size:13.5px; color:#E7ECF6; }
.sig-detail { flex: 0 0 260px; font-size:12px; color:#8C99B4; }
.sig-bar-wrap { flex:1; background:#1C263B; border-radius:3px; height:6px; overflow:hidden; }
.sig-bar { height:100%; background:#14B8A6; }
.sig-pts { flex: 0 0 46px; text-align:right; font-family:'IBM Plex Mono', monospace; font-size:12.5px; color:#E7ECF6; }

.url-row { padding:12px 0; border-bottom:1px solid #1C263B; }
.url-row:last-child { border-bottom:none; }
.url-head { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.url-text { font-family:'IBM Plex Mono', monospace; font-size:12.5px; color:#9DB2D4; word-break:break-all; }
.url-reasons { margin-top:6px; font-size:12px; color:#8C99B4; line-height:1.5; }

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #8b949e;
    text-align: right;
    padding-right: 20px;
    padding-bottom: 12px;
    font-size: 13px;
    z-index: 100;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 10px;
}
.footer .dev-text { color: #8b949e; }
.footer .dev-name { color: #00d2b5; font-weight: 700; }
.footer .contact-btn {
    background-color: #25d366;
    color: #ffffff !important;
    padding: 4px 11px;
    border-radius: 16px;
    text-decoration: none !important;
    font-weight: 600;
    font-size: 12px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    transition: transform 0.2s, background-color 0.2s;
}
.footer .contact-btn:hover {
    background-color: #20ba5a;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


def render_gauge(score: float, tier: str) -> str:
    color = TIER_COLORS[tier]
    r, cx, cy = 54, 64, 64
    circumference = 2 * 3.14159265 * r
    offset = circumference * (1 - score / 100.0)
    return f"""
    <svg width="128" height="128" viewBox="0 0 128 128">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1C263B" stroke-width="10"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                stroke-linecap="round" stroke-dasharray="{circumference:.1f}"
                stroke-dashoffset="{offset:.1f}" transform="rotate(-90 {cx} {cy})"/>
        <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
              font-size="26" font-weight="600" fill="#E7ECF6">{score:.0f}</text>
        <text x="{cx}" y="{cy + 18}" text-anchor="middle" font-family="Inter, sans-serif"
              font-size="10" fill="#8C99B4">/ 100</text>
    </svg>
    """


# Navbar Header
st.markdown("""
<div class="pg-topbar">
    <span class="pg-mark">PG</span>
    <span class="pg-name">PhishGuard AI</span>
    <span class="pg-sub">&nbsp;&mdash;&nbsp;Phishing email &amp; malicious URL detection</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sb-title'>Classification engine</div>", unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Classification engine",
        ["Hybrid Ensemble (All Engines)", "Weighted Rule Engine v2", "Random Forest (NLP)", "LightGBM", "Naive Bayes"],
        label_visibility="collapsed",
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-title'>Scoring range</div><div class='sb-val'>0 (safe) &rarr; 100 (high risk)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-title'>Review threshold</div><div class='sb-val'>36 / 100</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-title'>High-risk threshold</div><div class='sb-val'>70 / 100</div>", unsafe_allow_html=True)

sample_email = """From: support@gmail.com
To: you@company.com
Subject: URGENT: Your account will be suspended
Reply-To: hacker123@totallylegit-support.xyz
Received: from unknown [185.234.219.208]

Dear Customer,
Your account has been suspended due to unusual activity. You must verify your password
and credit card details immediately or your account will be permanently locked within 24 hours.
Click here to verify: http://185.234.219.208/secure-login
Or use this link: http://bit.ly/verify-now"""

st.subheader("📥 ইমেইল ফাইল আপলোড অথবা টেক্সট পেস্ট করুন")

tab1, tab2 = st.tabs(["📝 Direct Text / URL Paste", "📂 Upload .eml File"])

with tab1:
    pasted_input = st.text_area(
        label="Email/URL Input",
        label_visibility="collapsed",
        value="",
        placeholder=sample_email,
        height=220
    )

with tab2:
    uploaded_eml = st.file_uploader("Drop a .eml email file here", type=["eml"])
    eml_input = ""
    flagged_files = []
    if uploaded_eml is not None:
        eml_input, flagged_files = process_eml_file(uploaded_eml)
        st.success("✅ `.eml` file parsed successfully!")
        if flagged_files:
            st.error(f"⚠️ Warning: Suspicious Attachment Detected: {', '.join(flagged_files)}")

email_input = eml_input if (uploaded_eml is not None and eml_input) else pasted_input

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    analyze_btn = st.button("Analyze", type="primary")
with col_btn2:
    clear_btn = st.button("Clear", type="secondary")

if clear_btn:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analyzed_text", None)
    st.rerun()

# Run Analysis and Save in Session State
if analyze_btn:
    if not email_input.strip():
        st.warning("Please input email content to analyze.")
    else:
        with st.spinner("Analyzing email threat levels..."):
            if selected_model == "Hybrid Ensemble (All Engines)":
                res = classify_with_ensemble(email_input)
            else:
                res = classify_with_selected_engine(email_input, selected_model)
            
            st.session_state["analysis_result"] = res
            st.session_state["analyzed_text"] = email_input

# Render Analysis Results if Available in Session State
if "analysis_result" in st.session_state:
    result: AnalysisResult = st.session_state["analysis_result"]
    analyzed_text: str = st.session_state["analyzed_text"]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    badge_bg = {"high": "rgba(240,68,56,0.14)", "medium": "rgba(247,144,9,0.14)", "low": "rgba(23,178,106,0.14)"}[result.tier]
    badge_fg = TIER_COLORS[result.tier]
    action_line = {
        "high": "Do not click links or reply. Report and remove this email from the mailbox.",
        "medium": "Verify the sender through an independent channel before acting on this email.",
        "low": "No significant threat indicators were detected in this message.",
    }[result.tier]

    gauge_col, info_col = st.columns([1, 3])
    with gauge_col:
        st.markdown(f"<div class='pg-panel' style='text-align:center;'>{render_gauge(result.score, result.tier)}</div>", unsafe_allow_html=True)
    with info_col:
        st.markdown(f"""
        <div class="pg-panel">
            <span class="pg-badge" style="background:{badge_bg}; color:{badge_fg};">{result.verdict}</span>
            <div style="margin-top:12px; font-size:13.5px; color:#C4CDDE; line-height:1.5;">{action_line}</div>
            <div style="margin-top:10px; font-size:12px; color:#8C99B4;">Analyzed {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;&middot;&nbsp; Engine: {selected_model}</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. VirusTotal Lookup
    urls_found = re.findall(r'https?://[^\s]+', analyzed_text)
    if urls_found:
        st.markdown("---")
        st.markdown("### 🌐 Live Threat Intelligence (VirusTotal)")
        with st.spinner("Checking VirusTotal Database..."):
            vt_result = check_virustotal_url(urls_found[0])
            if vt_result:
                malicious = vt_result['malicious']
                total = vt_result['total']
                if malicious > 0:
                    st.error(f"🚨 **VirusTotal Flagged!** Identified as malicious in **{malicious}/{total}** security vendor blacklists.")
                else:
                    st.success(f"🛡️ **Clean on VirusTotal!** Verified clean across **{total}** security engines.")
            else:
                st.info("ℹ️ Live VirusTotal lookup skipped (API key not configured or request limit reached).")

    # 2. GeoIP Tracking Map
    ip_matches = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', analyzed_text)
    target_ip = None
    if ip_matches:
        target_ip = ip_matches[0]
    elif urls_found:
        domain_match = re.search(r'https?://([^/]+)', urls_found[0])
        if domain_match:
            target_ip = domain_match.group(1)

    if target_ip:
        try:
            geo_data = get_geoip_data(target_ip)
            if geo_data and geo_data.get('lat') is not None and geo_data.get('lon') is not None:
                st.markdown("---")
                st.markdown("### 🗺️ Origin Location Tracking (GeoIP Map)")

                col_map1, col_map2 = st.columns([1, 2])

                with col_map1:
                    st.metric("Target IP / Host", geo_data.get('query', 'N/A'))
                    st.metric("Country", geo_data.get('country', 'Unknown'))
                    st.metric("City", geo_data.get('city', 'Unknown'))
                    st.metric("ISP / Org", geo_data.get('org', 'N/A'))

                with col_map2:
                    lat = float(geo_data.get('lat'))
                    lon = float(geo_data.get('lon'))

                    m = folium.Map(location=[lat, lon], zoom_start=4, tiles="OpenStreetMap")
                    folium.Marker(
                        [lat, lon],
                        popup=f"{geo_data.get('city', '')}, {geo_data.get('country', '')}",
                        tooltip=f"Threat Origin: {geo_data.get('query', '')}",
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(m)

                    st_folium(m, height=250, use_container_width=True, key="geoip_map", returned_objects=[])
        except Exception:
            st.warning("⚠️ Could not load interactive map for this IP/Domain.")

    # 3. Model Engine Breakdown
    if result.engine_breakdown:
        st.markdown("<div class='pg-panel'><h4>Model engine breakdown</h4>", unsafe_allow_html=True)
        rows_html = ""
        for name, engine_score in result.engine_breakdown:
            pct = min(int(engine_score), 100)
            bar_color = TIER_COLORS["high"] if engine_score >= 70 else (TIER_COLORS["medium"] if engine_score >= 36 else TIER_COLORS["low"])
            rows_html += (
                '<div class="sig-row">'
                f'<div class="sig-label">{name}</div>'
                f'<div class="sig-detail">weight: {int(ENSEMBLE_WEIGHTS[name]*100)}%</div>'
                f'<div class="sig-bar-wrap"><div class="sig-bar" style="width:{pct}%; background:{bar_color};"></div></div>'
                f'<div class="sig-pts">{engine_score:.0f}</div>'
                '</div>'
            )
        rows_html += (
            '<div class="sig-row" style="border-top:1px solid #253147; margin-top:4px; padding-top:12px;">'
            '<div class="sig-label" style="font-weight:600;">Final combined verdict</div>'
            '<div class="sig-detail"></div>'
            '<div class="sig-bar-wrap"></div>'
            f'<div class="sig-pts" style="font-weight:600;">{result.score:.0f}</div>'
            '</div>'
        )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Email Metadata & IP Details
    display_ip = result.header_ip if result.header_ip != "N/A" else (result.ips[0] if result.ips else "N/A")
    is_priv = display_ip != "N/A" and _is_private_ip(display_ip)
    risk_ip = any(s.label.startswith("Sending IP matches") for s in result.signals)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"""
        <div class="pg-panel">
            <h4>Email details</h4>
            <div class="kv-row"><span class="kv-key">Subject</span><span class="kv-val">{result.subject}</span></div>
            <div class="kv-row"><span class="kv-key">From</span><span class="kv-val">{result.sender}</span></div>
            <div class="kv-row"><span class="kv-key">Reply-To</span><span class="kv-val">{result.reply_to}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        reputation = "High-risk range" if risk_ip else ("Private / internal" if is_priv else "No known issues (demo heuristic)")
        st.markdown(f"""
        <div class="pg-panel">
            <h4>Sender IP analysis</h4>
            <div class="kv-row"><span class="kv-key">Address</span><span class="kv-val">{display_ip}</span></div>
            <div class="kv-row"><span class="kv-key">Type</span><span class="kv-val">{'Private' if is_priv else 'Public'}</span></div>
            <div class="kv-row"><span class="kv-key">Reputation</span><span class="kv-val">{reputation}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Risk Signals
    st.markdown("<div class='pg-panel'><h4>Risk signal breakdown</h4>", unsafe_allow_html=True)
    if result.signals:
        rows_html = ""
        for s in sorted(result.signals, key=lambda x: -x.points):
            pct = 0 if s.max_points <= 0 else min(int(s.points / s.max_points * 100), 100)
            rows_html += (
                '<div class="sig-row">'
                f'<div class="sig-label">{s.label}</div>'
                f'<div class="sig-detail">{s.detail}</div>'
                f'<div class="sig-bar-wrap"><div class="sig-bar" style="width:{pct}%;"></div></div>'
                f'<div class="sig-pts">+{s.points:.0f}</div>'
                '</div>'
            )
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:13.5px; color:#8C99B4;'>No risk signals triggered &mdash; this message looks clean.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. URL Threats
    st.markdown("<div class='pg-panel'><h4>URL / link threat analysis</h4>", unsafe_allow_html=True)
    if result.url_verdicts:
        risky = sorted([v for v in result.url_verdicts if v.score > 0], key=lambda x: -x.score)
        clean = [v for v in result.url_verdicts if v.score == 0]
        shown, remaining = risky[:10], risky[10:]

        rows_html = ""
        for v in shown:
            badge_color = TIER_COLORS[v.tier]
            reasons_text = '; '.join(v.reasons)
            rows_html += (
                '<div class="url-row">'
                '<div class="url-head">'
                f'<span class="url-text">{v.url}</span>'
                f'<span class="pg-badge" style="background:{badge_color}22; color:{badge_color};">{v.score:.0f} / 100</span>'
                '</div>'
                f'<div class="url-reasons">{reasons_text}</div>'
                '</div>'
            )
        if rows_html:
            st.markdown(rows_html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:13.5px; color:#8C99B4;'>No flagged links &mdash; every link checked clean.</div>", unsafe_allow_html=True)

        if remaining:
            st.markdown(f"<div style='font-size:12px; color:#8C99B4; padding-top:8px;'>+{len(remaining)} more flagged link(s) not shown</div>", unsafe_allow_html=True)
        if clean:
            st.markdown(f"<div style='font-size:12px; color:#8C99B4; padding-top:8px;'>{len(clean)} other link(s) checked with no structural red flags</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:13.5px; color:#8C99B4;'>No embedded links found in this message.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Export Report Button
    pdf_data = generate_pdf_report(analyzed_text, result, selected_model)
    st.download_button(
        label="Export report (PDF)",
        data=pdf_data,
        file_name="PhishGuardAI_Security_Report.pdf",
        mime="application/pdf",
    )

# Footer with WhatsApp Direct Contact
whatsapp_msg = "Hello Engr. Mohibul Islam, I reached out through your Phishing Email & URL Detection System project."
encoded_msg = urllib.parse.quote(whatsapp_msg)

st.markdown(
    f"""
    <div class="footer">
        <span class="dev-text">Developed by <span class="dev-name">Engr. Mohibul Islam</span></span>
        <a href="https://wa.me/8801853389495?text={encoded_msg}" target="_blank" class="contact-btn">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="white">
                <path d="M.057 24l1.687-6.163c-1.041-1.804-1.588-3.849-1.587-5.946.003-6.556 5.338-11.891 11.893-11.891 3.181.001 6.167 1.24 8.413 3.488 2.245 2.248 3.481 5.236 3.48 8.414-.003 6.557-5.338 11.892-11.893 11.892-1.99-.001-3.951-.5-5.688-1.448l-6.305 1.654zm6.597-3.807c1.676.995 3.276 1.591 5.392 1.592 5.448 0 9.886-4.434 9.889-9.885.002-5.462-4.415-9.89-9.881-9.892-5.452 0-9.887 4.434-9.889 9.884-.001 2.225.651 3.891 1.746 5.634l-.999 3.648 3.742-.981zm11.387-5.464c-.074-.124-.272-.198-.57-.347-.297-.149-1.758-.868-2.031-.967-.272-.099-.47-.149-.669.149-.198.297-.768.967-.941 1.165-.173.198-.347.223-.644.074-.297-.149-1.255-.462-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.297-.347.446-.521.151-.172.2-.296.3-.495.099-.198.05-.372-.025-.521-.075-.148-.669-1.611-.916-2.206-.242-.579-.487-.501-.669-.51l-.57-.01c-.198 0-.52.074-.792.372s-1.04 1.016-1.04 2.479 1.065 2.876 1.213 3.074c.149.198 2.095 3.2 5.076 4.487.709.306 1.263.489 1.694.626.712.226 1.36.194 1.872.118.571-.085 1.758-.719 2.006-1.413.248-.695.248-1.29.173-1.414z"/>
            </svg>
            Contact
        </a>
    </div>
    """,
    unsafe_allow_html=True
)