"""
PhishGuard AI — AI-Based Phishing Email and Malicious URL Detection System
Weighted explainable risk engine + dedicated URL/link threat module + SOC-style dashboard UI.
"""

import re
import ipaddress
from io import BytesIO
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

from url_analysis import analyze_urls, aggregate_url_risk, UrlVerdict


# =========================================================================
# 1. SCORING ENGINE
# =========================================================================
# Every signal contributes a bounded number of points to a 0-100 risk
# score. Weights are additive but individually capped so no single rule
# dominates the verdict, then the total is clamped to [0, 100]. URL/link
# risk is delegated entirely to url_analysis.py so that logic stays in
# one dedicated, independently testable module.

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
    engine_breakdown: List[tuple] = field(default_factory=list)
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

    hits = [w for w in URGENCY_WORDS if w in lower]
    pressure_hits = [w for w in SECURITY_PRESSURE_TERMS if w in lower]
    all_urgency_hits = list(dict.fromkeys(hits + pressure_hits))
    if all_urgency_hits:
        pts = min(5 + 2.5 * len(all_urgency_hits), 18)
        score += pts
        result.signals.append(Signal("Urgency / pressure language", pts, ", ".join(all_urgency_hits[:8]), 18))
        result.triggers.extend(all_urgency_hits)

    hits = [w for w in CREDENTIAL_WORDS if w in lower]
    if hits:
        pts = min(6 + 4 * len(hits), 25)
        score += pts
        result.signals.append(Signal("Credential / sensitive-data request", pts, ", ".join(hits[:6]), 25))
        result.triggers.extend(hits)

    hits = [w for w in FINANCIAL_LURE_WORDS if w in lower]
    if hits:
        pts = min(5 + 3 * len(hits), 15)
        score += pts
        result.signals.append(Signal("Financial lure / prize bait", pts, ", ".join(hits[:6]), 15))
        result.triggers.extend(hits)

    hits = [w for w in ACCOUNT_TAKEOVER_TERMS if w in lower]
    if hits:
        pts = min(8 + 4 * len(hits), 25)
        score += pts
        result.signals.append(
            Signal("Account-takeover / verification pressure", pts, ", ".join(hits[:8]), 25)
        )
        result.triggers.extend(hits)

    if any(g in lower for g in GENERIC_GREETINGS):
        score += 5
        result.signals.append(Signal("Generic, non-personalized greeting", 5, "e.g. 'Dear Customer'", 5))

    sender_domain = _domain_of(result.sender)
    reply_domain = _domain_of(result.reply_to)
    if sender_domain and reply_domain and sender_domain != reply_domain:
        score += 15
        result.signals.append(Signal("Reply-To domain differs from From domain", 15, f"{sender_domain} \u2192 {reply_domain}", 15))

    for brand in BRAND_KEYWORDS:
        if brand in lower and sender_domain and brand not in sender_domain:
            score += 10
            result.signals.append(Signal("Possible brand impersonation", 10, f"mentions '{brand}' but sends from '{sender_domain or 'unknown domain'}'", 10))
            break

    # ---- Dedicated URL / link threat analysis (url_analysis.py) ----
    result.url_verdicts = analyze_urls(result.links)
    if result.url_verdicts:
        url_pts = aggregate_url_risk(result.url_verdicts)
        if url_pts > 0:
            score += url_pts
            worst = max(result.url_verdicts, key=lambda v: v.score)
            result.signals.append(
                Signal("URL / link threat analysis", url_pts, "; ".join(worst.reasons[:3]), 25)
            )
            result.triggers.append("malicious-looking URL")

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

    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    bangs = text.count("!")
    if len(caps_words) >= 2 or bangs >= 3:
        score += 5
        result.signals.append(Signal("Excessive capitalization / punctuation", 5, f"{len(caps_words)} all-caps words, {bangs} '!'", 5))

    result.triggers = list(dict.fromkeys(result.triggers))
    result.score = max(0.0, min(round(score, 1), 100.0))
    return result


# =========================================================================
# 2. MACHINE-LEARNING CLASSIFICATION ENGINES
# =========================================================================
# The rule engine above is the primary, fully explainable safety anchor.
# These ML options are additional engines the user can pick from the
# sidebar. A small built-in training set keeps the app self-contained
# for live demos; for the project report, run train_model.py on the full
# dataset.csv instead — it prints accuracy/precision/recall/F1 for each
# model and saves a confusion-matrix chart.

ML_TRAINING_DATA = [
    # Legitimate (0)
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

    # Phishing / malicious-looking (1) -- synthetic training examples
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
        max_features=8000,
    )
    X = vectorizer.fit_transform(texts)

    models = {
        "Random Forest (NLP)": RandomForestClassifier(
            n_estimators=250, max_depth=18, min_samples_leaf=1,
            random_state=42, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(alpha=0.25),
        "LightGBM": LGBMClassifier(
            n_estimators=120, learning_rate=0.06, num_leaves=15,
            max_depth=7, min_child_samples=2, subsample=0.9,
            colsample_bytree=0.9, random_state=42, verbosity=-1
        ),
    }

    for model in models.values():
        model.fit(X, labels)

    return vectorizer, models


def classify_with_selected_engine(raw_text: str, selected_model: str) -> AnalysisResult:
    """Run the selected engine and return the same AnalysisResult used by the UI."""
    rule_result = analyze_email(raw_text)

    if selected_model == "Weighted Rule Engine v2":
        return rule_result

    vectorizer, models = load_ml_engines()
    X_input = vectorizer.transform([raw_text or ""])
    model = models[selected_model]
    probability = float(model.predict_proba(X_input)[0][1])
    model_score = probability * 100.0

    # The rule engine (which now includes the dedicated URL module) is the
    # primary safety anchor. The ML model is a supporting signal because
    # the built-in training set is intentionally small and synthetic.
    rule_score = float(rule_result.score)
    blended_score = (0.70 * rule_score) + (0.30 * model_score)

    if rule_score <= 10:
        blended_score = min(blended_score, 30.0)
    elif rule_score >= 55 and model_score >= 55:
        blended_score = max(blended_score, 70.0)

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


# ---- Hybrid ensemble: combine the rule engine + all ML classifiers ----
# Each engine votes with a fixed weight. The rule engine (which already
# includes the explainable text-content checks AND the dedicated URL
# module) carries the most weight because it is fully deterministic and
# not limited by the small synthetic ML training set; the three ML
# classifiers contribute the rest as a supporting NLP-based signal.
ENSEMBLE_WEIGHTS = {
    "Weighted Rule Engine v2": 0.40,
    "Random Forest (NLP)": 0.20,
    "LightGBM": 0.20,
    "Naive Bayes": 0.20,
}


def classify_with_ensemble(raw_text: str) -> AnalysisResult:
    """Run every engine and return one weighted, explainable final verdict."""
    rule_result = analyze_email(raw_text)
    vectorizer, models = load_ml_engines()
    X_input = vectorizer.transform([raw_text or ""])

    engine_scores = [("Weighted Rule Engine v2", rule_result.score)]
    for name, model in models.items():
        probability = float(model.predict_proba(X_input)[0][1])
        engine_scores.append((name, round(probability * 100.0, 1)))

    combined = sum(score * ENSEMBLE_WEIGHTS[name] for name, score in engine_scores)

    # Same explainable guardrails as the single-ML-engine path, so the
    # ensemble never contradicts very strong or very weak rule evidence.
    rule_score = rule_result.score
    ml_scores = [s for n, s in engine_scores if n != "Weighted Rule Engine v2"]
    if rule_score <= 10:
        combined = min(combined, 30.0)
    elif rule_score >= 55 and all(s >= 50 for s in ml_scores):
        combined = max(combined, 70.0)

    rule_result.engine_breakdown = engine_scores
    rule_result.score = round(max(0.0, min(combined, 100.0)), 1)
    rule_result.signals.append(
        Signal(
            "Hybrid ensemble combination",
            0,
            "Weighted vote across the rule engine and 3 ML classifiers (see engine breakdown panel)",
            1,
        )
    )
    rule_result.triggers.append("Hybrid ensemble verdict")
    return rule_result


# =========================================================================
# 3. PDF REPORT
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

    # Table cells need Paragraph objects (not raw strings) to wrap properly.
    # wordWrap="CJK" breaks even long unbroken tokens like URLs that have no
    # spaces — without it, long text overflows into neighbouring columns.
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=INK, wordWrap="CJK")
    header_cell_style = ParagraphStyle("HeaderCell", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold")

    def cell(text, header=False):
        return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), header_cell_style if header else cell_style)

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
            story.append(Paragraph(f"{clean_count} other link(s) checked with no structural red flags (tracking pixels, CDN/image assets, etc.).", meta_style))
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
# 4. UI — visual identity
# =========================================================================
st.set_page_config(page_title="PhishGuard AI — Email & URL Threat Analysis", page_icon="◆", layout="wide")

TIER_COLORS = {"high": "#F04438", "medium": "#F79009", "low": "#17B26A"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #0B1220; color: #E7ECF6; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

#MainMenu, footer, header { visibility: hidden; }
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

/* download_button renders under a separate stDownloadButton wrapper,
   so it needs its own rule set (not covered by .stButton above). */
.stDownloadButton>button,
div[data-testid="stDownloadButton"] button {
    background-color: #14B8A6 !important;
    color: #06231F !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.5rem 1.1rem !important;
    box-shadow: none !important;
}
.stDownloadButton>button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background-color: #0D9488 !important;
    color: #FFFFFF !important;
}
.stDownloadButton>button p,
div[data-testid="stDownloadButton"] button p {
    color: inherit !important;
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

.link-row { display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-bottom:1px solid #1C263B; font-size:12.5px; }
.link-row:last-child { border-bottom:none; }
.link-url { font-family:'IBM Plex Mono', monospace; color:#9DB2D4; word-break:break-all; }
.link-tag { color:#8C99B4; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)


def render_gauge(score: float, tier: str) -> str:
    """Return an inline SVG radial gauge for the given score (0-100)."""
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


st.markdown("""
<div class="pg-topbar">
    <span class="pg-mark">PG</span>
    <span class="pg-name">PhishGuard AI</span>
    <span class="pg-sub">&nbsp;&mdash;&nbsp;Phishing email &amp; malicious URL detection</span>
</div>
""", unsafe_allow_html=True)

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

email_input = st.text_area(
   "আপনি যে ইমেইল বা ইউআরএল চেক করতে চান, তা নিচে পেস্ট করুন / Paste the email or URL you want to check below:",
    value="",
    placeholder=sample_email,
    height=220
)
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    analyze_btn = st.button("Analyze", type="primary")
with col_btn2:
    clear_btn = st.button("Clear", type="secondary")

if clear_btn:
    st.rerun()

if analyze_btn:
    if email_input.strip() == "":
        st.warning("Please input email content to analyze.")
    else:
        if selected_model == "Hybrid Ensemble (All Engines)":
            result = classify_with_ensemble(email_input)
        else:
            result = classify_with_selected_engine(email_input, selected_model)
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

        # ---- Engine breakdown panel (only populated in Hybrid Ensemble mode) ----
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

        # ---- Dedicated URL threat-analysis panel ----
        # Real emails (especially Gmail/Google Workspace notifications) can embed
        # dozens of harmless infrastructure links (tracking pixels, CDN scripts,
        # Google image proxies). Showing all of them buries the ones that matter,
        # so flagged links are listed in full and clean links are summarized.
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
                st.markdown(f"<div style='font-size:12px; color:#8C99B4; padding-top:8px;'>{len(clean)} other link(s) checked with no structural red flags (tracking pixels, CDN/image assets, etc.)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:13.5px; color:#8C99B4;'>No embedded links found in this message.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        pdf_data = generate_pdf_report(email_input, result, selected_model)
        st.download_button(
            label="Export report (PDF)",
            data=pdf_data,
            file_name="PhishGuardAI_Security_Report.pdf",
            mime="application/pdf",
        )