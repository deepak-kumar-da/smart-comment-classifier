import html as html_lib
import io
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Smart Comment Classifier"

SPAM_MODEL_ID = "siddhu915176/smart-comment-classifier-spam"
TOX_MODEL_ID = "siddhu915176/smart-comment-classifier-toxicity"
LLM_MODEL_ID = "google/flan-t5-base"

TOX_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TOX_THRESHOLDS = {
    "toxic": 0.85,
    "severe_toxic": 0.90,
    "obscene": 0.90,
    "threat": 0.65,
    "insult": 0.80,
    "identity_hate": 0.60,
}

MAX_LENGTH = 128
UNCERTAINTY_MARGIN = 0.08
LLM_BATCH_LIMIT = 50
BULK_MAX_ROWS = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Each button picks a different random comment from its pool
EXAMPLES = {
    "Friendly": [
        "Thanks for sharing this, the tutorial was really clear and helpful!",
        "I respectfully disagree with the second point, but I appreciate the effort you put in.",
        "Great video! Could you make one about deploying this on the cloud?",
        "Loved the examples. This helped me finally understand the concept.",
        "Congratulations on the launch, the new design looks fantastic!",
        "Does anyone know if this works on Windows as well? Thanks in advance.",
        "Well written article. Looking forward to the next part.",
        "This is a really thoughtful take. I hadn't considered that angle before.",
    ],
    "Toxic": [
        "You are absolutely stupid and useless. Nobody wants to hear your nonsense.",
        "What a pathetic idea. Only an idiot would believe this.",
        "Shut up, you moron. Nobody asked for your opinion.",
        "You're a complete waste of space and everyone knows it.",
        "This is garbage and so are you. Delete your account.",
        "Wow, you're so dumb it hurts to read your comments.",
        "Say that again and you'll regret it.",
        "Get lost, loser. Nobody likes you here.",
        "This is such a damn stupid post, who even writes this crap?",
    ],
    "Spam": [
        "WIN a FREE iPhone now!!! Click http://bit.ly/free-prize and claim your reward today",
        "Make $5000 a week from home!!! No experience needed. Click the link in my bio.",
        "Buy cheap followers and likes today, limited offer, DM now!!!",
        "Congratulations! You've been selected for a $1000 gift card. Claim now: http://claim-reward.biz",
        "Hot singles in your area are waiting. Click here to chat now!!!",
        "URGENT: Your account has been suspended. Verify your password at http://secure-login.xyz",
        "Earn passive income with crypto, guaranteed 300% returns. Join our Telegram group today!",
        "Cheap meds online, no prescription needed, best prices guaranteed, order now",
    ],
}

# Plain-English descriptions shown to end users
SIGNALS = {
    "spam": ("Spam", "🚫", "Promotions, scams and junk links"),
    "toxic": ("Toxic", "☣️", "Rude, disrespectful or hostile language"),
    "severe_toxic": ("Severe toxic", "🔥", "Extremely aggressive or hateful language"),
    "obscene": ("Obscene", "🤬", "Profanity and vulgar content"),
    "threat": ("Threat", "⚠️", "Threats of harm or violence"),
    "insult": ("Insult", "🎯", "Personal attacks and demeaning remarks"),
    "identity_hate": ("Identity hate", "🛑", "Hostility toward a group because of who they are"),
}

# Pre-written friendly alternatives, used when the AI draft is not clean enough
FRIENDLY_TEMPLATES = {
    "threat": [
        "I'm really upset about this, but let's keep the conversation safe and respectful.",
        "I strongly disagree with this, and I'd like us to resolve it without hostility.",
    ],
    "identity_hate": [
        "Let's keep this discussion respectful toward everyone and focus on the topic itself.",
        "I'd like to discuss the issue without targeting any group of people.",
    ],
    "obscene": [
        "I'm frustrated by this, and I'd like to explain why in a calm way.",
        "I strongly disagree here, and I'd like to explain my view calmly.",
    ],
    "default": [
        "I disagree with this. Could you explain your reasoning so we can talk it through respectfully?",
        "I don't think this is right, and I'd like to understand your point of view better.",
        "I see this differently. Can we discuss it without the personal attacks?",
    ],
}

VERDICT_ORDER = ["clean", "borderline", "spam", "toxic", "spam_toxic"]
VERDICT_NAMES = {
    "clean": "🟢 Clean",
    "borderline": "🟡 Borderline",
    "spam": "🟠 Spam",
    "toxic": "🔴 Toxic",
    "spam_toxic": "🔴 Spam + Toxic",
    "error": "⚪ Error",
}
VERDICT_COLORS = {
    "clean": "#22c55e",
    "borderline": "#f59e0b",
    "spam": "#f97316",
    "toxic": "#ef4444",
    "spam_toxic": "#991b1b",
}
LEVEL_BY_KEY = {
    "clean": "safe",
    "borderline": "warn",
    "spam": "danger",
    "toxic": "danger",
    "spam_toxic": "danger",
    "error": "warn",
}
LEVEL_ICON = {"safe": "✓", "warn": "?", "danger": "!"}
RING_COLOR = {"safe": "#22c55e", "warn": "#f59e0b", "danger": "#dc2626"}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@700;800&display=swap');

:root {
    --ink: #0f172a;
    --text: #334155;
    --muted: #64748b;
    --line: #e2e8f0;
    --card: #ffffff;
    --brand: #4f46e5;
    --safe: #15803d;
    --safe-bg: #f0fdf4;
    --safe-line: #bbf7d0;
    --warn: #b45309;
    --warn-bg: #fffbeb;
    --warn-line: #fde68a;
    --danger: #b91c1c;
    --danger-bg: #fef2f2;
    --danger-line: #fecaca;
}

.stApp, .stApp p, .stApp label, .stApp span, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button,
.stApp textarea, .stApp input {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
.stApp [data-testid="stIconMaterial"],
.stApp span[class*="material"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(900px 420px at 8% -4%, rgba(99,102,241,0.16), transparent 60%),
        radial-gradient(800px 400px at 96% 2%, rgba(14,165,233,0.15), transparent 60%),
        #f7f9fd;
}

.block-container { max-width: 1120px; padding-top: 1rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Navigation ---------- */
.nav { display: flex; align-items: center; justify-content: space-between; padding: 6px 2px 8px 2px; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-logo {
    width: 40px; height: 40px; border-radius: 12px; color: #fff; font-size: 20px;
    background: linear-gradient(135deg, #4f46e5, #0ea5e9);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 18px rgba(79,70,229,0.35);
}
.brand-name { font-size: 17px; font-weight: 800; color: var(--ink); letter-spacing: -0.3px; }
.nav-pill {
    font-size: 12px; font-weight: 600; color: var(--safe); background: #fff;
    border: 1px solid var(--safe-line); padding: 6px 13px; border-radius: 999px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}

/* ---------- Hero ---------- */
.hero { text-align: center; padding: 46px 10px 18px 10px; }
.hero-eyebrow {
    display: inline-block; font-size: 12.5px; font-weight: 700; color: var(--brand);
    background: #eef2ff; border: 1px solid #c7d2fe; padding: 6px 14px; border-radius: 999px;
    letter-spacing: 0.3px; margin-bottom: 22px;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    font-size: 56px; font-weight: 800; line-height: 1.07; letter-spacing: -2px;
    color: var(--ink); margin: 0 auto; max-width: 860px;
}
.grad {
    background: linear-gradient(90deg, #4f46e5 0%, #0ea5e9 55%, #14b8a6 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent !important;
}
.hero-sub { font-size: 18px; line-height: 1.65; color: var(--muted); max-width: 680px; margin: 20px auto 0 auto; }
.trust { display: flex; justify-content: center; gap: 22px; flex-wrap: wrap; margin-top: 22px; }
.trust span { font-size: 13px; font-weight: 600; color: var(--text); }
.trust span::before { content: "✓"; color: var(--safe); font-weight: 800; margin-right: 7px; }

.features { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 34px 0 26px 0; }
.feature {
    background: rgba(255,255,255,0.85); border: 1px solid var(--line); border-radius: 18px;
    padding: 20px 20px 18px 20px; box-shadow: 0 6px 20px rgba(15,23,42,0.04);
}
.feature-icon { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-bottom: 14px; }
.fi-1 { background: #fee2e2; } .fi-2 { background: #ffedd5; } .fi-3 { background: #e0e7ff; } .fi-4 { background: #dcfce7; }
.feature-title { font-size: 15px; font-weight: 800; color: var(--ink); margin-bottom: 4px; }
.feature-text { font-size: 13px; color: var(--muted); line-height: 1.55; }

/* ---------- Headings ---------- */
.sec-title { font-size: 24px; font-weight: 800; color: var(--ink); margin: 6px 0 4px 0; letter-spacing: -0.6px; }
.sec-sub { font-size: 14.5px; color: var(--muted); margin-bottom: 18px; }
.sec-label { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1.3px; margin: 30px 0 12px 0; }
.sec-note { font-size: 13.5px; color: var(--muted); margin: -4px 0 14px 0; }
.step-head { display: flex; align-items: center; gap: 12px; margin: 26px 0 12px 0; }
.step-num { width: 30px; height: 30px; border-radius: 50%; background: var(--brand); color: #fff; font-size: 14px; font-weight: 800; display: flex; align-items: center; justify-content: center; }
.step-title { font-size: 17px; font-weight: 800; color: var(--ink); letter-spacing: -0.3px; }

/* ---------- Verdict card ---------- */
.verdict {
    border-radius: 22px; padding: 28px 30px; margin: 6px 0 16px 0; border: 1px solid;
    display: flex; gap: 22px; align-items: center; box-shadow: 0 10px 28px rgba(15,23,42,0.07);
}
.verdict.safe   { background: var(--safe-bg);   border-color: var(--safe-line); }
.verdict.warn   { background: var(--warn-bg);   border-color: var(--warn-line); }
.verdict.danger { background: var(--danger-bg); border-color: var(--danger-line); }
.verdict-icon { flex: 0 0 62px; height: 62px; border-radius: 18px; color: #fff; font-size: 30px; font-weight: 800; display: flex; align-items: center; justify-content: center; }
.safe .verdict-icon { background: var(--safe); } .warn .verdict-icon { background: #d97706; } .danger .verdict-icon { background: var(--danger); }
.verdict-body { flex: 1; min-width: 0; }
.verdict-eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; }
.safe .verdict-eyebrow { color: var(--safe); } .warn .verdict-eyebrow { color: var(--warn); } .danger .verdict-eyebrow { color: var(--danger); }
.verdict-title { font-size: 30px; font-weight: 800; letter-spacing: -0.9px; color: var(--ink); margin: 4px 0 6px 0; line-height: 1.15; }
.verdict-summary { font-size: 15px; color: var(--text); line-height: 1.6; }
.verdict-action { display: inline-block; margin-top: 14px; font-size: 13px; font-weight: 700; padding: 8px 14px; border-radius: 10px; background: #fff; border: 1px solid var(--line); color: var(--ink); }
.chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-size: 12.5px; font-weight: 700; padding: 5px 12px; border-radius: 999px; border: 1px solid; background: #fff; }
.chip.danger { color: var(--danger); border-color: var(--danger-line); }
.chip.warn { color: var(--warn); border-color: var(--warn-line); }
.chip.safe { color: var(--safe); border-color: var(--safe-line); }

/* Risk ring */
.ring { flex: 0 0 auto; width: 104px; height: 104px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.ring-in { width: 80px; height: 80px; border-radius: 50%; background: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: inset 0 0 0 1px var(--line); }
.ring-in b { font-size: 22px; font-weight: 800; color: var(--ink); letter-spacing: -0.5px; line-height: 1; }
.ring-in span { font-size: 10px; font-weight: 700; color: var(--muted); letter-spacing: 1.2px; text-transform: uppercase; margin-top: 4px; }

.quote { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--brand); border-radius: 14px; padding: 14px 18px; color: var(--text); font-size: 14.5px; line-height: 1.6; margin-bottom: 14px; }
.quote small { display: block; color: var(--muted); font-weight: 700; letter-spacing: 1px; text-transform: uppercase; font-size: 11px; margin-bottom: 6px; }

/* ---------- Metric cards ---------- */
.metric { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 14px rgba(15,23,42,0.04); border-top: 3px solid #cbd5e1; }
.metric.t-green { border-top-color: #22c55e; } .metric.t-red { border-top-color: #ef4444; }
.metric.t-orange { border-top-color: #f97316; } .metric.t-blue { border-top-color: #4f46e5; } .metric.t-amber { border-top-color: #f59e0b; }
.metric-label { font-size: 11.5px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 30px; font-weight: 800; color: var(--ink); margin-top: 6px; letter-spacing: -0.8px; }
.metric-note { font-size: 12.5px; color: var(--muted); margin-top: 2px; }

/* ---------- Insight cards ---------- */
.ai-card { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 20px 22px; height: 100%; box-shadow: 0 4px 14px rgba(15,23,42,0.035); }
.ai-card h5 { margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.ai-card p { margin: 0; color: #1e293b; font-size: 14.5px; line-height: 1.7; }
.ai-card.friendly { background: linear-gradient(180deg, #ffffff, #f0fdf4); border-color: var(--safe-line); }
.tag-src { font-size: 10.5px; font-weight: 800; letter-spacing: 0.6px; color: var(--safe); background: #dcfce7; border-radius: 999px; padding: 3px 9px; text-transform: none; }

/* ---------- Analysis breakdown ---------- */
.an { background: #fff; border: 1px solid var(--line); border-radius: 20px; overflow: hidden; box-shadow: 0 6px 20px rgba(15,23,42,0.045); }
.an-head, .an-row { display: grid; grid-template-columns: minmax(0,1.5fr) minmax(0,2fr) 120px; gap: 18px; align-items: center; padding: 14px 24px; }
.an-head { background: #f8fafc; font-size: 11.5px; font-weight: 700; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
.an-group { padding: 10px 24px; background: #fbfcfe; border-top: 1px solid #eef2f7; font-size: 11.5px; font-weight: 800; color: var(--brand); letter-spacing: 1.2px; text-transform: uppercase; }
.an-row { border-top: 1px solid #f1f5f9; }
.an-row.danger { background: #fff7f7; } .an-row.warn { background: #fffdf5; }
.an-name b { display: block; font-size: 14px; color: var(--ink); }
.an-name span { font-size: 12.5px; color: var(--muted); }
.an-bar { display: flex; align-items: center; gap: 12px; }
.an-track { position: relative; flex: 1; height: 9px; background: #eef2f7; border-radius: 999px; }
.an-fill { height: 100%; border-radius: 999px; }
.an-fill.danger { background: linear-gradient(90deg, #fb923c, #dc2626); }
.an-fill.warn { background: linear-gradient(90deg, #fcd34d, #f59e0b); }
.an-fill.ok { background: linear-gradient(90deg, #93c5fd, #6366f1); }
.an-tick { position: absolute; top: -4px; width: 2px; height: 17px; background: #0f172a; opacity: 0.4; border-radius: 2px; }
.an-pct { width: 50px; text-align: right; font-weight: 800; font-size: 13.5px; color: var(--ink); font-variant-numeric: tabular-nums; }
.an-res { text-align: right; }
.pill { display: inline-block; font-size: 11.5px; font-weight: 800; padding: 4px 12px; border-radius: 999px; }
.pill.danger { background: var(--danger); color: #fff; } .pill.warn { background: #f59e0b; color: #fff; } .pill.ok { background: #dcfce7; color: var(--safe); }
.an-foot { padding: 12px 24px; border-top: 1px solid #eef2f7; background: #f8fafc; font-size: 12.5px; color: var(--muted); }

/* ---------- Bulk ---------- */
.empty { background: #fff; border: 1.5px dashed #cbd5e1; border-radius: 20px; padding: 34px 28px; text-align: center; }
.empty-icon { font-size: 34px; margin-bottom: 8px; }
.empty-title { font-size: 18px; font-weight: 800; color: var(--ink); }
.empty-text { font-size: 14px; color: var(--muted); max-width: 520px; margin: 6px auto 0 auto; line-height: 1.6; }
.panel { background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 22px 24px; box-shadow: 0 6px 20px rgba(15,23,42,0.045); height: 100%; }
.panel h5 { margin: 0 0 16px 0; font-size: 12px; font-weight: 700; color: var(--muted); letter-spacing: 1.2px; text-transform: uppercase; }
.donut-wrap { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.donut { position: relative; width: 168px; height: 168px; border-radius: 50%; flex: 0 0 auto; }
.donut::after { content: ""; position: absolute; inset: 26px; background: #fff; border-radius: 50%; }
.donut-center { position: absolute; inset: 0; z-index: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.donut-center b { font-size: 28px; font-weight: 800; color: var(--ink); letter-spacing: -0.8px; line-height: 1; }
.donut-center span { font-size: 11px; font-weight: 700; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }
.legend-col { display: flex; flex-direction: column; gap: 10px; font-size: 13.5px; color: var(--text); }
.legend-col i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 9px; }
.legend-col b { color: var(--ink); margin-left: 6px; }
.hbar { display: grid; grid-template-columns: 110px 1fr 34px; gap: 12px; align-items: center; margin: 11px 0; font-size: 13.5px; }
.hbar-name { color: var(--text); font-weight: 600; }
.hbar-track { height: 10px; background: #eef2f7; border-radius: 999px; overflow: hidden; }
.hbar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #818cf8, #4f46e5); }
.hbar-val { text-align: right; font-weight: 800; color: var(--ink); }

/* ---------- How it works ---------- */
.pipeline { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.pipe { position: relative; background: #fff; border: 1px solid var(--line); border-radius: 20px; padding: 24px 22px; box-shadow: 0 6px 20px rgba(15,23,42,0.045); }
.pipe-num { width: 38px; height: 38px; border-radius: 12px; color: #fff; font-weight: 800; font-size: 17px; background: linear-gradient(135deg, #4f46e5, #0ea5e9); display: flex; align-items: center; justify-content: center; margin-bottom: 14px; }
.pipe h4 { margin: 0 0 6px 0; font-size: 17px; font-weight: 800; color: var(--ink); }
.pipe p { margin: 0; font-size: 14px; color: var(--muted); line-height: 1.6; }
.pipe:not(:last-child)::after { content: "→"; position: absolute; right: -16px; top: 50%; transform: translateY(-50%); font-size: 20px; color: #94a3b8; font-weight: 800; z-index: 2; }
.gloss { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; }
.gloss-item { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; display: flex; gap: 14px; align-items: flex-start; }
.gloss-icon { font-size: 22px; flex: 0 0 auto; width: 42px; height: 42px; border-radius: 12px; background: #f1f5f9; display: flex; align-items: center; justify-content: center; }
.gloss-item b { display: block; font-size: 14.5px; color: var(--ink); margin-bottom: 2px; }
.gloss-item span { font-size: 12.5px; color: var(--muted); line-height: 1.45; }
.vtable { background: #fff; border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
.vrow { display: grid; grid-template-columns: 190px 1fr 190px; gap: 12px; padding: 14px 20px; align-items: center; border-bottom: 1px solid var(--line); font-size: 14px; }
.vrow:last-child { border-bottom: none; }
.vrow.head { background: #f8fafc; font-size: 11.5px; font-weight: 700; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
.vrow .v { font-weight: 800; color: var(--ink); } .vrow .v i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 9px; }
.vrow .m { color: var(--text); } .vrow .a { font-weight: 700; color: var(--ink); }
.tags { display: flex; flex-wrap: wrap; gap: 10px; }
.tag { font-size: 13px; font-weight: 700; color: var(--text); background: #fff; border: 1px solid var(--line); padding: 8px 14px; border-radius: 999px; }

/* ---------- Widgets ---------- */
div.stButton > button { border-radius: 12px; font-weight: 700; min-height: 44px; transition: all .15s ease; border: 1px solid var(--line); background: #fff; color: var(--ink); }
div.stButton > button:hover { border-color: var(--brand); color: var(--brand); }
div.stButton > button[kind="primary"] { background: linear-gradient(135deg, #4f46e5, #2563eb); color: #fff; border: none; }
div.stButton > button[kind="primary"]:hover { color: #fff; box-shadow: 0 10px 22px rgba(79,70,229,.32); transform: translateY(-1px); }
div[data-testid="stDownloadButton"] > button { border-radius: 12px; font-weight: 700; min-height: 44px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; border-color: var(--line); background: #fff; }
.stTextArea textarea { border-radius: 12px; border: 1px solid #cbd5e1; background: #fff; color: var(--ink); font-size: 15px; }
.stTextArea textarea:focus { border-color: var(--brand); box-shadow: 0 0 0 3px rgba(79,70,229,.14); }
div[data-testid="stFileUploader"] section { border-radius: 14px; border: 1.5px dashed #94a3b8; background: #fff; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #e9eef7; padding: 5px; border-radius: 14px; border-bottom: none; width: fit-content; margin: 0 auto 14px auto; justify-content: center; }
.stTabs [data-baseweb="tab"] { font-weight: 700; padding: 9px 20px; border-radius: 10px; color: var(--muted); height: auto; }
.stTabs [aria-selected="true"] { background: #fff; color: var(--brand) !important; box-shadow: 0 2px 8px rgba(15,23,42,0.1); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: #fff; }
div[data-testid="stExpander"] summary { font-weight: 700; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }

.footer { text-align: center; padding: 44px 0 6px 0; border-top: 1px solid var(--line); margin-top: 44px; }
.footer b { display: block; font-size: 15px; color: var(--ink); margin-bottom: 6px; }
.footer span { font-size: 12.5px; color: #94a3b8; }

@media (max-width: 860px) {
    .hero-title { font-size: 36px; letter-spacing: -1.2px; }
    .hero-sub { font-size: 16px; }
    .features { grid-template-columns: repeat(2, 1fr); }
    .pipeline { grid-template-columns: 1fr; }
    .pipe:not(:last-child)::after { content: "↓"; right: 50%; top: auto; bottom: -22px; transform: translateX(50%); }
    .verdict { flex-direction: column; align-items: flex-start; padding: 22px; }
    .verdict-title { font-size: 24px; }
    .an-head { display: none; }
    .an-row { grid-template-columns: 1fr; gap: 8px; }
    .an-res { text-align: left; }
    .vrow { grid-template-columns: 1fr; gap: 2px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def render(markup: str):
    """Render HTML in markdown (flattened so indentation never turns into a code block)."""
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def esc(value) -> str:
    return html_lib.escape(str(value)).replace("$", "&#36;")


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


def pct(p: float) -> str:
    return f"{p:.1%}" if p < 0.1 else f"{p:.0%}"


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def val(row, key, default=""):
    """Safe getter for a DataFrame row that may hold NaN."""
    value = row.get(key, default)
    return default if pd.isna(value) else value


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading moderation models...")
def load_classifiers():

    spam_tokenizer = AutoTokenizer.from_pretrained(SPAM_MODEL_ID)
    tox_tokenizer = AutoTokenizer.from_pretrained(TOX_MODEL_ID)

    spam_model = AutoModelForSequenceClassification.from_pretrained(
        SPAM_MODEL_ID
    ).to(DEVICE)

    tox_model = AutoModelForSequenceClassification.from_pretrained(
        TOX_MODEL_ID
    ).to(DEVICE)

    spam_model.eval()
    tox_model.eval()

    if int(tox_model.config.num_labels) != len(TOX_LABELS):
        raise ValueError(
            "Toxicity model output count does not match six categories."
        )

    return (
        spam_tokenizer,
        spam_model,
        tox_tokenizer,
        tox_model,
        TOX_THRESHOLDS,
    )


@st.cache_resource(show_spinner="Preparing the rewrite assistant...")
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16,
        ).to(DEVICE)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_ID
        ).to(DEVICE)

    model.eval()

    return tokenizer, model


# ============================================================
# CLASSIFICATION ENGINE
# ============================================================

def classify_spam(text, tokenizer, model):

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    if len(probabilities) != 2:
        raise ValueError("Spam model must return two class probabilities.")

    spam_probability = float(probabilities[1])
    is_spam = spam_probability >= 0.5

    return {
        "label": "Spam" if is_spam else "Not spam",
        "prediction": int(is_spam),
        "spam_probability": spam_probability,
        "confidence": float(np.max(probabilities)),
    }


def classify_toxicity(text, tokenizer, model, thresholds):

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    results = {}

    for index, label in enumerate(TOX_LABELS):

        probability = float(probabilities[index])
        threshold = float(thresholds[label])

        results[label] = {
            "probability": probability,
            "threshold": threshold,
            "prediction": int(probability >= threshold),
            "uncertain": abs(probability - threshold) <= UNCERTAINTY_MARGIN,
        }

    return results


def risk_score(result) -> float:
    """Single 0-1 number: how strong the strongest signal is."""
    top_tox = max(v["probability"] for v in result["toxicity"].values())
    return float(max(result["spam"]["spam_probability"], top_tox))


def get_verdict(result):
    """Turn raw model output into one clear, human-readable decision."""

    spam = result["spam"]
    categories = result["categories"]
    uncertain = result["uncertain_categories"]

    is_spam = spam["prediction"] == 1
    is_toxic = bool(categories)

    top_cat = None
    if categories:
        top_cat = max(categories, key=lambda c: result["toxicity"][c]["probability"])

    names = ", ".join(pretty(c) for c in categories)

    if is_spam and is_toxic:
        return {
            "key": "spam_toxic",
            "level": "danger",
            "icon": "✕",
            "title": "Spam and toxic content",
            "summary": (
                f"Spam score {spam['spam_probability']:.0%} and toxicity flagged for {names}. "
                f"Strongest signal: {pretty(top_cat)} at {result['toxicity'][top_cat]['probability']:.0%}."
            ),
            "action": "Recommended action: remove",
        }

    if is_toxic:
        return {
            "key": "toxic",
            "level": "danger",
            "icon": "!",
            "title": "Toxic content detected",
            "summary": (
                f"Flagged for {names}. Strongest signal: {pretty(top_cat)} "
                f"at {result['toxicity'][top_cat]['probability']:.0%}, "
                f"above its {result['toxicity'][top_cat]['threshold']:.0%} limit."
            ),
            "action": "Recommended action: hide or remove",
        }

    if is_spam:
        return {
            "key": "spam",
            "level": "danger",
            "icon": "!",
            "title": "Spam detected",
            "summary": (
                f"Spam probability is {spam['spam_probability']:.0%}, above the 50% limit. "
                "No toxicity categories were flagged."
            ),
            "action": "Recommended action: remove",
        }

    if result["needs_review"]:
        close = [pretty(c) for c in uncertain]
        if result["spam_uncertain"]:
            close.append("Spam")
        return {
            "key": "borderline",
            "level": "warn",
            "icon": "?",
            "title": "Borderline, needs a human check",
            "summary": (
                f"Nothing crossed a limit, but scores are close for: {', '.join(close)}. "
                "A moderator should confirm."
            ),
            "action": "Recommended action: manual review",
        }

    return {
        "key": "clean",
        "level": "safe",
        "icon": "✓",
        "title": "Clean comment",
        "summary": "No spam or toxicity signals crossed their limits.",
        "action": "Recommended action: approve",
    }


def build_explanation(result) -> str:
    """Clear, factual explanation written directly from the detectors' scores."""

    spam = result["spam"]
    tox = result["toxicity"]
    key = result["verdict"]["key"]

    flagged = sorted(result["categories"], key=lambda c: -tox[c]["probability"])
    sentences = []

    if flagged:
        detail = ", ".join(
            f"{pretty(c)} ({pct(tox[c]['probability'])} confidence, limit {tox[c]['threshold']:.0%})"
            for c in flagged
        )
        meaning = "; ".join(lower_first(SIGNALS[c][2]) for c in flagged)
        sentences.append(f"The toxicity detector flagged this comment for {detail}.")
        sentences.append(f"In plain terms, it contains {meaning}.")

    if spam["prediction"] == 1:
        sentences.append(
            f"The spam detector is {pct(spam['spam_probability'])} confident this is spam, "
            f"which means it looks like {lower_first(SIGNALS['spam'][2])}."
        )
        if not flagged:
            sentences.append("No toxicity categories crossed their limits.")
    elif flagged:
        sentences.append(f"The spam detector found no sign of spam ({pct(spam['spam_probability'])}).")

    if key == "borderline":
        close = [pretty(c) for c in result["uncertain_categories"]]
        if result["spam_uncertain"]:
            close.append("Spam")
        verb = "is" if len(close) == 1 else "are"
        sentences.append(
            f"No signal crossed its limit, but {', '.join(close)} {verb} very close, "
            "so a human should double-check."
        )

    if key == "clean":
        top = max(tox, key=lambda c: tox[c]["probability"])
        sentences.append(
            "Neither detector found anything above its safety limit. "
            f"The strongest signal was {pretty(top)} at {pct(tox[top]['probability'])}."
        )

    return " ".join(sentences)


def next_step(key: str) -> str:
    return {
        "clean": "This comment is safe to publish. No action is needed.",
        "borderline": "Send it to a moderator for a quick manual check before deciding.",
        "spam": "Remove the comment, and consider blocking the link or account if it keeps happening.",
        "toxic": "Hide or remove the comment, and consider a gentle warning if the author repeats this behavior.",
        "spam_toxic": "Remove the comment and review the author's account for repeated abuse.",
    }.get(key, "Review this comment manually.")


def classify_comment(text, components):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a valid comment.")

    (
        spam_tokenizer,
        spam_model,
        tox_tokenizer,
        tox_model,
        thresholds,
    ) = components

    text = text.strip()

    spam = classify_spam(text, spam_tokenizer, spam_model)
    toxicity = classify_toxicity(text, tox_tokenizer, tox_model, thresholds)

    categories = [l for l, r in toxicity.items() if r["prediction"] == 1]
    uncertain_categories = [l for l, r in toxicity.items() if r["uncertain"]]

    spam_uncertain = abs(spam["spam_probability"] - 0.5) <= UNCERTAINTY_MARGIN

    is_spam = spam["prediction"] == 1
    is_toxic = bool(categories)

    needs_review = (
        is_spam or is_toxic or spam_uncertain or bool(uncertain_categories)
    )

    if is_spam and is_toxic:
        status = "Spam and toxicity detected"
    elif is_spam:
        status = "Spam detected"
    elif is_toxic:
        status = "Toxicity detected"
    elif needs_review:
        status = "Uncertain - review recommended"
    else:
        status = "No flags detected"

    result = {
        "text": text,
        "spam": spam,
        "toxicity": toxicity,
        "categories": categories,
        "uncertain_categories": uncertain_categories,
        "spam_uncertain": spam_uncertain,
        "needs_review": needs_review,
        "status": status,
    }
    result["verdict"] = get_verdict(result)
    result["explanation"] = build_explanation(result)
    result["risk"] = risk_score(result)

    return result


# ============================================================
# FRIENDLY REWRITES (toxic comments only)
# ============================================================

def generate_text(prompt, tokenizer, model, max_tokens=110, **gen_kwargs):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    params = dict(max_new_tokens=max_tokens, num_beams=2, do_sample=False)
    params.update(gen_kwargs)

    with torch.inference_mode():
        output = model.generate(**inputs, **params)

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def is_clean_rewrite(candidate, original, components):
    """Reject empty, repetitive, copied or still-toxic drafts."""

    candidate = candidate.strip()
    words = candidate.lower().split()

    if len(candidate) < 12 or len(words) < 3:
        return False

    if len(set(words)) / len(words) < 0.6:
        return False

    if candidate.lower().strip(" .!") == original.lower().strip(" .!"):
        return False

    if len(candidate) > 3 * len(original) + 40:
        return False

    _, _, tox_tokenizer, tox_model, thresholds = components
    check = classify_toxicity(candidate, tox_tokenizer, tox_model, thresholds)

    # The draft must pass our own toxicity detector
    return not any(v["prediction"] == 1 for v in check.values()) and all(
        v["probability"] < 0.5 for v in check.values()
    )


def template_rewrite(result):
    cats = result["categories"]

    if "threat" in cats:
        pool = FRIENDLY_TEMPLATES["threat"]
    elif "identity_hate" in cats:
        pool = FRIENDLY_TEMPLATES["identity_hate"]
    elif "obscene" in cats and not ({"insult", "toxic", "severe_toxic"} & set(cats)):
        pool = FRIENDLY_TEMPLATES["obscene"]
    else:
        pool = FRIENDLY_TEMPLATES["default"]

    return pool[sum(ord(ch) for ch in result["text"]) % len(pool)]


def build_rewrite(result, components, use_llm=False):
    """Returns (text, source) for toxic comments, or (None, None) for everything else."""

    if not result["categories"]:
        return None, None

    original = result["text"][:600]

    if use_llm:
        try:
            tokenizer, model = load_llm()
            prompt = (
                "Rewrite the following comment so it is polite, calm and respectful "
                "while keeping the same opinion. Do not use insults or bad words.\n\n"
                f"Comment: {original}\n\nPolite version:"
            )
            candidate = generate_text(
                prompt,
                tokenizer,
                model,
                max_tokens=60,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.4,
            ).strip().strip('"')

            if is_clean_rewrite(candidate, original, components):
                return candidate, "AI rewrite"
        except Exception:
            pass

    return template_rewrite(result), "Suggested reply"


# ============================================================
# CSV OUTPUT
# ============================================================

def flatten_result(result):

    flags = (["Spam"] if result["spam"]["prediction"] == 1 else []) + [
        pretty(c) for c in result["categories"]
    ]

    row = {
        "comment": result["text"],
        "verdict": result["verdict"]["title"],
        "verdict_key": result["verdict"]["key"],
        "recommended_action": result["verdict"]["action"].replace("Recommended action: ", "").capitalize(),
        "risk_score": result["risk"],
        "flags": ", ".join(flags),
        "reason": result["verdict"]["summary"],
        "explanation": result["explanation"],
        "friendly_rewrite": result.get("rewrite") or "",
        "rewrite_source": result.get("rewrite_source") or "",
        "spam_probability": result["spam"]["spam_probability"],
        "uncertain_categories": ", ".join(pretty(c) for c in result["uncertain_categories"]),
        "human_review_required": result["needs_review"],
    }

    for label, values in result["toxicity"].items():
        row[f"{label}_probability"] = values["probability"]
        row[f"{label}_prediction"] = values["prediction"]

    return row


# ============================================================
# UI COMPONENTS
# ============================================================

def metric_card(label, value, note="", tone=""):
    render(
        f"""
        <div class="metric {tone}">
            <div class="metric-label">{esc(label)}</div>
            <div class="metric-value">{esc(value)}</div>
            <div class="metric-note">{esc(note)}</div>
        </div>
        """
    )


def render_verdict(result):
    v = result["verdict"]

    chips = []
    if result["spam"]["prediction"] == 1:
        chips.append('<span class="chip danger">Spam</span>')
    for c in result["categories"]:
        chips.append(f'<span class="chip danger">{esc(pretty(c))}</span>')
    for c in result["uncertain_categories"]:
        if c not in result["categories"]:
            chips.append(f'<span class="chip warn">Borderline: {esc(pretty(c))}</span>')
    if result["spam_uncertain"] and result["spam"]["prediction"] == 0:
        chips.append('<span class="chip warn">Borderline: Spam</span>')
    if not chips:
        chips.append('<span class="chip safe">No flags</span>')

    risk = result["risk"]
    ring_color = RING_COLOR[v["level"]]

    render(
        f"""
        <div class="verdict {v['level']}">
            <div class="verdict-icon">{v['icon']}</div>
            <div class="verdict-body">
                <div class="verdict-eyebrow">Moderation verdict</div>
                <div class="verdict-title">{esc(v['title'])}</div>
                <div class="verdict-summary">{esc(v['summary'])}</div>
                <div class="chips">{''.join(chips)}</div>
                <div class="verdict-action">{esc(v['action'])}</div>
            </div>
            <div class="ring" style="background: conic-gradient({ring_color} {risk * 360:.0f}deg, #e2e8f0 0);">
                <div class="ring-in"><b>{risk:.0%}</b><span>Risk</span></div>
            </div>
        </div>
        """
    )


def render_insights(explanation, key, rewrite, source):
    a, b = st.columns(2)
    with a:
        render(
            f"""
            <div class="ai-card"><h5>Why this decision</h5>
            <p>{esc(explanation)}</p></div>
            """
        )
    with b:
        if rewrite:
            render(
                f"""
                <div class="ai-card friendly"><h5>Friendlier way to say it
                <span class="tag-src">{esc(source)}</span></h5>
                <p>{esc(rewrite)}</p></div>
                """
            )
        else:
            render(
                f"""
                <div class="ai-card"><h5>What to do next</h5>
                <p>{esc(next_step(key))}</p></div>
                """
            )


def an_row(key, probability, threshold, flagged, uncertain):
    name, _icon, desc = SIGNALS[key]

    if flagged:
        state, label = "danger", "Flagged"
    elif uncertain:
        state, label = "warn", "Borderline"
    else:
        state, label = "ok", "Clear"

    width = max(min(probability, 1.0) * 100, 1.5)
    row_cls = state if state != "ok" else ""

    return (
        f'<div class="an-row {row_cls}">'
        f'<div class="an-name"><b>{esc(name)}</b><span>{esc(desc)}</span></div>'
        f'<div class="an-bar"><div class="an-track">'
        f'<div class="an-fill {state}" style="width:{width:.1f}%"></div>'
        f'<i class="an-tick" style="left:{threshold * 100:.1f}%"></i></div>'
        f'<span class="an-pct">{pct(probability)}</span></div>'
        f'<div class="an-res"><span class="pill {state}">{label}</span></div>'
        f"</div>"
    )


def render_breakdown(result):
    spam_html = an_row(
        "spam",
        result["spam"]["spam_probability"],
        0.5,
        result["spam"]["prediction"] == 1,
        result["spam_uncertain"] and result["spam"]["prediction"] == 0,
    )

    tox_items = sorted(
        result["toxicity"].items(),
        key=lambda kv: (-kv[1]["prediction"], -kv[1]["probability"]),
    )
    tox_html = "".join(
        an_row(
            label,
            v["probability"],
            v["threshold"],
            v["prediction"] == 1,
            v["uncertain"] and v["prediction"] == 0,
        )
        for label, v in tox_items
    )

    render(
        '<div class="an">'
        '<div class="an-head"><div>Signal</div><div>Detector confidence</div><div style="text-align:right">Result</div></div>'
        '<div class="an-group">Spam detection</div>'
        + spam_html
        + '<div class="an-group">Toxicity detection</div>'
        + tox_html
        + '<div class="an-foot">The dark tick on each bar marks that signal\'s safety limit. '
        "A signal is flagged when its confidence reaches the tick.</div>"
        "</div>"
    )


def render_single_result(result):

    render_verdict(result)

    preview = result["text"]
    if len(preview) > 400:
        preview = preview[:400] + "…"

    render(f'<div class="quote"><small>Analyzed comment</small>{esc(preview)}</div>')

    tox_values = result["toxicity"]
    top_label = max(tox_values, key=lambda k: tox_values[k]["probability"])

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Spam score", pct(result["spam"]["spam_probability"]), result["spam"]["label"], "t-orange")
    with c2:
        metric_card("Toxicity flags", f"{len(result['categories'])} of {len(TOX_LABELS)}", "Categories above their limit", "t-red")
    with c3:
        metric_card("Strongest toxicity signal", pct(tox_values[top_label]["probability"]), pretty(top_label), "t-blue")

    if result.get("with_insights"):
        render('<div class="sec-label">Insights</div>')
        render_insights(
            result["explanation"],
            result["verdict"]["key"],
            result.get("rewrite"),
            result.get("rewrite_source"),
        )

    render('<div class="sec-label">Analysis breakdown</div>')
    render('<div class="sec-note">How confident each detector is, and whether it crossed its safety limit.</div>')
    render_breakdown(result)

    buffer = io.StringIO()
    pd.DataFrame([flatten_result(result)]).to_csv(buffer, index=False)
    st.write("")
    st.download_button(
        "Download this report (CSV)",
        data=buffer.getvalue().encode("utf-8"),
        file_name="comment_report.csv",
        mime="text/csv",
        key="single_download",
    )


def render_row_detail(row):
    key = val(row, "verdict_key", "error")
    level = LEVEL_BY_KEY.get(key, "warn")
    risk = float(val(row, "risk_score", 0.0))

    flags = val(row, "flags", "")
    chips = "".join(f'<span class="chip {level}">{esc(f.strip())}</span>' for f in flags.split(",") if f.strip()) or '<span class="chip safe">No flags</span>'

    render(
        f"""
        <div class="verdict {level}">
            <div class="verdict-icon">{LEVEL_ICON[level]}</div>
            <div class="verdict-body">
                <div class="verdict-eyebrow">Moderation verdict</div>
                <div class="verdict-title" style="font-size:24px">{esc(val(row, 'verdict', 'Error'))}</div>
                <div class="verdict-summary">{esc(val(row, 'reason', ''))}</div>
                <div class="chips">{chips}</div>
            </div>
            <div class="ring" style="background: conic-gradient({RING_COLOR[level]} {risk * 360:.0f}deg, #e2e8f0 0);">
                <div class="ring-in"><b>{risk:.0%}</b><span>Risk</span></div>
            </div>
        </div>
        """
    )

    render(f'<div class="quote"><small>Comment</small>{esc(val(row, "comment"))}</div>')

    explanation = val(row, "explanation", "")
    if explanation:
        render_insights(
            explanation,
            key,
            val(row, "friendly_rewrite", "") or None,
            val(row, "rewrite_source", ""),
        )


def pick_example(group):
    if group == "Random":
        pool = [t for items in EXAMPLES.values() for t in items]
    else:
        pool = EXAMPLES[group]

    current = st.session_state.get("single_comment_input", "")
    choices = [t for t in pool if t != current] or pool
    st.session_state["single_comment_input"] = random.choice(choices)


def sample_csv() -> bytes:
    sample = pd.DataFrame(
        {
            "comment": [
                "Great article, thanks for the clear explanation!",
                "You are absolutely stupid and useless. Nobody wants to hear your nonsense.",
                "WIN a FREE iPhone now!!! Click http://bit.ly/free-prize and claim your reward today",
                "I disagree with the second point, but I appreciate the effort.",
                "Buy cheap followers today, limited offer, DM now!!!",
                "Shut up, idiot. Nobody asked for your opinion.",
                "Does this work on Windows too? Thanks in advance.",
                "Say that again and you'll regret it.",
                "Loved the examples, this finally made the concept click for me.",
                "URGENT: Your account has been suspended. Verify your password at http://secure-login.xyz",
            ]
        }
    )
    return sample.to_csv(index=False).encode("utf-8")


# ============================================================
# HEADER + HERO
# ============================================================

render(
    """
    <div class="nav">
        <div class="brand">
            <div class="brand-logo">🛡️</div>
            <div class="brand-name">Smart Comment Classifier</div>
        </div>
        <div class="nav-pill">● All models online</div>
    </div>
    """
)

render(
    """
    <div class="hero">
        <div class="hero-eyebrow">✨ AI-powered content moderation</div>
        <div class="hero-title">Moderate every comment <span class="grad">with confidence</span></div>
        <div class="hero-sub">
            Instantly detect spam and toxic language, understand exactly why a comment was flagged,
            and get a clear recommendation you can act on. Built for creators, communities and platforms.
        </div>
        <div class="trust">
            <span>Spam and 6 toxicity signals</span>
            <span>Plain-English explanations</span>
            <span>Bulk CSV moderation</span>
        </div>
    </div>
    """
)

render(
    """
    <div class="features">
        <div class="feature">
            <div class="feature-icon fi-1">🚫</div>
            <div class="feature-title">Spam shield</div>
            <div class="feature-text">Catches promotions, scams and junk links before they reach your audience.</div>
        </div>
        <div class="feature">
            <div class="feature-icon fi-2">🧪</div>
            <div class="feature-title">6 toxicity types</div>
            <div class="feature-text">Insults, threats, obscenity, identity hate and more, scored separately.</div>
        </div>
        <div class="feature">
            <div class="feature-icon fi-3">✨</div>
            <div class="feature-title">Clear explanations</div>
            <div class="feature-text">Understand every decision, and get a friendlier rewrite for toxic comments.</div>
        </div>
        <div class="feature">
            <div class="feature-icon fi-4">📁</div>
            <div class="feature-title">Bulk moderation</div>
            <div class="feature-text">Upload a CSV and get a full report with a decision for every row.</div>
        </div>
    </div>
    """
)

try:
    components = load_classifiers()
except Exception as error:
    st.error("The moderation models could not be loaded.")
    st.exception(error)
    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

tab_single, tab_batch, tab_about = st.tabs(
    ["🔍  Comment Review", "📁  Bulk Moderation", "💡  How it works"]
)


# ============================================================
# SINGLE COMMENT
# ============================================================

with tab_single:

    render('<div class="sec-title">Review a comment</div>')
    render('<div class="sec-sub">Paste any comment to get an instant moderation decision.</div>')

    with st.container(border=True):

        comment = st.text_area(
            "Comment content",
            placeholder="Paste or write the comment you want to analyze...",
            height=130,
            key="single_comment_input",
            label_visibility="collapsed",
        )

        ex_cols = st.columns([1.15, 1, 1, 1, 1.1, 1.1])
        with ex_cols[0]:
            render('<div style="padding-top:12px;color:#64748b;font-size:13px;">Try an example:</div>')
        for col, name in zip(ex_cols[1:4], EXAMPLES.keys()):
            with col:
                st.button(name, key=f"ex_{name}", on_click=pick_example, args=(name,), use_container_width=True)
        with ex_cols[4]:
            st.button("🎲 Random", key="ex_random", on_click=pick_example, args=("Random",), use_container_width=True)

        opt_col, btn_col = st.columns([3, 1])
        with opt_col:
            use_ai_single = st.toggle(
                "Include explanation and friendly rewrite",
                value=True,
                help="Adds a plain-English explanation, and a friendlier rewrite for toxic comments.",
            )
        with btn_col:
            analyze_button = st.button("Analyze comment", type="primary", use_container_width=True)

    if analyze_button:

        if not comment.strip():
            st.warning("Please enter a comment before analysis.")

        else:
            with st.spinner("Analyzing comment..."):
                try:
                    result = classify_comment(comment, components)
                    result["with_insights"] = use_ai_single
                    if use_ai_single and result["categories"]:
                        result["rewrite"], result["rewrite_source"] = build_rewrite(
                            result, components, use_llm=True
                        )
                    st.session_state["single_result"] = result
                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    if "single_result" in st.session_state:
        render('<div class="sec-label">Result</div>')
        render_single_result(st.session_state["single_result"])


# ============================================================
# BULK MODERATION
# ============================================================

with tab_batch:

    render('<div class="sec-title">Bulk moderation</div>')
    render('<div class="sec-sub">Moderate hundreds of comments at once. Upload a CSV and get a decision, an explanation and a friendlier rewrite where needed for every row.</div>')

    # ---- Step 1: upload ----
    render('<div class="step-head"><div class="step-num">1</div><div class="step-title">Upload your comments</div></div>')

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    raw_df = None
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")

    if uploaded is None:
        render(
            """
            <div class="empty">
                <div class="empty-icon">📄</div>
                <div class="empty-title">No file yet</div>
                <div class="empty-text">
                    Upload a CSV with one column of comments (up to 2,000 rows).
                    You'll get a verdict, a risk score, a recommended action and a clear explanation for each comment,
                    plus a downloadable report.
                </div>
            </div>
            """
        )
        st.write("")
        st.download_button(
            "Download a sample CSV to try",
            data=sample_csv(),
            file_name="sample_comments.csv",
            mime="text/csv",
        )

    elif raw_df is not None and raw_df.empty:
        st.warning("The uploaded file has no rows.")

    elif raw_df is not None:

        # ---- Step 2: configure ----
        render('<div class="step-head"><div class="step-num">2</div><div class="step-title">Choose what to analyze</div></div>')

        text_cols = [c for c in raw_df.columns if raw_df[c].dtype == object] or list(raw_df.columns)
        preferred = next(
            (c for c in text_cols if c.lower() in ("comment_text", "comment", "text", "content", "message")),
            text_cols[0],
        )
        n_rows = len(raw_df)

        with st.container(border=True):
            s1, s2 = st.columns(2)
            with s1:
                column = st.selectbox(
                    "Column containing the comments",
                    text_cols,
                    index=text_cols.index(preferred),
                )
            with s2:
                if n_rows == 1:
                    max_rows = 1
                    st.caption("File contains 1 row.")
                else:
                    max_rows = st.slider(
                        "Rows to analyze",
                        min_value=1,
                        max_value=min(BULK_MAX_ROWS, n_rows),
                        value=min(200, n_rows, BULK_MAX_ROWS),
                    )

            use_ai_bulk = st.toggle(
                f"Use AI to draft friendlier rewrites for toxic comments (first {LLM_BATCH_LIMIT})",
                value=False,
                help="Slower on CPU. When off, toxic comments still get a ready-made friendly suggestion.",
            )

            run_bulk = st.button("Run bulk moderation", type="primary")

        if run_bulk:

            subset = raw_df[[column]].dropna().head(max_rows).copy()
            subset[column] = subset[column].astype(str)
            subset = subset[subset[column].str.strip() != ""]

            if subset.empty:
                st.warning("No usable comments found in that column.")
            else:
                rows = []
                progress = st.progress(0.0, text="Starting...")
                total = len(subset)
                ai_used = 0

                for i, text in enumerate(subset[column].tolist(), start=1):
                    try:
                        res = classify_comment(text, components)
                        if res["categories"]:
                            res["rewrite"], res["rewrite_source"] = build_rewrite(
                                res, components, use_llm=use_ai_bulk and ai_used < LLM_BATCH_LIMIT
                            )
                            if res["rewrite_source"] == "AI rewrite":
                                ai_used += 1
                        rows.append(flatten_result(res))
                    except Exception as error:
                        rows.append(
                            {
                                "comment": text,
                                "verdict": "Error",
                                "verdict_key": "error",
                                "recommended_action": "Check input",
                                "risk_score": 0.0,
                                "reason": str(error),
                            }
                        )

                    progress.progress(i / total, text=f"Analyzed {i} of {total} comments")

                progress.empty()
                st.session_state["bulk_df"] = pd.DataFrame(rows)

    # ---- Step 3: results ----
    if "bulk_df" in st.session_state:

        out = st.session_state["bulk_df"]
        counts = out["verdict_key"].value_counts()
        total = len(out)

        def n(key):
            return int(counts.get(key, 0))

        clean_n = n("clean")
        spam_n = n("spam") + n("spam_toxic")
        toxic_n = n("toxic") + n("spam_toxic")
        attention_n = total - clean_n

        render('<div class="step-head"><div class="step-num">3</div><div class="step-title">Review the results</div></div>')

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            metric_card("Analyzed", f"{total:,}", "Comments", "t-blue")
        with k2:
            metric_card("Clean", f"{clean_n:,}", f"{clean_n / total:.0%} of total", "t-green")
        with k3:
            metric_card("Needs attention", f"{attention_n:,}", f"{attention_n / total:.0%} of total", "t-amber")
        with k4:
            metric_card("Spam", f"{spam_n:,}", f"{spam_n / total:.0%} of total", "t-orange")
        with k5:
            metric_card("Toxic", f"{toxic_n:,}", f"{toxic_n / total:.0%} of total", "t-red")

        st.write("")
        left, right = st.columns(2)

        # Donut chart
        stops, acc = [], 0.0
        for k in VERDICT_ORDER:
            if n(k) == 0:
                continue
            start = acc
            acc += n(k) / total * 100
            stops.append(f"{VERDICT_COLORS[k]} {start:.2f}% {acc:.2f}%")
        if acc < 99.99:
            stops.append(f"#e2e8f0 {acc:.2f}% 100%")
        gradient = "conic-gradient(" + ", ".join(stops) + ")"

        legend = "".join(
            f'<div><i style="background:{VERDICT_COLORS[k]}"></i>{esc(VERDICT_NAMES[k][2:].strip())}<b>{n(k):,}</b></div>'
            for k in VERDICT_ORDER
        )

        with left:
            render(
                f"""
                <div class="panel"><h5>Verdict breakdown</h5>
                <div class="donut-wrap">
                    <div class="donut" style="background:{gradient}">
                        <div class="donut-center"><b>{total:,}</b><span>Comments</span></div>
                    </div>
                    <div class="legend-col">{legend}</div>
                </div></div>
                """
            )

        # Issue bars
        issues = [("Spam", spam_n)]
        for lbl in TOX_LABELS:
            col = f"{lbl}_prediction"
            issues.append((SIGNALS[lbl][0], int(out[col].sum()) if col in out.columns else 0))
        issues.sort(key=lambda kv: -kv[1])
        max_issue = max((c for _, c in issues), default=0) or 1

        bars = "".join(
            f'<div class="hbar"><div class="hbar-name">{esc(name)}</div>'
            f'<div class="hbar-track"><div class="hbar-fill" style="width:{c / max_issue * 100:.1f}%"></div></div>'
            f'<div class="hbar-val">{c:,}</div></div>'
            for name, c in issues
        )

        with right:
            render(f'<div class="panel"><h5>What was found</h5>{bars}</div>')

        # Table
        render('<div class="sec-label">Comment-by-comment results</div>')

        f1, f2, f3 = st.columns([3, 2, 2])
        status_options = [VERDICT_NAMES[k] for k in VERDICT_ORDER]
        with f1:
            chosen = st.multiselect(
                "Show",
                options=status_options,
                default=[s for s in status_options if s != VERDICT_NAMES["clean"]],
                help="Clean comments are hidden by default so you can focus on what needs action.",
            )
        with f2:
            query = st.text_input("Search comments", placeholder="Type a word or phrase...")
        with f3:
            sort_by = st.selectbox("Sort by", ["Highest risk first", "Original order"])

        view = out.copy()
        view["status"] = view["verdict_key"].map(VERDICT_NAMES).fillna("⚪ Error")
        view["risk_score"] = view["risk_score"].fillna(0.0)

        if chosen:
            view = view[view["status"].isin(chosen)]
        if query.strip():
            view = view[view["comment"].astype(str).str.contains(query.strip(), case=False, na=False, regex=False)]
        if sort_by == "Highest risk first":
            view = view.sort_values("risk_score", ascending=False)

        show_cols = [
            c
            for c in ["status", "comment", "risk_score", "recommended_action", "flags", "reason"]
            if c in view.columns
        ]

        st.dataframe(
            view[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Verdict", width="small"),
                "comment": st.column_config.TextColumn("Comment", width="large"),
                "risk_score": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
                "recommended_action": st.column_config.TextColumn("Action", width="small"),
                "flags": st.column_config.TextColumn("Flags", width="medium"),
                "reason": st.column_config.TextColumn("Why", width="large"),
            },
        )
        st.caption(f"Showing {len(view):,} of {total:,} comments.")

        # Inspector
        render('<div class="sec-label">Inspect a comment</div>')
        render('<div class="sec-note">Pick any comment to see the full explanation and, for toxic comments, a friendlier rewrite.</div>')

        if len(view) == 0:
            st.info("No comments match the current filters.")
        else:
            options = list(view.index)

            def label_for(i):
                text = str(out.loc[i, "comment"])
                short = text if len(text) <= 80 else text[:80] + "…"
                status = VERDICT_NAMES.get(out.loc[i, "verdict_key"], "⚪ Error")
                return f"{status}  ·  {short}"

            picked = st.selectbox(
                "Choose a comment",
                options,
                format_func=label_for,
                label_visibility="collapsed",
            )
            render_row_detail(out.loc[picked])

        # Downloads
        render('<div class="sec-label">Export</div>')

        full_buffer = io.StringIO()
        out.to_csv(full_buffer, index=False)
        flagged_buffer = io.StringIO()
        out[out["verdict_key"] != "clean"].to_csv(flagged_buffer, index=False)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download full report (CSV)",
                data=full_buffer.getvalue().encode("utf-8"),
                file_name="moderation_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download flagged comments only (CSV)",
                data=flagged_buffer.getvalue().encode("utf-8"),
                file_name="flagged_comments.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# HOW IT WORKS
# ============================================================

with tab_about:

    render('<div class="sec-title">How it works</div>')
    render('<div class="sec-sub">Every comment goes through three steps before you see a decision.</div>')

    render(
        """
        <div class="pipeline">
            <div class="pipe">
                <div class="pipe-num">1</div>
                <h4>Analyze</h4>
                <p>Two specialised language models read the comment: one looks for spam,
                the other scores six kinds of toxic language.</p>
            </div>
            <div class="pipe">
                <div class="pipe-num">2</div>
                <h4>Decide</h4>
                <p>Each score is compared with a safety limit tuned for that category.
                The result is a single verdict, a risk score and a recommended action.</p>
            </div>
            <div class="pipe">
                <div class="pipe-num">3</div>
                <h4>Explain</h4>
                <p>You get a plain-English reason for every decision and, for toxic comments,
                a friendlier way to say the same thing. It never changes the verdict.</p>
            </div>
        </div>
        """
    )

    render('<div class="sec-label">What we detect</div>')
    render(
        '<div class="gloss">'
        + "".join(
            f'<div class="gloss-item"><div class="gloss-icon">{icon}</div>'
            f"<div><b>{esc(name)}</b><span>{esc(desc)}</span></div></div>"
            for name, icon, desc in SIGNALS.values()
        )
        + "</div>"
    )

    render('<div class="sec-label">Understanding the verdicts</div>')
    render(
        """
        <div class="vtable">
            <div class="vrow head"><div>Verdict</div><div>What it means</div><div>Recommended action</div></div>
            <div class="vrow"><div class="v"><i style="background:#22c55e"></i>Clean</div><div class="m">No signal crossed its limit.</div><div class="a">Approve</div></div>
            <div class="vrow"><div class="v"><i style="background:#f59e0b"></i>Borderline</div><div class="m">Nothing crossed a limit, but a score is very close to one.</div><div class="a">Manual review</div></div>
            <div class="vrow"><div class="v"><i style="background:#f97316"></i>Spam</div><div class="m">Looks like a promotion, scam or junk link.</div><div class="a">Remove</div></div>
            <div class="vrow"><div class="v"><i style="background:#ef4444"></i>Toxic</div><div class="m">One or more toxicity categories were flagged.</div><div class="a">Hide or remove</div></div>
            <div class="vrow"><div class="v"><i style="background:#991b1b"></i>Spam + Toxic</div><div class="m">Both spam and toxicity were flagged.</div><div class="a">Remove</div></div>
        </div>
        """
    )

    render('<div class="sec-label">Frequently asked questions</div>')

    with st.expander("How accurate is it?"):
        st.write(
            "No automated system is perfect. That is why comments that land close to a limit are marked "
            "Borderline and sent to a human instead of being decided automatically. For important decisions, "
            "keep a person in the loop."
        )
    with st.expander("What does the risk score mean?"):
        st.write(
            "It is the confidence of the strongest signal found in the comment, from 0% to 100%. "
            "A high score means at least one detector is very sure; the verdict still depends on each "
            "signal's own safety limit."
        )
    with st.expander("Where do the friendlier rewrites come from?"):
        st.write(
            "For toxic comments, an AI model drafts a calmer version, and the toxicity detector checks it. "
            "If the draft is not clean enough, the app falls back to a safe, pre-written suggestion. "
            "Rewrites are only suggestions and never change the verdict."
        )
    with st.expander("Are my comments saved?"):
        st.write(
            "This app processes comments in memory for your current session and does not store them. "
            "Reports are only created when you download them."
        )
    with st.expander("Which languages are supported?"):
        st.write("Best results are expected on English text. Other languages may be less reliable.")

    render('<div class="sec-label">Built with</div>')
    render(
        '<div class="tags"><span class="tag">PyTorch</span><span class="tag">Hugging Face Transformers</span>'
        '<span class="tag">FLAN-T5</span><span class="tag">Streamlit</span></div>'
    )


render(
    """
    <div class="footer">
        <b>🛡️ Smart Comment Classifier</b>
        <span>AI-assisted moderation. Always review important decisions.</span>
    </div>
    """
)
