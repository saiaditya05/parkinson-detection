import base64
import hashlib
import html
import io
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG = 224
INCONCLUSIVE_LOW = 0.35
INCONCLUSIVE_HIGH = 0.65
HIGH_RISK_THRESHOLD = 0.85

THEMES = {
    "Apple Health White": {
        "bg": "#F8FAFC",
        "surface": "#FFFFFF",
        "surface2": "#F1F5F9",
        "border": "#E2E8F0",
        "text": "#0F172A",
        "muted": "#64748B",
        "accent": "#0F766E",
        "accent_soft": "#CCFBF1",
        "accent_text": "#115E59",
        "hero_bg": "linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 55%, #ECFEFF 100%)",
        "hero_border": "#CCFBF1",
        "hero_text": "#0F172A",
        "hero_sub": "#475569",
        "warn_bg": "#FFFBEB",
        "warn_bd": "#FDE68A",
        "warn_fg": "#B45309",
        "ok_bg": "#F0FDF4",
        "ok_bd": "#BBF7D0",
        "ok_fg": "#15803D",
        "unsure_bg": "#EEF2FF",
        "unsure_bd": "#C7D2FE",
        "unsure_fg": "#4338CA",
        "track": "#E2E8F0",
    },
    "Sleek Dark AI": {
        "bg": "#090E1A",
        "surface": "#111827",
        "surface2": "#1E293B",
        "border": "#2D3A4F",
        "text": "#F1F5F9",
        "muted": "#94A3B8",
        "accent": "#22D3EE",
        "accent_soft": "rgba(34, 211, 238, 0.14)",
        "accent_text": "#67E8F9",
        "hero_bg": "linear-gradient(135deg, #0F172A 0%, #16233B 55%, #0F293A 100%)",
        "hero_border": "#1E3A5F",
        "hero_text": "#F8FAFC",
        "hero_sub": "#CBD5E1",
        "warn_bg": "rgba(245, 158, 11, 0.12)",
        "warn_bd": "rgba(245, 158, 11, 0.38)",
        "warn_fg": "#FBBF24",
        "ok_bg": "rgba(16, 185, 129, 0.12)",
        "ok_bd": "rgba(16, 185, 129, 0.35)",
        "ok_fg": "#34D399",
        "unsure_bg": "rgba(129, 140, 248, 0.14)",
        "unsure_bd": "rgba(129, 140, 248, 0.38)",
        "unsure_fg": "#A5B4FC",
        "track": "#334155",
    },
    "SaaS Indigo": {
        "bg": "#F5F7FF",
        "surface": "#FFFFFF",
        "surface2": "#EEF2FF",
        "border": "#DDE3F4",
        "text": "#1E1B4B",
        "muted": "#475569",
        "accent": "#4F46E5",
        "accent_soft": "rgba(255, 255, 255, 0.18)",
        "accent_text": "#E0E7FF",
        "hero_bg": "linear-gradient(125deg, #1E1B4B 0%, #3730A3 55%, #0284C7 100%)",
        "hero_border": "#3730A3",
        "hero_text": "#FFFFFF",
        "hero_sub": "#E0E7FF",
        "warn_bg": "#FFFBEB",
        "warn_bd": "#FDE68A",
        "warn_fg": "#B45309",
        "ok_bg": "#ECFDF5",
        "ok_bd": "#A7F3D0",
        "ok_fg": "#047857",
        "unsure_bg": "#EEF2FF",
        "unsure_bd": "#C7D2FE",
        "unsure_fg": "#4338CA",
        "track": "#E2E8F0",
    },
    "Warm Editorial": {
        "bg": "#FAF7F2",
        "surface": "#FFFFFF",
        "surface2": "#F3EDE2",
        "border": "#E5DCCB",
        "text": "#1C1917",
        "muted": "#57534E",
        "accent": "#15803D",
        "accent_soft": "#DCFCE7",
        "accent_text": "#166534",
        "hero_bg": "linear-gradient(135deg, #FFFFFF 0%, #F4F9F4 58%, #FEF6EE 100%)",
        "hero_border": "#D6E5D8",
        "hero_text": "#1C1917",
        "hero_sub": "#57534E",
        "warn_bg": "#FFF7ED",
        "warn_bd": "#FED7AA",
        "warn_fg": "#C2410C",
        "ok_bg": "#F0FDF4",
        "ok_bd": "#BBF7D0",
        "ok_fg": "#15803D",
        "unsure_bg": "#F5F3FF",
        "unsure_bd": "#DDD6FE",
        "unsure_fg": "#5B21B6",
        "track": "#E7E0D3",
    },
}

STRINGS = {
    "en": {
        "badge": "🌀 AI Screening Aid · VGG16 + Grad-CAM Explainability",
        "title": "Parkinson's Disease Drawing Screening",
        "subtitle": (
            "Upload multiple hand-drawn spiral or wave images—or scan live with your camera. "
            "The AI estimates Parkinson-like tremor likelihood and highlights exactly where it detected cues."
        ),
        "step1_t": "Upload or Camera Scan",
        "step1_s": "Select 1 to 20+ drawings or snap live",
        "step2_t": "See Exact Chance %",
        "step2_s": "Gauge score + plain-language meaning",
        "step3_t": "Spot Where It Detected",
        "step3_s": "Grad-CAM heatmap highlights key strokes",
        "tab_screen": "🔬 Screen Drawings",
        "tab_bench": "📊 Model & Accuracy",
        "tab_guide": "ℹ️ How It Works",
        "input_title": "1. Input Drawings",
        "input_sub": "Choose multiple files at once or use your camera.",
        "mode_upload": "📁 Upload Files",
        "mode_camera": "📷 Live Camera",
        "heatmap_lbl": "Heatmap overlay intensity",
        "empty_title": "Ready to Analyse Your Drawings",
        "empty_sub": "Upload 1, 2, 3 or more spiral/wave drawings on the left, or switch to Live Camera to scan paper directly.",
        "batch_title": "Batch Screening Overview",
        "table_heading": "Quick Comparison Table",
        "download_btn": "📄 Download Printable Screening Report (HTML)",
        "disclaimer": (
            "Medical Disclaimer: This application is a deep learning research & educational project "
            "(Project Cheyali) and is not a certified medical diagnosis. Always consult a neurologist."
        ),
        "warn_badge": "⚠️ Parkinson-like Pattern",
        "unsure_badge": "❓ Borderline / Inconclusive",
        "ok_badge": "✅ Healthy-like Pattern",
        "chance_lbl": "Parkinson's Pattern Chance",
        "parkinson_bar": "Parkinson-like cues",
        "healthy_bar": "Healthy / steady cues",
        "meaning_lbl": "What this means",
        "next_lbl": "Suggested next step",
        "orig_cap": "Original Drawing",
        "cam_cap": "Where AI Focused (Grad-CAM)",
    },
    "te": {
        "badge": "🌀 AI స్క్రీనింగ్ సహాయం · VGG16 + Grad-CAM విశ్లేషణ",
        "title": "పార్కిన్సన్ వ్యాధి డ్రాయింగ్ స్క్రీనింగ్",
        "subtitle": (
            "మీరు గీసిన స్పైరల్ లేదా వేవ్ చిత్రాలను ఒకేసారి ఎక్కువ సంఖ్యలో అప్‌లోడ్ చేయండి లేదా కెమెరాతో స్కాన్ చేయండి. "
            "పార్కిన్సన్ లక్షణాలు ఎంత శాతం ఉన్నాయో మరియు చిత్రంలో ఎక్కడ గుర్తించబడిందో స్పష్టంగా చూపిస్తుంది."
        ),
        "step1_t": "అప్‌లోడ్ లేదా కెమెరా స్కాన్",
        "step1_s": "ఒకేసారి 1 నుండి 20+ చిత్రాలు ఎంచుకోండి",
        "step2_t": "ఎంత శాతం ఉందో చూడండి",
        "step2_s": "మీటర్ స్కోర్ + సులభమైన వివరణ",
        "step3_t": "ఎక్కడ గుర్తించిందో చూడండి",
        "step3_s": "Grad-CAM రంగులు ముఖ్య గీతలను చూపుతాయి",
        "tab_screen": "🔬 డ్రాయింగ్ స్క్రీనింగ్",
        "tab_bench": "📊 మోడల్ ఖచ్చితత్వం",
        "tab_guide": "ℹ️ ఎలా పనిచేస్తుంది",
        "input_title": "1. డ్రాయింగ్‌లు ఇవ్వండి",
        "input_sub": "ఒకేసారి చాలా ఫైళ్లు అప్‌లోడ్ చేయండి లేదా కెమెరా వాడండి.",
        "mode_upload": "📁 ఫైళ్లు అప్‌లోడ్",
        "mode_camera": "📷 లైవ్ కెమెరా",
        "heatmap_lbl": "హీట్‌మ్యాప్ స్పష్టత (Intensity)",
        "empty_title": "మీ డ్రాయింగ్‌లను పరీక్షించడానికి సిద్ధంగా ఉంది",
        "empty_sub": "ఎడమ వైపున 1, 2, 3 లేదా ఎక్కువ స్పైరల్/వేవ్ ఫోటోలు అప్‌లోడ్ చేయండి లేదా కెమెరాతో స్కాన్ చేయండి.",
        "batch_title": "మొత్తం ఫలితాల సారాంశం",
        "table_heading": "త్వరిత పోలిక పట్టిక",
        "download_btn": "📄 స్క్రీనింగ్ రిపోర్ట్ డౌన్‌లోడ్ చేయండి (HTML)",
        "disclaimer": (
            "గమనిక: ఇది కేవలం రీసెర్చ్ మరియు అవగాహన కోసం చేసిన ప్రాజెక్ట్ (Project Cheyali) మాత్రమే — "
            "వైద్య నిర్ధారణ కాదు. ఏదైనా ఆరోగ్య సందేహం ఉంటే న్యూరాలజిస్ట్‌ను సంప్రదించండి."
        ),
        "warn_badge": "⚠️ పార్కిన్సన్ తరహా లక్షణాలు",
        "unsure_badge": "❓ స్పష్టంగా చెప్పలేము (మధ్యస్థం)",
        "ok_badge": "✅ ఆరోగ్యకరమైన నమూనా",
        "chance_lbl": "పార్కిన్సన్ లక్షణాల అవకాశం",
        "parkinson_bar": "పార్కిన్సన్ సంకేతాలు",
        "healthy_bar": "స్థిరమైన / ఆరోగ్య సంకేతాలు",
        "meaning_lbl": "దీని అర్థం ఏమిటి",
        "next_lbl": "తదుపరి సూచన",
        "orig_cap": "అసలు డ్రాయింగ్",
        "cam_cap": "AI ఎక్కడ గుర్తించింది (Grad-CAM)",
    },
}

MODEL_BENCHMARKS = [
    {
        "name": "VGG16 (Active Model)",
        "tag": "Best Overall",
        "test_acc": 81.7,
        "precision": 78.8,
        "recall": 86.7,
        "f1": 82.5,
        "specificity": 76.7,
        "auc": 0.914,
        "highlight": True,
    },
    {
        "name": "VGG16 + ResNet50 Ensemble",
        "tag": "Runner-up",
        "test_acc": 80.0,
        "precision": None,
        "recall": 80.0,
        "f1": 80.0,
        "specificity": None,
        "auc": 0.903,
        "highlight": False,
    },
    {
        "name": "ResNet50 (Fine-tuned)",
        "tag": "Pretrained",
        "test_acc": 75.0,
        "precision": 75.9,
        "recall": 73.3,
        "f1": 74.6,
        "specificity": 76.7,
        "auc": 0.894,
        "highlight": False,
    },
    {
        "name": "Custom CNN (From Scratch)",
        "tag": "Baseline",
        "test_acc": 50.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "specificity": 100.0,
        "auc": 0.524,
        "highlight": False,
    },
]

st.set_page_config(
    page_title="Parkinson's Drawing Screening",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles(theme_name: str) -> None:
    """Apply full-page CSS variables for the selected theme."""
    t = THEMES[theme_name]
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {{
    --bg: {t["bg"]};
    --surface: {t["surface"]};
    --surface2: {t["surface2"]};
    --border: {t["border"]};
    --text: {t["text"]};
    --muted: {t["muted"]};
    --accent: {t["accent"]};
    --accent-soft: {t["accent_soft"]};
    --accent-text: {t["accent_text"]};
    --hero-bg: {t["hero_bg"]};
    --hero-border: {t["hero_border"]};
    --hero-text: {t["hero_text"]};
    --hero-sub: {t["hero_sub"]};
    --warn-bg: {t["warn_bg"]};
    --warn-bd: {t["warn_bd"]};
    --warn-fg: {t["warn_fg"]};
    --ok-bg: {t["ok_bg"]};
    --ok-bd: {t["ok_bd"]};
    --ok-fg: {t["ok_fg"]};
    --unsure-bg: {t["unsure_bg"]};
    --unsure-bd: {t["unsure_bd"]};
    --unsure-fg: {t["unsure_fg"]};
    --track: {t["track"]};
}}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 2.1rem !important;
}}

.block-container,
.stMainBlockContainer,
[data-testid="stMainBlockContainer"],
section.main .block-container {{
    max-width: 100% !important;
    width: 100% !important;
    padding-top: 0.7rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2.4rem !important;
}}
@media (max-width: 768px) {{
    .block-container,
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"],
    section.main .block-container {{
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }}
}}

/* Global Streamlit widget theming */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 16px !important;
}}
label, .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label div, [data-testid="stCaptionContainer"] p {{
    color: var(--text) !important;
}}
[data-testid="stCaptionContainer"] p {{
    color: var(--muted) !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background-color: var(--surface2) !important;
    border: 1.5px dashed var(--accent) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {{
    color: var(--muted) !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background-color: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}}
[data-testid="stFileUploader"] ul {{
    max-height: 160px;
    overflow-y: auto;
    padding-top: 0.35rem;
}}
[data-testid="stFileUploaderFile"] {{
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}}
[data-testid="stDownloadButton"] button {{
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    width: 100%;
}}
[data-testid="stExpander"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}}
[data-testid="stExpander"] summary p {{
    color: var(--text) !important;
    font-weight: 700 !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: var(--surface2);
    padding: 0.35rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
.stTabs button[data-baseweb="tab"] {{
    border-radius: 9px !important;
    padding: 0.48rem 1.15rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--muted) !important;
}}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {{
    background: var(--surface) !important;
    color: var(--accent) !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.1);
}}

/* Hero & Step Strip */
.hero-card {{
    background: var(--hero-bg);
    border: 1px solid var(--hero-border);
    border-radius: 18px;
    padding: 1.45rem 1.85rem;
    margin-bottom: 0.9rem;
}}
.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--accent-soft);
    color: var(--accent-text);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.55rem;
}}
.hero-title {{
    font-size: 2rem;
    font-weight: 800;
    color: var(--hero-text);
    letter-spacing: -0.025em;
    line-height: 1.16;
    margin: 0 0 0.4rem 0;
}}
.hero-sub {{
    font-size: 1rem;
    color: var(--hero-sub);
    line-height: 1.55;
    margin: 0;
}}
.step-strip {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-bottom: 1rem;
}}
@media (max-width: 768px) {{
    .step-strip {{ grid-template-columns: 1fr; }}
}}
.step-pill {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.78rem 1rem;
}}
.step-num {{
    flex-shrink: 0;
    width: 1.8rem;
    height: 1.8rem;
    border-radius: 50%;
    background: var(--accent);
    color: #FFFFFF;
    font-weight: 800;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}}
.step-text strong {{
    display: block;
    font-size: 0.92rem;
    color: var(--text);
    font-weight: 700;
}}
.step-text span {{
    font-size: 0.82rem;
    color: var(--muted);
}}

/* Section Headings */
.section-heading {{
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.18rem 0;
}}
.section-caption {{
    font-size: 0.89rem;
    color: var(--muted);
    margin-bottom: 0.65rem;
}}

/* Batch Summary & Table */
.batch-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.85rem 1.15rem;
    margin-bottom: 0.85rem;
}}
.batch-title {{
    font-weight: 700;
    font-size: 0.96rem;
    color: var(--text);
}}
.batch-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}}
.mini-pill {{
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.25rem 0.68rem;
    border-radius: 999px;
}}
.mini-pill.warn {{ background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-bd); }}
.mini-pill.ok {{ background: var(--ok-bg); color: var(--ok-fg); border: 1px solid var(--ok-bd); }}
.mini-pill.unsure {{ background: var(--unsure-bg); color: var(--unsure-fg); border: 1px solid var(--unsure-bd); }}

.summary-table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface);
    margin-bottom: 0.95rem;
}}
.summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}}
.summary-table th {{
    text-align: left;
    padding: 0.65rem 0.9rem;
    background: var(--surface2);
    color: var(--muted);
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
}}
.summary-table td {{
    padding: 0.62rem 0.9rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
}}
.summary-table tr:last-child td {{ border-bottom: none; }}

/* Result Card Internals */
.result-header {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.55rem;
    padding-bottom: 0.75rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid var(--border);
}}
.file-chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--text);
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 0.25rem 0.65rem;
    border-radius: 8px;
}}
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.83rem;
    font-weight: 700;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
}}
.status-badge.warn {{ background: var(--warn-bg); color: var(--warn-fg); border: 1px solid var(--warn-bd); }}
.status-badge.ok {{ background: var(--ok-bg); color: var(--ok-fg); border: 1px solid var(--ok-bd); }}
.status-badge.unsure {{ background: var(--unsure-bg); color: var(--unsure-fg); border: 1px solid var(--unsure-bd); }}

.quality-tag {{
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.22rem 0.55rem;
    border-radius: 6px;
    background: var(--surface2);
    color: var(--muted);
    border: 1px solid var(--border);
}}

.chance-card {{
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
}}
.chance-card.warn {{ background: var(--warn-bg); border: 1px solid var(--warn-bd); }}
.chance-card.ok {{ background: var(--ok-bg); border: 1px solid var(--ok-bd); }}
.chance-card.unsure {{ background: var(--unsure-bg); border: 1px solid var(--unsure-bd); }}

.chance-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    flex-wrap: wrap;
}}
.chance-label {{
    font-size: 0.77rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
}}
.chance-big {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.95rem;
    font-weight: 800;
    line-height: 1.1;
}}
.chance-big.warn {{ color: var(--warn-fg); }}
.chance-big.ok {{ color: var(--ok-fg); }}
.chance-big.unsure {{ color: var(--unsure-fg); }}

.risk-tag {{
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.24rem 0.65rem;
    border-radius: 8px;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
}}

.prob-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
    margin: 0.65rem 0 0.25rem 0;
}}
.prob-track {{
    height: 10px;
    background: var(--track);
    border-radius: 999px;
    overflow: hidden;
}}
.prob-fill {{ height: 100%; border-radius: 999px; }}

.explain-box {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-size: 0.89rem;
    line-height: 1.52;
    color: var(--text);
}}
.cam-legend {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.4rem;
    font-size: 0.78rem;
    color: var(--muted);
}}
.legend-bar {{
    flex: 1;
    height: 7px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(14,165,233,0.18) 0%, #14B8A6 35%, #F59E0B 70%, #EF4444 100%);
}}

/* Empty State & Benchmarks */
.empty-panel {{
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 16px;
    padding: 2.2rem 1.5rem;
    text-align: center;
}}
.empty-icon {{ font-size: 2.6rem; margin-bottom: 0.45rem; }}
.empty-title {{ font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem; }}
.empty-sub {{ font-size: 0.93rem; color: var(--muted); max-width: 540px; margin: 0 auto 1.2rem auto; }}

.tips-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    text-align: left;
}}
@media (max-width: 768px) {{
    .tips-grid {{ grid-template-columns: 1fr; }}
}}
.tip-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
}}
.tip-title {{ font-weight: 700; font-size: 0.9rem; color: var(--text); margin-bottom: 0.15rem; }}
.tip-desc {{ font-size: 0.82rem; color: var(--muted); line-height: 1.4; }}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
    margin: 0.4rem 0 1.2rem 0;
}}
@media (max-width: 768px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
.kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
}}
.kpi-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1.1;
}}
.kpi-lbl {{ font-size: 0.84rem; color: var(--muted); margin-top: 0.28rem; }}

.bench-row {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
}}
.bench-row.active {{
    border-color: var(--accent);
    background: var(--surface2);
}}
.bench-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
}}
.bench-name {{ font-weight: 700; font-size: 0.95rem; color: var(--text); }}
.bench-tag {{
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: var(--surface2);
    color: var(--muted);
    border: 1px solid var(--border);
}}
.bench-row.active .bench-tag {{ background: var(--accent); color: #FFFFFF; border-color: var(--accent); }}
.bench-metrics {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.4rem;
}}
.bench-metrics b {{ color: var(--text); font-family: 'JetBrains Mono', monospace; }}

.cm-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.55rem;
    margin-top: 0.6rem;
}}
.cm-cell {{
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
}}
.cm-cell.good {{ background: var(--ok-bg); border: 1px solid var(--ok-bd); color: var(--ok-fg); }}
.cm-cell.bad {{ background: var(--warn-bg); border: 1px solid var(--warn-bd); color: var(--warn-fg); }}
.cm-num {{ font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 800; }}
.cm-cap {{ font-size: 0.78rem; font-weight: 600; margin-top: 0.12rem; }}

.medical-banner {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 0.85rem 1.05rem;
    margin-top: 1.15rem;
    font-size: 0.88rem;
    color: var(--muted);
}}
.app-footer {{
    text-align: center;
    color: var(--muted);
    font-size: 0.81rem;
    margin-top: 1.1rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading Parkinson's VGG16 screening model...")
def load_model():
    """Load VGG16 weights from Hugging Face and prepare Grad-CAM extractor."""
    path = hf_hub_download("saiaditya05/cheyali-vgg16", "vgg16.weights.h5")
    base = VGG16(weights=None, include_top=False, input_shape=(IMG, IMG, 3))
    inp = layers.Input((IMG, IMG, 3))
    x = layers.Lambda(preprocess_input)(inp)
    x = base(x, training=False)
    gap = layers.GlobalAveragePooling2D()
    x = gap(x)
    x = layers.Dropout(0.5)(x)
    head = layers.Dense(1, activation="sigmoid")
    out = head(x)
    model = keras.Model(inp, out)
    model.load_weights(path)
    conv_model = keras.Model(
        base.input,
        [base.get_layer("block5_conv3").output, base.output],
    )
    return model, conv_model, gap, head


def to_rgb_white_bg(img_pil: Image.Image) -> Image.Image:
    """Convert any image (including transparent PNGs) to RGB on a white background."""
    if img_pil.mode in ("RGBA", "LA") or (
        img_pil.mode == "P" and "transparency" in img_pil.info
    ):
        rgba = img_pil.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img_pil.convert("RGB")


def assess_image_quality(img_pil: Image.Image) -> tuple[str, str]:
    """Check brightness and stroke contrast so blank or dark photos are flagged."""
    gray = np.asarray(img_pil.convert("L"), dtype=np.float32)
    mean_lum = float(gray.mean())
    std_lum = float(gray.std())
    if mean_lum < 55.0:
        return "Dark Photo", "Image looks dark; bright even lighting improves accuracy."
    if std_lum < 10.0:
        return "Low Contrast", "Very faint lines or blank page detected; check photo clarity."
    return "Clear Quality", "Good brightness and stroke contrast."


@st.cache_data(max_entries=64, show_spinner=False)
def analyse_image_bytes(image_bytes: bytes) -> tuple[float, np.ndarray, tuple[str, str]]:
    """Run cached VGG16 prediction + Grad-CAM + quality check keyed by image hash."""
    model, conv_model, gap, head = load_model()
    resample = getattr(Image, "Resampling", Image).BILINEAR
    img_pil = to_rgb_white_bg(Image.open(io.BytesIO(image_bytes)))
    quality = assess_image_quality(img_pil)

    resized = img_pil.resize((IMG, IMG), resample)
    arr = np.array(resized, dtype="float32")
    batch = arr[None, ...]

    prob = float(model.predict(batch, verbose=0)[0][0])

    x_pre = preprocess_input(batch.copy())
    x_tf = tf.convert_to_tensor(x_pre, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        feat, conv_out = conv_model(x_tf, training=False)
        gap_out = gap(conv_out)
        try:
            logit = tf.matmul(gap_out, tf.convert_to_tensor(head.kernel)) + tf.convert_to_tensor(
                head.bias
            )
            target = logit if prob >= 0.5 else -logit
        except Exception:
            p_out = head(gap_out)
            target = p_out if prob >= 0.5 else (1.0 - p_out)

    grads = tape.gradient(target, feat)
    if grads is None:
        return prob, np.zeros((14, 14), dtype=np.float32), quality

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1))[0].numpy()
    max_val = float(cam.max())
    cam = cam / max_val if max_val > 1e-8 else np.zeros_like(cam, dtype=np.float32)
    return prob, cam.astype(np.float32), quality


def render_gradcam_overlay(
    img_pil: Image.Image,
    cam: np.ndarray,
    max_alpha: float = 0.62,
) -> Image.Image:
    """Blend a transparent clinical heatmap over the original drawing."""
    base_rgba = img_pil.convert("RGBA")
    w, h = base_rgba.size
    resample = getattr(Image, "Resampling", Image).BILINEAR

    cam_img = Image.fromarray(np.clip(cam * 255.0, 0, 255).astype(np.uint8), mode="L")
    cam_resized = np.array(cam_img.resize((w, h), resample), dtype=np.float32) / 255.0

    stops = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    r_ch = np.interp(cam_resized, stops, [20.0, 245.0, 239.0])
    g_ch = np.interp(cam_resized, stops, [184.0, 158.0, 68.0])
    b_ch = np.interp(cam_resized, stops, [166.0, 11.0, 68.0])

    alpha = np.clip((cam_resized - 0.16) / 0.84, 0.0, 1.0) ** 1.15
    a_ch = alpha * (max_alpha * 255.0)

    heatmap_rgba = np.stack([r_ch, g_ch, b_ch, a_ch], axis=-1).astype(np.uint8)
    overlay = Image.fromarray(heatmap_rgba, mode="RGBA")
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def interpret_probability(prob: float, lang: str) -> dict:
    """Map Parkinson probability to localized plain-language explanation."""
    s = STRINGS[lang]
    pct = prob * 100.0
    if prob >= INCONCLUSIVE_HIGH:
        if lang == "te":
            risk = "అధిక అవకాశం" if prob >= HIGH_RISK_THRESHOLD else "మధ్యస్థ-అధిక అవకాశం"
            meaning = (
                "డ్రాయింగ్ గీతలలో వణుకు (tremor), అసమాన దూరం లేదా అకస్మాత్తుగా మారిన మలుపులను AI గుర్తించింది."
            )
            cam_caption = "ఎరుపు/నారింజ రంగులో హైలైట్ అయిన భాగాల్లో పార్కిన్సన్ సంకేతాలు ఎక్కువగా కనిపించాయి."
            next_step = "ఇది స్క్రీనింగ్ సహాయం మాత్రమే (81.7% ఖచ్చితత్వం). వైద్య నిర్ధారణ కోసం న్యూరాలజిస్ట్‌ను కలవండి."
        else:
            risk = "High Likelihood" if prob >= HIGH_RISK_THRESHOLD else "Moderate-High Likelihood"
            meaning = (
                "The AI detected stroke irregularities—such as tremor oscillations, uneven spacing, "
                "or abrupt shifts—matching Parkinson's training drawings."
            )
            cam_caption = (
                "Highlighted red & orange areas show the exact strokes where Parkinson-like cues were detected."
            )
            next_step = (
                "As an AI screening aid (81.7% test accuracy), consider sharing this result with a neurologist."
            )
        return {
            "css": "warn",
            "badge": s["warn_badge"],
            "risk": risk,
            "chance_text": f"{pct:.1f}%",
            "meaning": meaning,
            "cam_caption": cam_caption,
            "next_step": next_step,
        }
    if prob >= INCONCLUSIVE_LOW:
        if lang == "te":
            risk = "సందిగ్ధం (50/50 దగ్గర)"
            meaning = "స్థిరమైన మరియు వణుకు గీతలు రెండూ మిశ్రమంగా ఉన్నాయి. ఈ ఫోటోపై స్పష్టమైన నిర్ణయం తీసుకోలేకపోయింది."
            cam_caption = "రెండు తరగతులను పోలుస్తూ మోడల్ గమనించిన ముఖ్య ప్రాంతాలను హైలైట్ చేసింది."
            next_step = "మరింత స్పష్టమైన వెలుతురులో పూర్తి స్పైరల్ లేదా వేవ్ ఫోటో తీసి మళ్లీ ప్రయత్నించండి."
        else:
            risk = "Uncertain (Near 50/50)"
            meaning = (
                "The drawing sits near the middle of the scale with a mix of steady and uneven strokes."
            )
            cam_caption = "Highlighted regions show mild irregularities weighed across both classes."
            next_step = "Try a sharper, brighter photo showing the complete spiral or wave on plain paper."
        return {
            "css": "unsure",
            "badge": s["unsure_badge"],
            "risk": risk,
            "chance_text": f"{pct:.1f}%",
            "meaning": meaning,
            "cam_caption": cam_caption,
            "next_step": next_step,
        }
    if lang == "te":
        risk = "తక్కువ అవకాశం"
        meaning = "పెన్ గీతలు సాఫీగా, నిరంతరంగా మరియు సమానమైన దూరంతో ఉన్నాయి — ఆరోగ్యకరమైన నమూనాకు దగ్గరగా ఉంది."
        cam_caption = "ఆరోగ్యకరమైన ఫలితానికి మద్దతు ఇచ్చిన సాఫీ గీతల భాగాలను హైలైట్ చేసింది."
        next_step = "పార్కిన్సన్ లక్షణాలు కనిపించలేదు; ఏవైనా శారీరక లక్షణాలు ఉంటే డాక్టర్‌ను సంప్రదించండి."
    else:
        risk = "Low Likelihood"
        meaning = "Pen strokes look smooth, continuous, and evenly spaced—consistent with healthy drawings."
        cam_caption = "Highlighted segments show the smooth strokes that most supported the Healthy result."
        next_step = "No Parkinson-like pattern was flagged; consult a doctor if physical symptoms persist."
    return {
        "css": "ok",
        "badge": s["ok_badge"],
        "risk": risk,
        "chance_text": f"{pct:.1f}%",
        "meaning": meaning,
        "cam_caption": cam_caption,
        "next_step": next_step,
    }


def build_gauge_svg(prob: float, css_cls: str, theme_name: str) -> str:
    """Build a crisp circular progress ring SVG matching the active theme."""
    t = THEMES[theme_name]
    pct = max(0.0, min(100.0, prob * 100.0))
    radius = 27
    circumference = 2.0 * 3.14159265 * radius
    dash = (pct / 100.0) * circumference
    color_map = {"warn": t["warn_fg"], "unsure": t["unsure_fg"], "ok": t["ok_fg"]}
    stroke_color = color_map.get(css_cls, t["accent"])
    return f"""
<svg width="68" height="68" viewBox="0 0 68 68" aria-label="{pct:.1f}%">
  <circle cx="34" cy="34" r="{radius}" fill="none" stroke="{t['track']}" stroke-width="6.5"/>
  <circle cx="34" cy="34" r="{radius}" fill="none" stroke="{stroke_color}" stroke-width="6.5"
          stroke-linecap="round" stroke-dasharray="{dash:.2f} {circumference:.2f}"
          transform="rotate(-90 34 34)"/>
  <text x="34" y="38" text-anchor="middle" font-family="JetBrains Mono, monospace"
        font-size="12.5" font-weight="800" fill="{t['text']}">{pct:.0f}%</text>
</svg>"""


def pil_to_b64(img_pil: Image.Image, max_dim: int = 280) -> str:
    """Encode a PIL image thumbnail as base64 PNG for the downloadable report."""
    copy_img = img_pil.copy()
    resample = getattr(Image, "Resampling", Image).LANCZOS
    copy_img.thumbnail((max_dim, max_dim), resample)
    buf = io.BytesIO()
    copy_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_html_report(results: list, lang: str) -> str:
    """Generate a self-contained printable HTML screening report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    cards_html = []
    for idx, item in enumerate(results, start=1):
        info = interpret_probability(item["prob"], lang)
        orig_b64 = pil_to_b64(item["orig"])
        cam_b64 = pil_to_b64(item["overlay"])
        safe_name = html.escape(item["name"])
        cards_html.append(
            f"""
<div style="border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin-bottom:16px;page-break-inside:avoid;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <strong>#{idx} · {safe_name}</strong>
    <span style="font-weight:700;">{info['badge']} ({info['chance_text']} Parkinson chance)</span>
  </div>
  <div style="display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;">
    <div><div style="font-size:12px;color:#64748b;">Original</div>
      <img src="data:image/png;base64,{orig_b64}" style="width:190px;border-radius:8px;border:1px solid #e2e8f0;"/></div>
    <div><div style="font-size:12px;color:#64748b;">Where AI Focused (Grad-CAM)</div>
      <img src="data:image/png;base64,{cam_b64}" style="width:190px;border-radius:8px;border:1px solid #e2e8f0;"/></div>
    <div style="flex:1;min-width:220px;font-size:14px;line-height:1.5;">
      <p><b>Risk Level:</b> {info['risk']} | <b>Image Quality:</b> {html.escape(item['quality'][0])}</p>
      <p><b>Meaning:</b> {info['meaning']}</p>
      <p><b>Next Step:</b> {info['next_step']}</p>
    </div>
  </div>
</div>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Parkinson's Drawing Screening Report</title>
<style>body{{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#0f172a;}}</style>
</head><body>
<h2>Parkinson's Disease Drawing Screening Report</h2>
<p style="color:#475569;">Generated: {timestamp} · Model: Fine-tuned VGG16 + Grad-CAM (Test Accuracy: 81.7%, AUC: 0.914)</p>
<hr style="border:none;border-top:1px solid #e2e8f0;margin:16px 0;"/>
{''.join(cards_html)}
<p style="font-size:12px;color:#64748b;margin-top:20px;">{STRINGS[lang]['disclaimer']}</p>
</body></html>"""


def render_top_controls() -> tuple[str, str]:
    """Render top theme and language selectors cleanly across full width."""
    c_brand, c_lang, c_theme = st.columns([2.4, 1.0, 1.3], gap="medium")
    with c_brand:
        st.markdown(
            "<div style='padding-top:0.35rem;font-weight:800;font-size:1.05rem;'>"
            "🌀 Project Cheyali <span style='font-weight:500;opacity:0.7;'>· Clinical AI Drawing Screening</span></div>",
            unsafe_allow_html=True,
        )
    with c_lang:
        lang_choice = st.selectbox(
            "Language",
            options=["English", "తెలుగు (Telugu)"],
            label_visibility="collapsed",
        )
    with c_theme:
        theme_choice = st.selectbox(
            "UI Theme",
            options=list(THEMES.keys()),
            index=0,
            label_visibility="collapsed",
        )
    lang = "te" if lang_choice.startswith("తెలుగు") else "en"
    return lang, theme_choice


def render_hero(lang: str) -> None:
    """Render full-width hero header and 3-step cards."""
    s = STRINGS[lang]
    st.markdown(
        f"""
<div class="hero-card">
  <div class="hero-badge">{s['badge']}</div>
  <h1 class="hero-title">{s['title']}</h1>
  <p class="hero-sub">{s['subtitle']}</p>
</div>
<div class="step-strip">
  <div class="step-pill">
    <span class="step-num">1</span>
    <div class="step-text"><strong>{s['step1_t']}</strong><span>{s['step1_s']}</span></div>
  </div>
  <div class="step-pill">
    <span class="step-num">2</span>
    <div class="step-text"><strong>{s['step2_t']}</strong><span>{s['step2_s']}</span></div>
  </div>
  <div class="step-pill">
    <span class="step-num">3</span>
    <div class="step-text"><strong>{s['step3_t']}</strong><span>{s['step3_s']}</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_result_card(
    index: int,
    item: dict,
    lang: str,
    theme_name: str,
    compact: bool = False,
) -> None:
    """Render a diagnostic card showing original, Grad-CAM, gauge SVG, and explanation."""
    s = STRINGS[lang]
    prob = item["prob"]
    info = interpret_probability(prob, lang)
    safe_name = html.escape(item["name"])
    parkinson_pct = prob * 100.0
    healthy_pct = (1.0 - prob) * 100.0
    gauge_svg = build_gauge_svg(prob, info["css"], theme_name)
    q_label, q_tip = item["quality"]

    details_html = f"""
<div class="chance-card {info['css']}">
  <div class="chance-top">
    <div style="display:flex;align-items:center;gap:0.8rem;">
      {gauge_svg}
      <div>
        <div class="chance-label">{s['chance_lbl']}</div>
        <div class="chance-big {info['css']}">{info['chance_text']}</div>
      </div>
    </div>
    <span class="risk-tag">{info['risk']}</span>
  </div>
  <div class="prob-row"><span>{s['parkinson_bar']}</span><span>{parkinson_pct:.1f}%</span></div>
  <div class="prob-track"><div class="prob-fill" style="width:{parkinson_pct:.1f}%;background:#D97706;"></div></div>
  <div class="prob-row"><span>{s['healthy_bar']}</span><span>{healthy_pct:.1f}%</span></div>
  <div class="prob-track"><div class="prob-fill" style="width:{healthy_pct:.1f}%;background:#0F766E;"></div></div>
</div>
<div class="explain-box">
  <b>{s['meaning_lbl']}:</b> {info['meaning']}<br><br>
  <b>{s['next_lbl']}:</b> {info['next_step']}
</div>
"""

    legend_html = f"""
<div class="cam-legend"><span>Low focus</span><div class="legend-bar"></div><span>Strongest cue</span></div>
<div style="font-size:0.82rem;opacity:0.85;margin-top:0.35rem;line-height:1.45;">{info['cam_caption']}</div>
"""

    with st.container(border=True):
        st.markdown(
            f"""
<div class="result-header">
  <span class="file-chip">#{index} · {safe_name}</span>
  <div style="display:flex;align-items:center;gap:0.45rem;flex-wrap:wrap;">
    <span class="quality-tag" title="{html.escape(q_tip)}">{html.escape(q_label)}</span>
    <span class="status-badge {info['css']}">{info['badge']}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if compact:
            c_orig, c_cam = st.columns(2, gap="small")
            with c_orig:
                st.caption(s["orig_cap"])
                st.image(item["orig"], use_container_width=True)
            with c_cam:
                st.caption(s["cam_cap"])
                st.image(item["overlay"], use_container_width=True)
            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)
            st.markdown(details_html, unsafe_allow_html=True)
        else:
            col_imgs, col_details = st.columns([1.1, 1.0], gap="large")
            with col_imgs:
                c_orig, c_cam = st.columns(2, gap="small")
                with c_orig:
                    st.caption(s["orig_cap"])
                    st.image(item["orig"], use_container_width=True)
                with c_cam:
                    st.caption(s["cam_cap"])
                    st.image(item["overlay"], use_container_width=True)
                st.markdown(legend_html, unsafe_allow_html=True)
            with col_details:
                st.markdown(details_html, unsafe_allow_html=True)


def render_screening_tab(lang: str, theme_name: str) -> None:
    """Render full-page split workspace: Left input controls, Right live results."""
    s = STRINGS[lang]
    left_panel, right_panel = st.columns([0.92, 2.08], gap="large")

    raw_inputs = []
    with left_panel:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-heading">{s["input_title"]}</div>'
                f'<div class="section-caption">{s["input_sub"]}</div>',
                unsafe_allow_html=True,
            )
            input_mode = st.radio(
                "Input method",
                options=[s["mode_upload"], s["mode_camera"]],
                horizontal=True,
                label_visibility="collapsed",
            )
            if input_mode == s["mode_upload"]:
                files = st.file_uploader(
                    "Upload spiral or wave drawings",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                for f in files or []:
                    raw_inputs.append((f.name, f.getvalue()))
            else:
                cam_photo = st.camera_input("Scan drawing", label_visibility="collapsed")
                if cam_photo is not None:
                    raw_inputs.append(("camera_scan.jpg", cam_photo.getvalue()))

            heatmap_strength = st.slider(
                s["heatmap_lbl"],
                min_value=0.25,
                max_value=0.85,
                value=0.62,
                step=0.05,
            )

    if not raw_inputs:
        with right_panel:
            st.markdown(
                f"""
<div class="empty-panel">
  <div class="empty-icon">🌀</div>
  <div class="empty-title">{s['empty_title']}</div>
  <div class="empty-sub">{s['empty_sub']}</div>
  <div class="tips-grid">
    <div class="tip-card">
      <div class="tip-title">📄 Clean Paper</div>
      <div class="tip-desc">Plain white paper with dark pen ink gives the sharpest stroke analysis.</div>
    </div>
    <div class="tip-card">
      <div class="tip-title">💡 Even Lighting</div>
      <div class="tip-desc">Avoid heavy shadows across the spiral or wave when snapping photos.</div>
    </div>
    <div class="tip-card">
      <div class="tip-title">📚 Batch Ready</div>
      <div class="tip-desc">Upload 2, 3, or 20 drawings together to filter &amp; compare side by side.</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        return

    results = []
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with right_panel:
        progress = st.progress(0, text="Analysing drawings...") if len(raw_inputs) > 1 else None
        total = len(raw_inputs)
        for idx, (fname, img_bytes) in enumerate(raw_inputs, start=1):
            if progress is not None:
                progress.progress(idx / total, text=f"Analysing {idx} of {total}: {fname}")
            try:
                img_pil = to_rgb_white_bg(Image.open(io.BytesIO(img_bytes)))
                img_pil.thumbnail((900, 900), resample)
                prob, cam, quality = analyse_image_bytes(img_bytes)
                overlay_pil = render_gradcam_overlay(img_pil, cam, max_alpha=heatmap_strength)
                results.append(
                    {
                        "name": fname,
                        "orig": img_pil,
                        "overlay": overlay_pil,
                        "prob": prob,
                        "quality": quality,
                    }
                )
            except Exception:
                st.warning(f"Could not read `{fname}` as an image. Skipping.")
        if progress is not None:
            progress.empty()

    if not results:
        return

    with left_panel:
        report_html = build_html_report(results, lang)
        st.download_button(
            label=s["download_btn"],
            data=report_html.encode("utf-8"),
            file_name="parkinson_screening_report.html",
            mime="text/html",
        )

    with right_panel:
        if len(results) > 1:
            n_parkinson = sum(1 for r in results if r["prob"] >= INCONCLUSIVE_HIGH)
            n_unsure = sum(1 for r in results if INCONCLUSIVE_LOW <= r["prob"] < INCONCLUSIVE_HIGH)
            n_healthy = sum(1 for r in results if r["prob"] < INCONCLUSIVE_LOW)
            st.markdown(
                f"""
<div class="batch-bar">
  <div class="batch-title">{s['batch_title']} · {len(results)} Drawings</div>
  <div class="batch-pills">
    <span class="mini-pill warn">{n_parkinson} Parkinson-like</span>
    <span class="mini-pill unsure">{n_unsure} Borderline</span>
    <span class="mini-pill ok">{n_healthy} Healthy-like</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander(f"📋 {s['table_heading']} ({len(results)} files)", expanded=True):
                rows_html = []
                for idx, r in enumerate(results, start=1):
                    info = interpret_probability(r["prob"], lang)
                    rows_html.append(
                        f"<tr><td>#{idx}</td><td><code>{html.escape(r['name'])}</code></td>"
                        f"<td><span class='mini-pill {info['css']}'>{info['badge']}</span></td>"
                        f"<td><b>{info['chance_text']}</b></td><td>{html.escape(r['quality'][0])}</td></tr>"
                    )
                st.markdown(
                    f"""
<div class="summary-table-wrap">
  <table class="summary-table">
    <thead><tr><th>#</th><th>Drawing</th><th>Result</th><th>Parkinson Chance</th><th>Quality</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
""",
                    unsafe_allow_html=True,
                )

            f_col, v_col = st.columns([2.0, 1.3], gap="medium")
            with f_col:
                filter_choice = st.radio(
                    "Filter",
                    options=[
                        f"All ({len(results)})",
                        f"Parkinson-like ({n_parkinson})",
                        f"Borderline ({n_unsure})",
                        f"Healthy-like ({n_healthy})",
                    ],
                    horizontal=True,
                    label_visibility="collapsed",
                )
            with v_col:
                view_mode = st.radio(
                    "View mode",
                    options=["2-Column Grid", "Expandable List", "Full-Width Rows"],
                    horizontal=True,
                    label_visibility="collapsed",
                )

            indexed = list(enumerate(results, start=1))
            if filter_choice.startswith("Parkinson-like"):
                filtered = [(i, r) for i, r in indexed if r["prob"] >= INCONCLUSIVE_HIGH]
            elif filter_choice.startswith("Borderline"):
                filtered = [
                    (i, r) for i, r in indexed if INCONCLUSIVE_LOW <= r["prob"] < INCONCLUSIVE_HIGH
                ]
            elif filter_choice.startswith("Healthy-like"):
                filtered = [(i, r) for i, r in indexed if r["prob"] < INCONCLUSIVE_LOW]
            else:
                filtered = indexed

            if not filtered:
                st.info("No drawings match this filter.")
                return

            if view_mode == "2-Column Grid" and len(filtered) > 1:
                for row_start in range(0, len(filtered), 2):
                    cols = st.columns(2, gap="medium")
                    for col, (orig_idx, item) in zip(cols, filtered[row_start : row_start + 2]):
                        with col:
                            render_result_card(orig_idx, item, lang, theme_name, compact=True)
            elif view_mode == "Expandable List":
                for orig_idx, item in filtered:
                    info = interpret_probability(item["prob"], lang)
                    auto_open = item["prob"] >= INCONCLUSIVE_HIGH
                    with st.expander(
                        f"#{orig_idx} · {item['name']} — {info['badge']} ({info['chance_text']})",
                        expanded=auto_open,
                    ):
                        render_result_card(orig_idx, item, lang, theme_name, compact=False)
            else:
                for orig_idx, item in filtered:
                    render_result_card(orig_idx, item, lang, theme_name, compact=False)
        else:
            render_result_card(1, results[0], lang, theme_name, compact=False)


def render_benchmarks_tab() -> None:
    """Render model comparison metrics and confusion matrix from Parkinson.ipynb."""
    st.markdown(
        """
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-val">81.7%</div>
    <div class="kpi-lbl">✅ Accuracy — 49 of 60 unseen test drawings correct (95% CI: 70%–91.7%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">78.8%</div>
    <div class="kpi-lbl">🎯 Precision — of all drawings flagged as Parkinson, 78.8% were actually Parkinson</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">86.7%</div>
    <div class="kpi-lbl">🔍 Recall (Sensitivity) — 26 of 30 Parkinson cases correctly detected</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">82.5%</div>
    <div class="kpi-lbl">⚖️ F1 Score — balanced measure of Precision + Recall (best single metric)</div>
  </div>
</div>
<div style="margin-bottom:1.1rem;">
  <div class="kpi-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem;">
    <div class="kpi-card">
      <div class="kpi-val" style="font-size:1.5rem;">0.914</div>
      <div class="kpi-lbl">ROC-AUC score — 1.000 is perfect, 0.5 is random chance</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-val" style="font-size:1.5rem;">204</div>
      <div class="kpi-lbl">Unique drawings used — 115 train · 29 validation · 60 test</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_list, col_cm = st.columns([1.45, 1.0], gap="large")
    with col_list:
        st.markdown(
            '<div class="section-heading">Model Comparison on Unseen Test Set (60 Images)</div>'
            '<div class="section-caption">Why VGG16 was chosen over Custom CNN, ResNet50, and Ensemble.</div>',
            unsafe_allow_html=True,
        )
        for m in MODEL_BENCHMARKS:
            active_cls = "active" if m["highlight"] else ""
            bar_color = "var(--accent)" if m["highlight"] else "#64748B"
            spec_str = f"{m['specificity']:.1f}%" if m["specificity"] is not None else "—"
            prec_str = f"{m['precision']:.1f}%" if m["precision"] is not None else "—"
            st.markdown(
                f"""
<div class="bench-row {active_cls}">
  <div class="bench-head">
    <span class="bench-name">{m['name']}</span>
    <span class="bench-tag">{m['tag']}</span>
  </div>
  <div class="prob-track" style="height:10px;">
    <div class="prob-fill" style="width:{m['test_acc']}%;background:{bar_color};"></div>
  </div>
  <div class="bench-metrics">
    <span>Accuracy: <b>{m['test_acc']:.1f}%</b></span>
    <span>Precision: <b>{prec_str}</b></span>
    <span>Recall: <b>{m['recall']:.1f}%</b></span>
    <span>F1: <b>{m['f1']:.1f}%</b></span>
    <span>Specificity: <b>{spec_str}</b></span>
    <span>AUC: <b>{m['auc']:.3f}</b></span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with col_cm:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">VGG16 Test Confusion Matrix</div>'
                '<div class="section-caption">Breakdown across 30 Healthy and 30 Parkinson test drawings.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
<div class="cm-grid">
  <div class="cm-cell good"><div class="cm-num">23</div><div class="cm-cap">True Healthy<br>(Correct)</div></div>
  <div class="cm-cell bad"><div class="cm-num">7</div><div class="cm-cap">False Positive<br>(Flagged as PD)</div></div>
  <div class="cm-cell bad"><div class="cm-num">4</div><div class="cm-cap">False Negative<br>(Missed PD)</div></div>
  <div class="cm-cell good"><div class="cm-num">26</div><div class="cm-cap">True Parkinson<br>(Correct)</div></div>
</div>
<div class="explain-box" style="margin-top:0.95rem;">
  <b>Key Takeaway:</b> Fine-tuned <b>VGG16</b> achieved the highest sensitivity (<b>86.7%</b>) and F1 score
  (<b>82.5%</b>), missing only 4 out of 30 Parkinson cases while training from scratch collapsed to 50%.
</div>
""",
                unsafe_allow_html=True,
            )


def render_guide_tab() -> None:
    """Render architecture explanation and limitations."""
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">How the AI Analyses Drawings</div>'
                '<div class="section-caption">From raw photo to tremor heatmap in 3 stages.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
1. **Standardization (`224 × 224 RGB`):** Every uploaded spiral or wave photo is placed on a clean white background, resized, and normalized with VGG16 preprocessing.
2. **Feature Extraction (`VGG16`):** Pre-trained ImageNet filters—fine-tuned on the last 20 layers using the Kaggle Parkinson's Drawings dataset—check pen strokes for tremor oscillations and uneven spacing.
3. **Where It Detected (`Grad-CAM`):** Gradients from the prediction flow back to `block5_conv3` to highlight the exact drawing regions driving the score.
"""
            )
    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">Important Limitations to Know</div>'
                '<div class="section-caption">Transparency &amp; responsible AI notes.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
- **Small Research Dataset:** Trained on **204 unique drawings** (115 train, 29 validation, 60 test). Different pens, ruled paper, or shadows may lower reliability.
- **Confidence Range:** Test accuracy is **81.7%** (95% CI: **70.0%–91.7%**). Scores between **35% and 65%** are marked **Borderline / Inconclusive**.
- **Domain Scope:** Only spiral and wave drawings are valid inputs; unrelated images produce meaningless outputs.
- **Not a Medical Diagnosis:** Built strictly for research &amp; education. Always consult a qualified neurologist.
"""
            )


def main() -> None:
    lang, theme_name = render_top_controls()
    inject_styles(theme_name)
    load_model()
    render_hero(lang)

    s = STRINGS[lang]
    tab_screen, tab_bench, tab_guide = st.tabs([s["tab_screen"], s["tab_bench"], s["tab_guide"]])
    with tab_screen:
        render_screening_tab(lang, theme_name)
    with tab_bench:
        render_benchmarks_tab()
    with tab_guide:
        render_guide_tab()

    st.markdown(
        f"""
<div class="medical-banner">{s['disclaimer']}</div>
<div class="app-footer">
  Project Cheyali · Dataset: Kaggle Parkinson's Drawings (204 unique images) · Model: Fine-tuned VGG16 + Grad-CAM
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
