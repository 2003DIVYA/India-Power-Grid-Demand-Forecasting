"""
================================================================
  India Hourly Electricity Load — Streamlit Prediction App
  Algorithm : Ridge Regression (RidgeCV with auto alpha tuning)
  UI        : GRID//TERMINAL — Cyberpunk Industrial Design
  Features  : Dashboard · Predict & Analyze · Multi-Day Forecast
              · Data Explorer · Model Insights
  Run       : streamlit run app.py
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline      import Pipeline
from sklearn.linear_model  import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error, r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Energy Consumption Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS — Cyberpunk Industrial Terminal
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
    --bg:         #0a0b0d;
    --bg2:        #0e1014;
    --bg3:        #131519;
    --panel:      #111318;
    --border:     #1e2128;
    --border2:    #252930;
    --amber:      #f5a623;
    --amber-dim:  #8a5e0e;
    --amber-glow: rgba(245,166,35,0.10);
    --cyan:       #00d4ff;
    --cyan-dim:   #006880;
    --cyan-glow:  rgba(0,212,255,0.08);
    --red:        #ff4757;
    --green:      #2ed573;
    --text:       #c8cdd6;
    --text-dim:   #5a6175;
    --text-muted: #2e333c;
    --mono:       'IBM Plex Mono', monospace;
    --display:    'Rajdhani', sans-serif;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }

.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px
    );
    pointer-events: none; z-index: 9999;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 260px !important; max-width: 280px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; background: var(--bg2) !important; }

.sb-logo { padding: 24px 20px 20px; border-bottom: 1px solid var(--border); }
.sb-logo-top {
    font-family: var(--display); font-size: 1.55rem; font-weight: 700;
    color: var(--amber); letter-spacing: 0.06em; line-height: 1;
    text-shadow: 0 0 20px rgba(245,166,35,0.35);
}
.sb-logo-slash { color: #555; font-weight: 300; }
.sb-logo-sub {
    font-size: 0.6rem; color: var(--text-dim); letter-spacing: 0.18em;
    text-transform: uppercase; margin-top: 6px; font-family: var(--mono);
}
.sb-logo-dot {
    display: inline-block; width: 5px; height: 5px; background: var(--green);
    border-radius: 50%; margin-right: 6px; box-shadow: 0 0 7px var(--green);
    animation: blink 2.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

.sb-nav-label {
    font-size: 0.56rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-muted); padding: 18px 20px 8px; font-family: var(--mono);
}

section[data-testid="stSidebar"] .stRadio { padding: 0 10px; }
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex !important; flex-direction: column !important; gap: 1px !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] {
    background: transparent !important; border-radius: 2px !important;
    padding: 9px 12px !important; transition: all 0.12s !important;
    border-left: 2px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"]:hover {
    background: rgba(245,166,35,0.04) !important; border-left-color: var(--amber-dim) !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"][data-checked="true"] {
    background: var(--amber-glow) !important; border-left: 2px solid var(--amber) !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] label {
    font-family: var(--mono) !important; font-size: 0.8rem !important;
    color: var(--text-dim) !important; letter-spacing: 0.06em !important; cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"][data-checked="true"] label {
    color: var(--amber) !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="radio"] div[class*="radioMark"] { display: none !important; }

.sb-stats { margin: 0 10px; border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }
.sb-stats-header {
    background: var(--bg3); padding: 7px 12px; font-size: 0.56rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted);
    border-bottom: 1px solid var(--border); font-family: var(--mono);
}
.sb-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 12px; border-bottom: 1px solid var(--border);
    font-size: 0.72rem; font-family: var(--mono);
}
.sb-stat-row:last-child { border-bottom: none; }
.sb-stat-key   { color: var(--text-muted); }
.sb-stat-val-amber { color: var(--amber); font-weight: 600; }
.sb-stat-val-cyan  { color: var(--cyan);  font-weight: 600; }
.sb-stat-val-green { color: var(--green); font-weight: 600; }

/* ── Page header ── */
.page-header {
    background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 15px 28px; display: flex; align-items: center; justify-content: space-between;
}
.page-title {
    font-family: var(--display); font-size: 1.2rem; font-weight: 700;
    color: var(--text); letter-spacing: 0.06em; text-transform: uppercase;
}
.page-breadcrumb {
    font-size: 0.58rem; color: var(--text-muted); letter-spacing: 0.16em;
    text-transform: uppercase; margin-top: 3px; font-family: var(--mono);
}
.page-badge {
    font-size: 0.6rem; padding: 4px 10px; border-radius: 2px;
    background: var(--amber-glow); border: 1px solid var(--amber-dim);
    color: var(--amber); letter-spacing: 0.12em; font-weight: 600; font-family: var(--mono);
}

/* ── Content ── */
.content-area { padding: 22px 28px; }
.sec-label {
    font-size: 0.56rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 12px; margin-top: 4px;
    display: flex; align-items: center; gap: 8px; font-family: var(--mono);
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: var(--panel) !important; border: 1px solid var(--border) !important;
    border-top: 2px solid var(--amber) !important; border-radius: 2px !important;
    padding: 14px 16px !important;
}
div[data-testid="stMetricValue"] {
    color: var(--amber) !important; font-size: 1.4rem !important; font-weight: 700 !important;
    font-family: var(--mono) !important; text-shadow: 0 0 14px rgba(245,166,35,0.25) !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important; font-size: 0.6rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    font-family: var(--mono) !important;
}
div[data-testid="stMetricDelta"] { font-size: 0.7rem !important; font-family: var(--mono) !important; }

/* ── Inputs ── */
.stSlider > div > div > div > div { background: var(--amber) !important; }
.stSlider > div > div > div { background: var(--border2) !important; }
div[data-baseweb="select"] > div {
    background: var(--panel) !important; border-color: var(--border2) !important;
    border-radius: 2px !important; color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 0.8rem !important;
}
label[data-testid="stWidgetLabel"] p {
    color: var(--text-dim) !important; font-size: 0.68rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    font-family: var(--mono) !important;
}
div[data-testid="stNumberInput"] > div {
    background: var(--panel) !important; border-color: var(--border2) !important;
    border-radius: 2px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important; color: var(--amber) !important;
    border: 1px solid var(--amber) !important; border-radius: 2px !important;
    padding: 10px 0 !important; font-size: 0.78rem !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.2em !important; text-transform: uppercase !important;
    font-family: var(--mono) !important; transition: all 0.18s !important;
}
.stButton > button:hover {
    background: var(--amber-glow) !important;
    box-shadow: 0 0 18px rgba(245,166,35,0.15) !important;
}
.stDownloadButton > button {
    background: transparent !important; color: var(--cyan) !important;
    border: 1px solid var(--cyan-dim) !important; border-radius: 2px !important;
    font-family: var(--mono) !important; font-size: 0.7rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
}
.stDownloadButton > button:hover { background: var(--cyan-glow) !important; }

/* ── Alert boxes ── */
.alert-red {
    background: linear-gradient(135deg, #200000, #380000);
    border-left: 3px solid var(--red); border-radius: 2px;
    padding: 14px 18px; margin: 8px 0; color: #ffcccc;
    font-size: 0.82rem; font-family: var(--mono); line-height: 1.9;
}
.alert-yellow {
    background: linear-gradient(135deg, #1e1500, #2d1f00);
    border-left: 3px solid var(--amber); border-radius: 2px;
    padding: 14px 18px; margin: 8px 0; color: #ffe09a;
    font-size: 0.82rem; font-family: var(--mono); line-height: 1.9;
}
.alert-green {
    background: linear-gradient(135deg, #001a08, #002d10);
    border-left: 3px solid var(--green); border-radius: 2px;
    padding: 14px 18px; margin: 8px 0; color: #a0ffcc;
    font-size: 0.82rem; font-family: var(--mono); line-height: 1.9;
}

/* ── Insight box ── */
.insight-box {
    background: var(--panel); border-radius: 2px; padding: 14px 18px; margin: 8px 0;
    border: 1px solid var(--border); border-left: 3px solid var(--cyan);
    font-size: 0.82rem; color: var(--text); font-family: var(--mono); line-height: 2;
}

/* ── Result card ── */
.result-card {
    background: var(--panel); border: 1px solid var(--border);
    border-top: 3px solid var(--amber); border-radius: 2px;
    padding: 28px 22px; text-align: center; position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at top, rgba(245,166,35,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.result-value {
    font-family: var(--mono); font-size: 3rem; font-weight: 700;
    color: var(--amber); text-shadow: 0 0 28px rgba(245,166,35,0.35); line-height: 1;
}
.result-unit  { font-size: 0.85rem; color: var(--text-dim); margin-top: 6px; letter-spacing: 0.12em; }
.result-label {
    font-size: 0.56rem; color: var(--text-muted); letter-spacing: 0.24em;
    text-transform: uppercase; margin-bottom: 14px; font-family: var(--mono);
}

/* ── Badges ── */
.badge-high { background:transparent; color:var(--red); padding:4px 14px;
              border:1px solid #7a2030; border-radius:2px; font-weight:600;
              font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; }
.badge-med  { background:transparent; color:var(--amber); padding:4px 14px;
              border:1px solid var(--amber-dim); border-radius:2px; font-weight:600;
              font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; }
.badge-low  { background:transparent; color:var(--green); padding:4px 14px;
              border:1px solid #1a6635; border-radius:2px; font-weight:600;
              font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; }

/* ── Term box ── */
.term-box {
    background: var(--panel); border: 1px solid var(--border);
    border-left: 3px solid var(--cyan); border-radius: 2px;
    padding: 12px 15px; font-size: 0.74rem; color: var(--cyan);
    font-family: var(--mono); letter-spacing: 0.04em;
}
.term-box-amber { border-left-color: var(--amber) !important; color: var(--amber) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important; gap: 0;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important; font-size: 0.7rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    color: var(--text-muted) !important; padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important; border-bottom: 2px solid var(--amber) !important;
    background: var(--amber-glow) !important;
}

/* ── Misc ── */
h1,h2,h3,h4 { font-family: var(--display) !important; letter-spacing:0.04em !important; color: var(--text) !important; }
hr { border-color: var(--border) !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 2px !important; }
.stProgress > div > div > div > div { background: var(--amber) !important; }
.streamlit-expanderHeader {
    background: var(--panel) !important; color: var(--text-dim) !important;
    font-family: var(--mono) !important; font-size: 0.75rem !important;
    letter-spacing: 0.1em !important; border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────
AMBER = '#f5a623'
CYAN  = '#00d4ff'
RED   = '#ff4757'
GREEN = '#2ed573'

def term_fig(figsize=(13, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0e1014')
    ax.set_facecolor('#111318')
    for spine in ax.spines.values():
        spine.set_color('#1e2128'); spine.set_linewidth(0.8)
    ax.tick_params(colors='#5a6175', labelsize=7.5, width=0.6)
    ax.xaxis.label.set_color('#5a6175'); ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_color('#5a6175'); ax.yaxis.label.set_fontsize(8)
    ax.title.set_color('#c8cdd6'); ax.title.set_fontsize(9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily('monospace')
    return fig, ax

def term_figs(n, figsize=(13, 4)):
    fig, axes = plt.subplots(1, n, figsize=figsize, facecolor='#0e1014')
    for ax in axes:
        ax.set_facecolor('#111318')
        for spine in ax.spines.values():
            spine.set_color('#1e2128'); spine.set_linewidth(0.8)
        ax.tick_params(colors='#5a6175', labelsize=7.5, width=0.6)
        ax.xaxis.label.set_color('#5a6175'); ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_color('#5a6175'); ax.yaxis.label.set_fontsize(8)
        ax.title.set_color('#c8cdd6'); ax.title.set_fontsize(9)
    return fig, axes


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATA_PATH = "hourlyLoadDataIndia.csv"
TARGET    = 'National Hourly Demand'
FEATURES  = ['hour', 'day', 'month', 'dayofweek', 'year', 'lag_1', 'lag_24']

# Grid thresholds (MW) — adjust to match your dataset's actual range
PEAK_LIMIT = 230000
HIGH_LIMIT = 190000
LOW_LIMIT  = 120000

MONTHS_LBL = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
DAYS_LBL   = ["MON","TUE","WED","THU","FRI","SAT","SUN"]


# ─────────────────────────────────────────────────────────────
# GRID STATUS HELPER
# ─────────────────────────────────────────────────────────────
def get_grid_status(mw):
    if mw >= PEAK_LIMIT:
        return "!! CRITICAL", "alert-red",    "Extreme stress on grid. Risk of outages."
    elif mw >= HIGH_LIMIT:
        return ">> HIGH",     "alert-yellow", "High demand. Grid under pressure."
    elif mw >= LOW_LIMIT:
        return "-- NORMAL",   "alert-yellow", "Demand within normal operating range."
    else:
        return "OK LOW",      "alert-green",  "Low demand. Grid comfortable."


# ─────────────────────────────────────────────────────────────
# DATA LOAD + FEATURE ENGINEERING  ← your original code
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df               = pd.read_csv(DATA_PATH)
    df['datetime']   = pd.to_datetime(df['datetime'])
    df               = df.sort_values('datetime').reset_index(drop=True)
    df['hour']       = df['datetime'].dt.hour
    df['day']        = df['datetime'].dt.day
    df['month']      = df['datetime'].dt.month
    df['dayofweek']  = df['datetime'].dt.dayofweek
    df['year']       = df['datetime'].dt.year
    df['lag_1']      = df[TARGET].shift(1)
    df['lag_24']     = df[TARGET].shift(24)
    df               = df.dropna().reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# TRAIN MODEL  ← your original Ridge pipeline, untouched
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    df         = load_data()
    X          = df[FEATURES]
    y          = df[TARGET]
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    alphas = [0.01, 0.1, 1, 10, 100]
    model  = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  RidgeCV(alphas=alphas, cv=5))
    ])
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
    mae        = mean_absolute_error(y_test, y_pred)
    r2         = r2_score(y_test, y_pred)
    mape       = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    best_alpha = model.named_steps['ridge'].alpha_

    # Keep test slice with datetime index for time-series plots
    test_slice = df.iloc[train_size:].set_index('datetime')[[TARGET]].copy()
    test_slice['prediction'] = y_pred

    return model, rmse, mae, r2, mape, best_alpha, test_slice, df.iloc[train_size:]


# ─────────────────────────────────────────────────────────────
# ITERATIVE FUTURE PREDICTION HELPER
# Bootstraps lags from last known data, then chains predictions
# ─────────────────────────────────────────────────────────────
def predict_future_hours(model, df, start_dt, n_hours):
    history = list(df[TARGET].values)
    rows    = []
    for i in range(n_hours):
        ts    = start_dt + pd.Timedelta(hours=i)
        lag1  = history[-1]
        lag24 = history[-24] if len(history) >= 24 else np.mean(history)
        row   = {
            'hour':      ts.hour,
            'day':       ts.day,
            'month':     ts.month,
            'dayofweek': ts.dayofweek,
            'year':      ts.year,
            'lag_1':     lag1,
            'lag_24':    lag24,
        }
        yhat = model.predict(pd.DataFrame([row]))[0]
        rows.append({'datetime': ts, TARGET: yhat})
        history.append(yhat)
    return pd.DataFrame(rows).set_index('datetime')


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-top">GRID<span class="sb-logo-slash">//</span>TERMINAL</div>
        <div class="sb-logo-sub"><span class="sb-logo-dot"></span>India Power Grid · Forecast System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">// navigation</div>', unsafe_allow_html=True)

    page = st.radio("_nav", options=[
        "⬡  OVERVIEW",
        "◈  PREDICT & ANALYZE",
        "◉  MULTI-DAY FORECAST",
        "▣  DATA EXPLORER",
        "◫  MODEL INSIGHTS",
    ], label_visibility="hidden")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    try:
        _df_s = pd.read_csv(DATA_PATH, usecols=['datetime', TARGET])
        _df_s['datetime'] = pd.to_datetime(_df_s['datetime'])
        _yr_min = int(_df_s['datetime'].dt.year.min())
        _yr_max = int(_df_s['datetime'].dt.year.max())
        _nrec   = f"{len(_df_s):,}"
        _avg_mw = f"{_df_s[TARGET].mean():,.0f}"
    except Exception:
        _yr_min, _yr_max, _nrec, _avg_mw = "—", "—", "—", "—"

    st.markdown(f"""
    <div class="sb-stats">
        <div class="sb-stats-header">// dataset status</div>
        <div class="sb-stat-row"><span class="sb-stat-key">RECORDS</span>
            <span class="sb-stat-val-green">{_nrec}</span></div>
        <div class="sb-stat-row"><span class="sb-stat-key">PERIOD</span>
            <span class="sb-stat-val-cyan">{_yr_min}–{_yr_max}</span></div>
        <div class="sb-stat-row"><span class="sb-stat-key">AVG LOAD</span>
            <span class="sb-stat-val-amber">{_avg_mw} MW</span></div>
        <div class="sb-stat-row"><span class="sb-stat-key">ALGORITHM</span>
            <span class="sb-stat-val-cyan">RidgeCV</span></div>
        <div class="sb-stat-row"><span class="sb-stat-key">FEATURES</span>
            <span class="sb-stat-val-amber">7</span></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────────────────────
try:
    with st.spinner("// INITIALIZING MODEL..."):
        df = load_data()
        model, rmse, mae, r2, mape, best_alpha, test_result, test_raw = train_model()
except FileNotFoundError:
    st.error(f"// ERROR: `{DATA_PATH}` not found. Place CSV in same directory and restart.")
    st.stop()

MIN_D    = df[TARGET].min()
MAX_D    = df[TARGET].max()
MEAN_D   = df[TARGET].mean()
P25      = df[TARGET].quantile(0.25)
P75      = df[TARGET].quantile(0.75)
HIST_AVG = int(MEAN_D)
LAST_DT  = pd.to_datetime(df['datetime'].max())

_page = page.split("  ", 1)[-1].strip()


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def page_header(title, crumb, badge=None):
    b = f'<span class="page-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div class="page-title">{title}</div>
            <div class="page-breadcrumb">GRID//TERMINAL &rsaquo; {crumb}</div>
        </div>
        {b}
    </div>""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)

def sp(h=16):
    st.markdown(f'<div style="height:{h}px"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW / DASHBOARD
# ══════════════════════════════════════════════════════════════
if _page == "OVERVIEW":
    page_header("SYSTEM OVERVIEW", "Overview", f"R² {r2:.4f}")
    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    # Model metrics row
    sec("// model performance · test set (20%)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("RMSE",       f"{rmse:,.0f} MW")
    k2.metric("MAE",        f"{mae:,.0f} MW")
    k3.metric("R² SCORE",   f"{r2:.4f}")
    k4.metric("MAPE",       f"{mape:.2f}%")
    k5.metric("BEST ALPHA", f"{best_alpha}")

    sp(20)

    # Historical consumption chart
    sec("// historical demand · daily average")
    df_ts = df.set_index('datetime')
    daily = df_ts[TARGET].resample("D").mean()
    fig, ax = term_fig(figsize=(14, 4))
    ax.fill_between(daily.index, daily.values / 1000, alpha=0.1, color=AMBER)
    ax.plot(daily.index, daily.values / 1000, color=AMBER, linewidth=0.9)
    ax.set_ylabel("DEMAND (GW)")
    ax.set_title("DAILY AVERAGE NATIONAL ELECTRICITY DEMAND — INDIA")
    ax.grid(True, alpha=0.06, color='#252a35')
    plt.tight_layout(pad=1.2)
    st.pyplot(fig); plt.close()

    sp(8)

    # Actual vs Predicted
    sec("// actual vs predicted · test set (daily avg)")
    da = test_result[TARGET].resample("D").mean()
    dp = test_result["prediction"].resample("D").mean()
    fig, ax = term_fig(figsize=(14, 4))
    ax.fill_between(da.index, da.values / 1000, alpha=0.05, color=CYAN)
    ax.plot(da.index, da.values / 1000, color=CYAN,  linewidth=1.3, label="ACTUAL")
    ax.plot(dp.index, dp.values / 1000, color=AMBER, linewidth=1.3,
            linestyle="--", label="PREDICTED", alpha=0.9)
    ax.set_ylabel("DEMAND (GW)")
    ax.set_title("ACTUAL VS PREDICTED DAILY AVERAGE — TEST SET")
    ax.legend(facecolor='#111318', labelcolor='#c8cdd6', fontsize=7.5,
              framealpha=1, edgecolor='#1e2128', prop={'family': 'monospace'})
    ax.grid(True, alpha=0.06, color='#252a35')
    plt.tight_layout(pad=1.2)
    st.pyplot(fig); plt.close()

    sp(8)

    # Dataset-level stats
    sec("// dataset statistics")
    d1, d2, d3 = st.columns(3)
    d1.metric("⚡ ALL-TIME PEAK", f"{MAX_D:,.0f} MW")
    d2.metric("📉 ALL-TIME LOW",  f"{MIN_D:,.0f} MW")
    d3.metric("📊 OVERALL AVG",   f"{MEAN_D:,.0f} MW")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT & ANALYZE
# ══════════════════════════════════════════════════════════════
elif _page == "PREDICT & ANALYZE":
    page_header("PREDICT & ANALYZE", "Predict & Analyze", "24-HR PROFILE")
    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    sec("// select future date")
    min_future = (LAST_DT + pd.Timedelta(days=1)).date()

    col1, col2 = st.columns([2, 1])
    with col1:
        predict_date = st.date_input(
            "PREDICTION DATE",
            value=min_future,
            min_value=min_future,
        )
    with col2:
        sp(8)
        st.markdown(
            f'<div class="term-box">// Dataset ends {LAST_DT.date()}<br>'
            f'Pick any future date → get full 24-hr hourly demand prediction.</div>',
            unsafe_allow_html=True)

    sp(12)
    _, bcol, _ = st.columns([1, 2, 1])
    with bcol:
        run_pred = st.button("// GENERATE 24-HOUR FORECAST", use_container_width=True)

    if run_pred:
        start_dt  = pd.Timestamp(str(predict_date))
        future_df = predict_future_hours(model, df, start_dt, 24)
        preds     = future_df[TARGET].values

        avg_mw  = preds.mean()
        peak_mw = preds.max()
        min_mw  = preds.min()
        peak_hr = int(np.argmax(preds))
        low_hr  = int(np.argmin(preds))

        # Historical baseline: same weekday + month
        hist_sub = df[
            (df['datetime'].dt.dayofweek == start_dt.dayofweek) &
            (df['datetime'].dt.month     == start_dt.month)
        ][TARGET].mean()
        diff_pct = ((avg_mw - hist_sub) / hist_sub) * 100 if hist_sub > 0 else 0.0

        # ── KPIs ──
        sp(16); sec("// prediction summary")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("DATE",        str(predict_date))
        k2.metric("AVG MW",      f"{avg_mw:,.0f}")
        k3.metric("PEAK MW",     f"{peak_mw:,.0f}", f"@ {peak_hr:02d}:00")
        k4.metric("MIN MW",      f"{min_mw:,.0f}",  f"@ {low_hr:02d}:00")
        k5.metric("VS HISTORY",  f"{diff_pct:+.1f}%",
                  delta_color="inverse" if diff_pct > 10 else "normal")

        # ── 24-Hour Charts ──
        sp(16); sec("// 24-hour forecast profile")
        fig, axes = term_figs(2, figsize=(14, 4))

        bar_clrs = [RED if v == peak_mw else (CYAN if v == min_mw else AMBER) for v in preds]
        axes[0].bar(range(24), preds / 1000, color=bar_clrs, edgecolor='#111318', alpha=0.85)
        axes[0].axhline(hist_sub / 1000, color='white', linestyle='--', linewidth=0.9,
                        label=f'HIST. AVG: {hist_sub/1000:.1f} GW')
        axes[0].set_xlabel("HOUR"); axes[0].set_ylabel("GW")
        axes[0].set_xticks(range(24))
        axes[0].set_title(f"HOURLY FORECAST — {predict_date}")
        axes[0].legend(facecolor='#111318', labelcolor='#c8cdd6', fontsize=7.5,
                       framealpha=1, edgecolor='#1e2128', prop={'family': 'monospace'})
        axes[0].grid(True, alpha=0.05, axis='y', color='#252a35')

        axes[1].plot(range(24), preds / 1000, color=AMBER, linewidth=2.2,
                     marker='o', markersize=3.5)
        axes[1].fill_between(range(24), preds / 1000, alpha=0.12, color=AMBER)
        axes[1].axhline(hist_sub / 1000, color='white', linestyle='--', linewidth=0.9)
        axes[1].set_xlabel("HOUR"); axes[1].set_ylabel("GW")
        axes[1].set_xticks(range(24))
        axes[1].set_title("CONSUMPTION CURVE")
        axes[1].grid(True, alpha=0.05, axis='y', color='#252a35')

        plt.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()

        # ── Grid Status Alert ──
        sp(16); sec("// grid load status")
        status_lbl, alert_cls, status_msg = get_grid_status(peak_mw)
        st.markdown(f"""
        <div class="{alert_cls}">
            <b style="font-size:1rem">{status_lbl} — PEAK DEMAND: {peak_mw:,.0f} MW</b><br>
            {status_msg}<br><br>
            <span style="font-size:0.75rem;opacity:0.65;">
            !! CRITICAL ≥{PEAK_LIMIT:,} MW &nbsp;|&nbsp;
            >> HIGH {HIGH_LIMIT:,}–{PEAK_LIMIT:,} MW &nbsp;|&nbsp;
            -- NORMAL {LOW_LIMIT:,}–{HIGH_LIMIT:,} MW &nbsp;|&nbsp;
            OK LOW &lt;{LOW_LIMIT:,} MW
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Hourly Status Grid ──
        sp(8); sec("// hourly grid status (all 24 hours)")
        hours_status = []
        for h, v in enumerate(preds):
            lbl, _, _ = get_grid_status(v)
            hours_status.append(f"**{lbl.split()[0]}** {h:02d}:00\n{v/1000:.2f} GW")
        for row_start in [0, 8, 16]:
            cols = st.columns(8)
            for i, col in enumerate(cols):
                col.caption(hours_status[row_start + i])

        # ── Smart Recommendations ──
        sp(16); sec("// smart recommendations")
        sorted_hrs = np.argsort(preds)
        best_3     = sorted_hrs[:3]
        worst_3    = sorted_hrs[-3:][::-1]

        if peak_mw > HIGH_LIMIT:
            st.markdown(
                f'<div class="alert-yellow">⚠️ <b>GRID ALERT:</b> High demand predicted. '
                f'Consider demand response or load-shedding during peak hours.</div>',
                unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
        ⚡ <b>PEAK DEMAND</b> at <b>{peak_hr:02d}:00</b> ({peak_mw:,.0f} MW) —
        avoid heavy industrial and commercial loads at this hour.<br>
        ✅ <b>BEST WINDOW</b> for high-consumption activity: <b>{low_hr:02d}:00</b>
        ({min_mw:,.0f} MW) — lowest demand of the day.<br>
        📊 Predicted average of <b>{avg_mw:,.0f} MW</b> is
        <b>{abs(diff_pct):.1f}% {"above" if diff_pct > 0 else "below"}</b>
        the historical average for this weekday &amp; month.
        </div>
        """, unsafe_allow_html=True)

        ra, rb = st.columns(2)
        with ra:
            st.success("✅ BEST HOURS — LOW DEMAND WINDOW")
            for h in best_3:
                st.markdown(f"&nbsp;&nbsp;**{h:02d}:00** — {preds[h]:,.0f} MW")
        with rb:
            st.error("❌ PEAK HOURS — AVOID HEAVY LOADS")
            for h in worst_3:
                st.markdown(f"&nbsp;&nbsp;**{h:02d}:00** — {preds[h]:,.0f} MW")

        # ── Full hourly table ──
        with st.expander("// FULL HOURLY TABLE"):
            tbl = pd.DataFrame({
                "HOUR":         [f"{h:02d}:00" for h in range(24)],
                "PREDICTED MW": preds.round(0).astype(int),
                "PREDICTED GW": (preds / 1000).round(3),
                "GRID STATUS":  [get_grid_status(v)[0] for v in preds],
                "VS HIST AVG":  [f"{((v - hist_sub)/hist_sub)*100:+.1f}%" for v in preds],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        # ── Download ──
        sp(8)
        out_df = future_df.copy()
        out_df.columns = ["Predicted_MW"]
        out_df["Predicted_GW"] = out_df["Predicted_MW"] / 1000
        out_df["Grid_Status"]  = [get_grid_status(v)[0] for v in preds]
        st.download_button(
            "// EXPORT FORECAST CSV",
            data=out_df.to_csv().encode("utf-8"),
            file_name=f"forecast_{predict_date}.csv",
            mime="text/csv"
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — MULTI-DAY FORECAST
# ══════════════════════════════════════════════════════════════
elif _page == "MULTI-DAY FORECAST":
    page_header("MULTI-DAY FORECAST", "Multi-Day Forecast")
    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    min_future = (LAST_DT + pd.Timedelta(days=1)).date()

    sec("// forecast configuration")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        start_date = st.date_input("FORECAST START DATE", value=min_future, min_value=min_future)
    with col_b:
        n_days = st.slider("FORECAST HORIZON (DAYS)", 1, 90, 14)

    sp(12)
    _, bcol, _ = st.columns([1, 2, 1])
    with bcol:
        run_fc = st.button("// EXECUTE MULTI-DAY FORECAST", use_container_width=True)

    if run_fc:
        start_dt  = pd.Timestamp(str(start_date))
        future_df = predict_future_hours(model, df, start_dt, n_days * 24)

        daily_pred = future_df[TARGET].resample("D").mean()
        daily_peak = future_df[TARGET].resample("D").max()
        daily_min  = future_df[TARGET].resample("D").min()

        # KPIs
        sp(16); sec("// forecast statistics")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("DAYS FORECAST",  n_days)
        f2.metric("PERIOD PEAK MW", f"{future_df[TARGET].max():,.0f}")
        f3.metric("PERIOD AVG MW",  f"{future_df[TARGET].mean():,.0f}")
        f4.metric("PERIOD MIN MW",  f"{future_df[TARGET].min():,.0f}")

        # Daily bar chart
        sp(16); sec("// daily average forecast")
        bar_clrs = [RED if v >= HIGH_LIMIT else (AMBER if v >= LOW_LIMIT else CYAN)
                    for v in daily_pred.values]
        fig, ax = term_fig(figsize=(14, 5))
        ax.bar(daily_pred.index, daily_pred.values / 1000,
               color=bar_clrs, edgecolor='#111318', width=0.7, alpha=0.85)
        ax.plot(daily_pred.index, daily_pred.values / 1000,
                color='white', linewidth=1.2, marker='o', markersize=3.5)
        ax.axhline(HIST_AVG / 1000, color='#555', linestyle='--', linewidth=1,
                   label=f'HIST. AVG: {HIST_AVG/1000:.1f} GW')
        ax.legend(facecolor='#111318', labelcolor='#c8cdd6', fontsize=7.5,
                  framealpha=1, edgecolor='#1e2128', prop={'family': 'monospace'})
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_days // 10)))
        plt.xticks(rotation=45, color='#5a6175')
        ax.set_ylabel("AVG DAILY GW")
        ax.set_title(f"{n_days}-DAY DAILY AVERAGE FORECAST FROM {start_date}")
        ax.grid(True, alpha=0.06, color='#252a35')
        plt.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()

        # Hourly line chart
        sp(8); sec("// hourly detail")
        fig, ax = term_fig(figsize=(14, 4))
        ax.plot(future_df.index, future_df[TARGET] / 1000, color=AMBER, linewidth=0.8)
        ax.fill_between(future_df.index, future_df[TARGET] / 1000, alpha=0.1, color=AMBER)
        ax.axhline(HIST_AVG / 1000, color='#555', linestyle='--', linewidth=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_days // 7)))
        plt.xticks(rotation=45, color='#5a6175')
        ax.set_ylabel("GW")
        ax.set_title("HOURLY FORECAST DETAIL")
        ax.grid(True, alpha=0.06, color='#252a35')
        plt.tight_layout(pad=1.2)
        st.pyplot(fig); plt.close()

        # High-demand alert days
        sp(16); sec("// high demand alert days")
        high_days = daily_peak[daily_peak >= HIGH_LIMIT]
        if len(high_days):
            for d, v in high_days.items():
                lbl, cls, msg = get_grid_status(v)
                st.markdown(
                    f'<div class="alert-yellow">📅 <b>{d.date()}</b> — Peak: {v:,.0f} MW'
                    f' &nbsp;|&nbsp; {lbl} &nbsp;|&nbsp; {msg}</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="alert-green">✅ No high-demand alert days in this forecast period.</div>',
                unsafe_allow_html=True)

        # Daily summary table
        sp(16); sec("// daily summary table")
        summary = pd.DataFrame({
            "DATE":        daily_pred.index.strftime("%Y-%m-%d"),
            "AVG MW":      daily_pred.values.round(0).astype(int),
            "PEAK MW":     daily_peak.values.round(0).astype(int),
            "MIN MW":      daily_min.values.round(0).astype(int),
            "GRID STATUS": [get_grid_status(v)[0] for v in daily_peak.values],
            "VS HIST AVG": [f"{((v - HIST_AVG)/HIST_AVG)*100:+.1f}%" for v in daily_pred.values],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Download
        csv = future_df.to_csv().encode("utf-8")
        st.download_button("// EXPORT HOURLY CSV", data=csv,
                           file_name=f"forecast_{start_date}_{n_days}d.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif _page == "DATA EXPLORER":
    page_header("DATA EXPLORER", "Data Explorer")
    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["⏰ BY HOUR", "📆 BY MONTH", "📈 BY YEAR", "📅 BY WEEKDAY"])

    with tab1:
        h_avg = df.groupby('hour')[TARGET].mean()
        fig, ax = term_fig(figsize=(12, 4))
        ax.bar(h_avg.index, h_avg.values / 1000,
               color=[AMBER if v == h_avg.max() else '#1a2e45' for v in h_avg.values],
               alpha=0.9, edgecolor='none')
        ax.set_xlabel("HOUR OF DAY"); ax.set_ylabel("AVG GW")
        ax.set_title("AVERAGE DEMAND BY HOUR OF DAY")
        ax.grid(True, alpha=0.06, axis='y', color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()
        st.caption(
            f"🔴 Peak hour: **{h_avg.idxmax()}:00** ({h_avg.max():,.0f} MW)"
            f" &nbsp;|&nbsp; Lowest: **{h_avg.idxmin()}:00** ({h_avg.min():,.0f} MW)")

    with tab2:
        m_avg = df.groupby('month')[TARGET].mean()
        fig, ax = term_fig(figsize=(12, 4))
        ax.bar(MONTHS_LBL, m_avg.values / 1000,
               color=[AMBER if v == m_avg.max() else '#1a2e45' for v in m_avg.values],
               alpha=0.9, edgecolor='none')
        ax.set_xlabel("MONTH"); ax.set_ylabel("AVG GW")
        ax.set_title("AVERAGE DEMAND BY MONTH")
        ax.grid(True, alpha=0.06, axis='y', color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()
        st.caption("Highest demand in summer months due to cooling load.")

    with tab3:
        y_avg = df.groupby('year')[TARGET].mean()
        fig, ax = term_fig(figsize=(12, 4))
        ax.plot(y_avg.index, y_avg.values / 1000,
                color=AMBER, linewidth=2.5, marker='o', markersize=8)
        ax.fill_between(y_avg.index, y_avg.values / 1000, alpha=0.1, color=AMBER)
        ax.set_xlabel("YEAR"); ax.set_ylabel("AVG GW")
        ax.set_title("AVERAGE DEMAND BY YEAR")
        ax.grid(True, alpha=0.06, color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()

    with tab4:
        d_avg = df.groupby('dayofweek')[TARGET].mean()
        fig, ax = term_fig(figsize=(12, 4))
        ax.bar(DAYS_LBL, d_avg.values / 1000,
               color=[AMBER] * 5 + [CYAN] * 2,
               alpha=0.9, edgecolor='none', width=0.6)
        ax.set_xlabel("DAY OF WEEK"); ax.set_ylabel("AVG GW")
        ax.set_title("AVERAGE DEMAND BY DAY OF WEEK")
        ax.grid(True, alpha=0.06, axis='y', color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()
        st.caption("🟠 Weekdays — higher demand &nbsp;|&nbsp; 🔵 Weekends — lower demand")

    sp(20)

    # Custom Date Range
    sec("// custom date range explorer")
    df_ts  = df.set_index('datetime')
    r1, r2 = st.columns(2)
    s = r1.date_input("FROM", value=df_ts.index.min().date(),
                      min_value=df_ts.index.min().date(),
                      max_value=df_ts.index.max().date())
    e = r2.date_input("TO",   value=df_ts.index.max().date(),
                      min_value=df_ts.index.min().date(),
                      max_value=df_ts.index.max().date())
    rd = df_ts.loc[str(s):str(e), TARGET].resample("D").mean()
    if len(rd):
        fig, ax = term_fig(figsize=(14, 4))
        ax.fill_between(rd.index, rd.values / 1000, alpha=0.1, color=CYAN)
        ax.plot(rd.index, rd.values / 1000, color=CYAN, linewidth=1.2)
        ax.set_title(f"DAILY AVG MW: {s} → {e}")
        ax.set_ylabel("GW")
        ax.grid(True, alpha=0.06, color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()
        st.caption(
            f"Avg: {rd.mean():,.0f} MW &nbsp;|&nbsp;"
            f" Peak: {rd.max():,.0f} MW &nbsp;|&nbsp;"
            f" Min: {rd.min():,.0f} MW")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
elif _page == "MODEL INSIGHTS":
    page_header("MODEL INSIGHTS", "Model Insights", f"ALPHA={best_alpha}")
    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 FEATURE COEFFICIENTS",
        "📉 RESIDUAL ANALYSIS",
        "🎯 ERROR BREAKDOWN"
    ])

    with tab1:
        sec("// ridge regression feature coefficients")
        coefs    = model.named_steps['ridge'].coef_
        coef_df  = pd.DataFrame({"FEATURE": FEATURES, "COEFFICIENT": coefs}).sort_values("COEFFICIENT")

        # Signed coefficients
        fig, ax = term_fig(figsize=(9, 5))
        clrs = [RED if v < 0 else AMBER for v in coef_df["COEFFICIENT"]]
        ax.barh(coef_df["FEATURE"], coef_df["COEFFICIENT"], color=clrs, alpha=0.85)
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
        ax.set_xlabel("COEFFICIENT VALUE")
        ax.set_title("RIDGE REGRESSION FEATURE COEFFICIENTS")
        ax.grid(True, alpha=0.06, axis='x', color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()
        st.caption("🟠 Positive → increases predicted demand &nbsp;|&nbsp; 🔴 Negative → decreases demand")

        sp(8); sec("// absolute coefficient magnitude  (feature importance proxy)")
        abs_df = coef_df.copy()
        abs_df["ABS"] = abs_df["COEFFICIENT"].abs()
        abs_df = abs_df.sort_values("ABS")
        fig, ax = term_fig(figsize=(9, 5))
        ax.barh(abs_df["FEATURE"], abs_df["ABS"],
                color=[AMBER if v == abs_df["ABS"].max() else CYAN for v in abs_df["ABS"]],
                alpha=0.85)
        ax.set_xlabel("ABSOLUTE COEFFICIENT")
        ax.set_title("FEATURE IMPORTANCE PROXY — ABSOLUTE RIDGE COEFFICIENTS")
        ax.grid(True, alpha=0.06, axis='x', color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()

    with tab2:
        sec("// residual analysis")
        y_true = test_result[TARGET].values
        y_hat  = test_result["prediction"].values
        res    = y_true - y_hat
        dr     = (test_result[TARGET] - test_result["prediction"]).resample("D").mean()

        fig, axes = term_figs(2, figsize=(14, 4))
        axes[0].plot(dr.index, dr.values / 1000, color=CYAN, linewidth=0.8)
        axes[0].axhline(0, color=AMBER, linestyle='--', linewidth=1.5)
        axes[0].set_title("DAILY RESIDUALS OVER TIME")
        axes[0].set_ylabel("ERROR (GW)")
        axes[0].grid(True, alpha=0.06, color='#252a35')

        axes[1].hist(res / 1000, bins=80, color=CYAN, edgecolor='#111318', alpha=0.75)
        axes[1].axvline(0, color=AMBER, linestyle='--', linewidth=1.5)
        axes[1].set_title("RESIDUAL DISTRIBUTION")
        axes[1].set_xlabel("PREDICTION ERROR (GW)")
        axes[1].grid(True, alpha=0.06, color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()

        st.info(
            f"Mean Error: {res.mean():,.1f} MW &nbsp;|&nbsp;"
            f" Std: {res.std():,.1f} MW &nbsp;|&nbsp;"
            f" Max Over-pred: {res.min():,.0f} MW &nbsp;|&nbsp;"
            f" Max Under-pred: {res.max():,.0f} MW")

        sp(8); sec("// scatter: actual vs predicted")
        fig, ax = term_fig(figsize=(8, 5))
        ax.scatter(y_true / 1000, y_hat / 1000, alpha=0.12, s=3, color=CYAN, edgecolors='none')
        lo = min(y_true.min(), y_hat.min()) / 1000 * 0.97
        hi = max(y_true.max(), y_hat.max()) / 1000 * 1.03
        ax.plot([lo, hi], [lo, hi], color=AMBER, linewidth=1.5, linestyle='--', label='PERFECT FIT')
        ax.set_xlabel("ACTUAL (GW)"); ax.set_ylabel("PREDICTED (GW)")
        ax.set_title(f"ACTUAL VS PREDICTED  ·  R² = {r2:.4f}")
        ax.legend(facecolor='#111318', labelcolor='#c8cdd6', fontsize=7.5,
                  framealpha=1, edgecolor='#1e2128', prop={'family': 'monospace'})
        ax.grid(True, alpha=0.06, color='#252a35')
        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()

    with tab3:
        sec("// error breakdown by hour and month")
        tc         = test_raw.copy()
        tc['pred'] = test_result["prediction"].values
        tc['err']  = np.abs(tc[TARGET] - tc['pred'])

        fig, axes = term_figs(2, figsize=(14, 4))

        eh = tc.groupby('hour')['err'].mean()
        axes[0].bar(eh.index, eh.values / 1000, color=AMBER, alpha=0.85)
        axes[0].set_title("AVG PREDICTION ERROR BY HOUR")
        axes[0].set_xlabel("HOUR"); axes[0].set_ylabel("MAE (GW)")
        axes[0].grid(True, alpha=0.06, axis='y', color='#252a35')

        em = tc.groupby('month')['err'].mean()
        axes[1].bar(MONTHS_LBL, em.values / 1000, color=CYAN, alpha=0.85)
        axes[1].set_title("AVG PREDICTION ERROR BY MONTH")
        axes[1].set_xlabel("MONTH"); axes[1].set_ylabel("MAE (GW)")
        axes[1].grid(True, alpha=0.06, axis='y', color='#252a35')

        plt.tight_layout(pad=1.2); st.pyplot(fig); plt.close()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#2e333c;font-size:0.72rem;padding:18px 0;
            border-top:1px solid #1e2128;font-family:monospace;letter-spacing:0.1em;'>
⚡ GRID//TERMINAL &nbsp;·&nbsp; RIDGE REGRESSION (RidgeCV) &nbsp;·&nbsp;
INDIA HOURLY LOAD DATASET &nbsp;·&nbsp; BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)