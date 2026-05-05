"""
S&P 500 Market Regime Dashboard — Bloomberg/finance-pro dark theme.
Self-contained; fetches data and retrains models on first load (~60 s).
Deploy to Streamlit Community Cloud with only app.py + requirements.txt.
"""
import warnings
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from hmmlearn.hmm import GaussianHMM
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2024-12-31"
TEST_START   = "2025-01-01"
TEST_END     = date.today().strftime("%Y-%m-%d")
PRED_START   = (pd.Timestamp(TEST_START) - relativedelta(months=12)).strftime("%Y-%m-%d")
FEATURE_COLS = ["Log_Return", "Volatility", "MA_Crossover", "VIX_Change", "term_spread"]
COST_BPS     = 5 / 10_000

# ── Theme ──────────────────────────────────────────────────────────────────────
BG       = "#0B1929"
SURFACE  = "#152844"
SURFACE2 = "#1B3355"
RULE     = "#27406B"
TEXT     = "#E8EEF6"
SUBTEXT  = "#8FA4BF"
GOLD     = "#E5B53D"
CYAN     = "#4FC3F7"
GREEN    = "#4CAF50"
ORANGE   = "#FFB74D"
RED      = "#EF5350"

COLORS_K3 = {"Bear": RED, "Neutral": ORANGE, "Bull": GREEN}
ALLOC_K3  = {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}

st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Bloomberg-style dark UI ───────────────────────────────────────
st.markdown(f"""
<style>
    /* App background */
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    /* Top gold accent bar */
    .stApp > header {{
        background-color: {BG};
        border-bottom: 3px solid {GOLD};
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {RULE};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT};
    }}
    /* Headings */
    h1, h2, h3, h4 {{
        color: {TEXT} !important;
        font-family: "Calibri", sans-serif;
    }}
    h1 {{
        border-left: 5px solid {GOLD};
        padding-left: 14px;
        margin-bottom: 0.2rem;
    }}
    /* Caption / small text */
    .stCaption, p, .stMarkdown {{
        color: {SUBTEXT};
    }}
    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {RULE};
        border-left: 4px solid {GOLD};
        padding: 14px 18px;
        border-radius: 6px;
    }}
    div[data-testid="stMetric"] label {{
        color: {SUBTEXT} !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {GOLD} !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricDelta"] {{
        color: {CYAN} !important;
    }}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0px;
        border-bottom: 1px solid {RULE};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {SUBTEXT};
        border-radius: 0;
        padding: 12px 24px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        color: {GOLD} !important;
        border-bottom: 3px solid {GOLD} !important;
        background-color: {SURFACE};
    }}
    /* Buttons */
    .stButton button {{
        background-color: {GOLD};
        color: {BG};
        border: none;
        font-weight: 700;
        border-radius: 4px;
    }}
    .stButton button:hover {{
        background-color: #c89a25;
        color: {BG};
    }}
    /* Dataframe */
    .stDataFrame {{
        background-color: {SURFACE};
        border: 1px solid {RULE};
        border-radius: 4px;
    }}
    /* Info / alert boxes */
    div[data-testid="stAlert"] {{
        background-color: {SURFACE2};
        border-left: 4px solid {CYAN};
        color: {TEXT};
    }}
    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background-color: {GOLD};
    }}
    /* Radio */
    .stRadio label {{
        color: {TEXT} !important;
    }}
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ───────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT, family="Calibri"),
        title=dict(font=dict(color=TEXT, size=14)),
        xaxis=dict(gridcolor=RULE, zerolinecolor=RULE, color=SUBTEXT),
        yaxis=dict(gridcolor=RULE, zerolinecolor=RULE, color=SUBTEXT),
        legend=dict(bgcolor=SURFACE, bordercolor=RULE, borderwidth=1,
                    font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor=SURFACE2, font=dict(color=TEXT)),
        margin=dict(l=60, r=20, t=60, b=50),
    )
)
pio.templates["bloomberg"] = PLOTLY_TEMPLATE
pio.templates.default = "bloomberg"


# ── Data helpers ───────────────────────────────────────────────────────────────
def yf_close(ticker, start, end, name=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name or ticker)


def fetch_fred(series, start, end):
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&observation_start={start}&observation_end={end}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
    return s.rename(series)


def merge_backward(base, other, name, tolerance="7 days"):
    base, other = base.copy(), other.copy()
    idx_name = base.index.name or "Date"
    base.index.name = other.index.name = idx_name
    a = base.reset_index().sort_values(idx_name)
    b = other.reset_index().sort_values(idx_name)
    b.columns = [idx_name, name]
    out = pd.merge_asof(a, b, on=idx_name, direction="backward",
                        tolerance=pd.Timedelta(tolerance))
    return out.set_index(idx_name)


@st.cache_data(ttl=86400, show_spinner=False)
def build_dataset(start, end):
    spy = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)

    vix = yf_close("^VIX", start, end, name="VIX")
    spy = spy.join(vix, how="inner")

    weekly = spy.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum", "VIX": "last",
    }).dropna()
    weekly.index.name = "Date"

    try:
        term = fetch_fred("T10Y2Y", start, end).rename("term_spread")
    except Exception:
        t10 = yf_close("^TNX", start, end)
        t3m = yf_close("^IRX", start, end)
        term = (t10 - t3m).rename("term_spread")

    weekly = merge_backward(weekly, term, "term_spread")
    return weekly.replace([np.inf, -np.inf], np.nan).dropna()


def compute_features(weekly):
    out = weekly.copy()
    out["Log_Return"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["Volatility"]   = out["Log_Return"].rolling(4).std()
    ma10 = out["Close"].rolling(10).mean()
    ma40 = out["Close"].rolling(40).mean()
    out["MA_Crossover"] = (ma10 - ma40) / out["Close"]
    out["VIX_Change"]   = out["VIX"].pct_change()
    return out[FEATURE_COLS + ["Close"]].replace([np.inf, -np.inf], np.nan).dropna()


class Preprocessor:
    def __init__(self, lower_q=0.01, upper_q=0.99, smooth_window=3):
        self.lower_q, self.upper_q, self.smooth_window = lower_q, upper_q, smooth_window
        self.scaler = StandardScaler()

    def fit(self, X):
        df = X[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        self.lower_ = df.quantile(self.lower_q)
        self.upper_ = df.quantile(self.upper_q)
        self.feature_names_ = list(df.columns)
        clipped  = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window, min_periods=self.smooth_window).mean().dropna()
        self.scaler.fit(smoothed.values)
        return self

    def transform(self, X):
        df = X[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        clipped  = df.clip(self.lower_, self.upper_, axis=1)
        smoothed = clipped.rolling(self.smooth_window, min_periods=self.smooth_window).mean().dropna()
        return pd.DataFrame(self.scaler.transform(smoothed.values),
                            index=smoothed.index, columns=self.feature_names_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ── Train all 4 models (cached for the full session) ──────────────────────────
@st.cache_resource(show_spinner="Fetching data and training models — only happens once per session...")
def train_models():
    train_df       = build_dataset(TRAIN_START, TRAIN_END)
    train_features = compute_features(train_df)
    prep           = Preprocessor()
    X_train        = prep.fit_transform(train_features[FEATURE_COLS])
    Xv             = X_train.values

    def best_gmm(K, seeds=5):
        best, best_bic = None, np.inf
        for s in range(seeds):
            m = GaussianMixture(n_components=K, covariance_type="full",
                                reg_covar=1e-3, n_init=10, random_state=s).fit(Xv)
            if m.bic(Xv) < best_bic:
                best, best_bic = m, m.bic(Xv)
        return best

    def best_hmm(K, seeds=10):
        best, best_ll = None, -np.inf
        for s in range(seeds):
            m = GaussianHMM(n_components=K, covariance_type="full",
                            n_iter=200, tol=1e-4, random_state=s).fit(Xv)
            if m.score(Xv) > best_ll:
                best, best_ll = m, m.score(Xv)
        return best

    gmm3, hmm3 = best_gmm(3), best_hmm(3)
    ret = train_features.loc[X_train.index, "Log_Return"]

    def make_map(states, labels):
        means = ret.groupby(states).mean().sort_values()
        return {s: lbl for s, lbl in zip(means.index, labels)}

    return {
        "prep": prep, "X_train": X_train, "train_features": train_features,
        "gmm3": gmm3, "gmm3_map": make_map(gmm3.predict(Xv), ["Bear", "Neutral", "Bull"]),
        "hmm3": hmm3, "hmm3_map": make_map(hmm3.predict(Xv), ["Bear", "Neutral", "Bull"]),
    }


@st.cache_data(ttl=86400, show_spinner="Loading out-of-sample predictions...")
def get_predictions():
    models        = train_models()
    pred_df       = build_dataset(PRED_START, TEST_END)
    pred_features = compute_features(pred_df)
    X_pred        = models["prep"].transform(pred_features[FEATURE_COLS])
    X_pred        = X_pred[X_pred.index >= TEST_START]

    out = {
        "price":   pred_features.loc[X_pred.index, "Close"],
        "log_ret": pred_features.loc[X_pred.index, "Log_Return"],
        "X_train": models["X_train"],
    }
    for key in ("gmm3", "hmm3"):
        m      = models[key]
        lbl    = models[f"{key}_map"]
        states = m.predict(X_pred.values)
        proba  = m.predict_proba(X_pred.values)
        out[key] = {
            "K":          3,
            "regime":     pd.Series(states, index=X_pred.index).map(lbl),
            "confidence": pd.Series(proba.max(axis=1), index=X_pred.index),
        }
    return out


# ── Backtest ───────────────────────────────────────────────────────────────────
def run_backtest(regime_series, alloc_map, weekly_returns):
    df = pd.DataFrame({"log_ret": weekly_returns, "regime": regime_series}).dropna()
    df["signal"]     = df["regime"].shift(1)
    df["allocation"] = df["signal"].map(alloc_map).fillna(0)
    df["strat_ret"]  = df["allocation"] * df["log_ret"]
    df["switched"]   = (df["signal"] != df["signal"].shift(1)).astype(int)
    df["strat_ret"] -= df["switched"] * COST_BPS
    df["cum_bnh"]    = df["log_ret"].cumsum().apply(np.exp)
    df["cum_strat"]  = df["strat_ret"].cumsum().apply(np.exp)
    return df


def perf_metrics(df):
    years  = max(len(df) / 52, 1e-9)
    cagr_s = df["cum_strat"].iloc[-1] ** (1 / years) - 1
    cagr_b = df["cum_bnh"].iloc[-1]   ** (1 / years) - 1
    vol_s  = df["strat_ret"].std() * np.sqrt(52)
    vol_b  = df["log_ret"].std()   * np.sqrt(52)
    def mdd(c): return ((c - c.cummax()) / c.cummax()).min()
    return {
        "Strategy CAGR":     cagr_s,  "Buy & Hold CAGR":   cagr_b,
        "Strategy Vol":      vol_s,   "Buy & Hold Vol":    vol_b,
        "Strategy Sharpe":   cagr_s / vol_s if vol_s > 0 else 0,
        "Buy & Hold Sharpe": cagr_b / vol_b if vol_b > 0 else 0,
        "Strategy Max DD":   mdd(df["cum_strat"]),
        "Buy & Hold Max DD": mdd(df["cum_bnh"]),
        "Switches":          int(df["switched"].sum()),
    }


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:space-between;
                padding: 6px 0 14px 0;">
      <div>
        <div style="color:{GOLD}; font-size:11px; font-weight:700;
                    letter-spacing:1px; text-transform:uppercase;">
          MARKET REGIME DETECTION  ·  CAPSTONE LIVE DASHBOARD
        </div>
        <div style="color:{TEXT}; font-size:34px; font-weight:700;
                    border-left: 5px solid {GOLD}; padding-left:14px; margin-top:6px;">
          S&P 500 · Regime Dashboard
        </div>
        <div style="color:{SUBTEXT}; font-size:13px; margin-top:6px;">
          Trained {TRAIN_START} → {TRAIN_END}  ·  Out-of-sample {TEST_START} → {TEST_END}
        </div>
      </div>
    </div>
    <hr style="border:none; border-top:1px solid {RULE}; margin: 4px 0 18px 0;">
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        f"<div style='color:{GOLD}; font-weight:700; font-size:11px;"
        f" letter-spacing:1px;'>CONTROLS</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<h3 style='color:{TEXT}; margin-top:0;'>Settings</h3>",
                unsafe_allow_html=True)
    model_choice = st.radio("Model", ["GMM", "HMM"], horizontal=True)
    st.markdown(
        f"<div style='color:{SUBTEXT}; font-size:11px; margin-top:6px;'>"
        f"Regimes: <b style='color:{GOLD};'>K = 3</b>  ·  Bull / Neutral / Bear"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{RULE}'>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='color:{GOLD}; font-weight:700; font-size:10px;"
        f" letter-spacing:1px;'>FEATURES</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style='color:{TEXT}; font-size:12.5px; line-height:1.7;'>
        ▪ Log Return<br>
        ▪ 4-week Volatility<br>
        ▪ MA Crossover (10w − 40w)<br>
        ▪ VIX Change<br>
        ▪ Term spread (10y − 2y)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{RULE}'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

preds  = get_predictions()
models = train_models()
key    = "gmm3" if model_choice == "GMM" else "hmm3"
active = preds[key]
regime, confidence, K = active["regime"], active["confidence"], active["K"]
colors = COLORS_K3
alloc  = ALLOC_K3

# ── KPI strip ──────────────────────────────────────────────────────────────────
latest_regime = regime.iloc[-1]
latest_conf   = confidence.iloc[-1]
weeks_in = 1
for r in regime.iloc[::-1][1:]:
    if r == latest_regime: weeks_in += 1
    else: break

bt      = run_backtest(regime, alloc, preds["log_ret"])
metrics = perf_metrics(bt)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Regime",  latest_regime)
c2.metric("Confidence",      f"{latest_conf:.1%}")
c3.metric("Weeks in regime", weeks_in)
c4.metric(
    "Strategy CAGR (OOS)",
    f"{metrics['Strategy CAGR']:.2%}",
    delta=f"{(metrics['Strategy CAGR'] - metrics['Buy & Hold CAGR']) * 100:+.2f}pp vs B&H",
)

tab_state, tab_bt, tab_table, tab_diag = st.tabs(
    ["📊 Current State", "💰 Backtest", "📋 Predictions", "🔬 Diagnostics"]
)

# ── Tab 1: Current State ───────────────────────────────────────────────────────
with tab_state:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.08,
        subplot_titles=("S&P 500 with Regime Shading", "Model Confidence (per week)"),
    )
    prev_date, prev_reg = preds["price"].index[0], regime.iloc[0]
    for i in range(1, len(regime)):
        if regime.iloc[i] != prev_reg or i == len(regime) - 1:
            fig.add_vrect(
                x0=prev_date, x1=preds["price"].index[i],
                fillcolor=colors[prev_reg], opacity=0.18,
                layer="below", line_width=0, row=1, col=1,
            )
            prev_date, prev_reg = preds["price"].index[i], regime.iloc[i]
    fig.add_trace(
        go.Scatter(x=preds["price"].index, y=preds["price"].values,
                   mode="lines", name="S&P 500", line=dict(color=GOLD, width=2)),
        row=1, col=1,
    )
    for r, c in colors.items():
        idx = regime == r
        if idx.any():
            fig.add_trace(
                go.Scatter(x=preds["price"].index[idx], y=preds["price"].values[idx],
                           mode="markers", name=r,
                           marker=dict(color=c, size=9, line=dict(width=0.6, color=BG))),
                row=1, col=1,
            )
    fig.add_trace(
        go.Bar(x=confidence.index, y=confidence.values,
               marker=dict(color=[colors[r] for r in regime]),
               showlegend=False, name="Confidence"),
        row=2, col=1,
    )
    fig.add_hline(y=0.8, line_dash="dash", line_color=SUBTEXT, row=2, col=1)
    fig.update_layout(height=700, hovermode="x unified",
                      legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title="S&P 500",    row=1, col=1)
    fig.update_yaxes(title="Confidence", range=[0, 1.05], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="background:{SURFACE2}; border-left:4px solid {CYAN};
                    padding:14px 18px; border-radius:4px; color:{TEXT};
                    margin-top:6px;">
          <b style="color:{CYAN};">CURRENT CALL</b> &nbsp; · &nbsp;
          {model_choice} (K={K}) classifies the market as
          <b style="color:{colors[latest_regime]};">{latest_regime}</b>
          at <b>{latest_conf:.1%}</b> confidence,
          holding for <b>{weeks_in}</b> consecutive week(s).
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Tab 2: Backtest ────────────────────────────────────────────────────────────
with tab_bt:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Regime shading
    reg_series = bt["regime"]
    prev_date, prev_reg = bt.index[0], reg_series.iloc[0]
    for i in range(1, len(reg_series)):
        cur_reg = reg_series.iloc[i]
        if cur_reg != prev_reg or i == len(reg_series) - 1:
            fig.add_vrect(
                x0=str(prev_date), x1=str(bt.index[i]),
                fillcolor=colors[prev_reg], opacity=0.16,
                layer="below", line_width=0,
            )
            prev_date, prev_reg = bt.index[i], cur_reg

    # Strategy & B&H (left axis)
    fig.add_trace(
        go.Scatter(x=bt.index, y=bt["cum_strat"], mode="lines",
                   name="Strategy", line=dict(color=GOLD, width=2.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=bt.index, y=bt["cum_bnh"], mode="lines",
                   name="Buy & Hold", line=dict(color=SUBTEXT, width=2, dash="dash")),
        secondary_y=False,
    )

    # S&P 500 price (right axis)
    price_bt = preds["price"].reindex(bt.index)
    fig.add_trace(
        go.Scatter(x=price_bt.index, y=price_bt.values, mode="lines",
                   name="S&P 500", line=dict(color=CYAN, width=1, dash="dot"),
                   opacity=0.55),
        secondary_y=True,
    )

    # End-value labels
    final_strat = bt["cum_strat"].iloc[-1]
    final_bnh   = bt["cum_bnh"].iloc[-1]
    last_date   = bt.index[-1]
    fig.add_annotation(x=last_date, y=final_bnh,   text=f"${final_bnh:.3f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=SUBTEXT, size=12))
    fig.add_annotation(x=last_date, y=final_strat, text=f"${final_strat:.3f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=GOLD, size=12, family="Calibri"))

    # Regime colour legend patches
    for lbl, col in colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=col, symbol="square", opacity=0.5),
            name=lbl, showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=(f"<b>OUT-OF-SAMPLE BACKTEST</b>  ·  {TEST_START} → {TEST_END}"
                  f"<br><span style='font-size:11px; color:{SUBTEXT};'>"
                  f"{model_choice} K={K}  ·  Regime-aware allocation vs Buy & Hold</span>"),
            font=dict(color=GOLD),
        ),
        xaxis_title="Date", height=520, hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_yaxes(title_text="Portfolio Value ($1 invested)", secondary_y=False)
    fig.update_yaxes(title_text="S&P 500 Price", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    dd_strat = (bt["cum_strat"] - bt["cum_strat"].cummax()) / bt["cum_strat"].cummax()
    dd_bnh   = (bt["cum_bnh"]   - bt["cum_bnh"].cummax())   / bt["cum_bnh"].cummax()
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat * 100, mode="lines",
                                name="Strategy",   line=dict(color=GOLD, width=2),
                                fill="tozeroy", fillcolor="rgba(229,181,61,0.2)"))
    fig_dd.add_trace(go.Scatter(x=dd_bnh.index,   y=dd_bnh   * 100, mode="lines",
                                name="Buy & Hold", line=dict(color=SUBTEXT, dash="dash")))
    fig_dd.update_layout(title=dict(text="<b>Drawdown (%)</b>", font=dict(color=GOLD)),
                         yaxis_title="Drawdown (%)", height=300, hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown(f"<h3 style='color:{GOLD};'>Performance Metrics</h3>",
                unsafe_allow_html=True)
    rows = [
        ["CAGR",                  f"{metrics['Strategy CAGR']:.2%}",   f"{metrics['Buy & Hold CAGR']:.2%}"],
        ["Annualised Volatility", f"{metrics['Strategy Vol']:.2%}",    f"{metrics['Buy & Hold Vol']:.2%}"],
        ["Sharpe Ratio",          f"{metrics['Strategy Sharpe']:.3f}", f"{metrics['Buy & Hold Sharpe']:.3f}"],
        ["Max Drawdown",          f"{metrics['Strategy Max DD']:.2%}", f"{metrics['Buy & Hold Max DD']:.2%}"],
        ["Regime Switches",       f"{metrics['Switches']}",            "—"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Strategy", "Buy & Hold"]),
                 use_container_width=True, hide_index=True)

# ── Tab 3: Predictions ─────────────────────────────────────────────────────────
with tab_table:
    n = st.slider("Show last N weeks", 5, len(regime), min(20, len(regime)))
    table = pd.DataFrame({
        "Price":         preds["price"].round(2),
        "Regime":        regime,
        "Confidence":    confidence.round(3),
        "Weekly Return": preds["log_ret"].apply(lambda x: f"{x:+.2%}"),
    }).tail(n).iloc[::-1]
    st.dataframe(table, use_container_width=True)
    csv = table.to_csv().encode("utf-8")
    st.download_button("📥 Download CSV", csv,
                       file_name=f"regime_predictions_{model_choice}_K{K}.csv",
                       mime="text/csv")

# ── Tab 4: Diagnostics ─────────────────────────────────────────────────────────
with tab_diag:
    if model_choice == "HMM":
        m       = models[key]
        labels  = ["Bear", "Neutral", "Bull"]
        lbl_map = models[f"{key}_map"]
        order   = sorted(lbl_map, key=lambda s: labels.index(lbl_map[s]))
        trans   = m.transmat_[np.ix_(order, order)]

        st.markdown(f"<h3 style='color:{GOLD};'>HMM Transition Matrix</h3>",
                    unsafe_allow_html=True)
        fig = go.Figure(data=go.Heatmap(
            z=trans, x=[f"→ {l}" for l in labels], y=labels,
            colorscale=[[0, SURFACE], [0.5, "#3D7BB8"], [1, GOLD]],
            zmin=0, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in trans],
            texttemplate="%{text}",
            textfont={"size": 16, "color": TEXT, "family": "Calibri"},
            colorbar=dict(tickfont=dict(color=SUBTEXT)),
        ))
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(K)
        for col, lbl, p in zip(cols, labels, np.diag(trans)):
            col.metric(f"Expected dwell · {lbl}", f"{1 / (1 - p + 1e-10):.1f} weeks")
    else:
        st.info("Transition matrix is meaningful only for HMM — switch to HMM in the sidebar.")

    st.markdown(f"<h3 style='color:{GOLD};'>Regime Distribution (OOS)</h3>",
                unsafe_allow_html=True)
    dist = regime.value_counts(normalize=True).mul(100).round(1)
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=[colors[r] for r in dist.index],
        marker_line_color=BG, marker_line_width=2,
        text=[f"{v}%" for v in dist.values], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(yaxis_title="% of weeks", height=350,
                      yaxis=dict(range=[0, max(dist.values) * 1.2]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<h3 style='color:{GOLD};'>Mean Feature Values per Regime (Training)</h3>",
                unsafe_allow_html=True)
    X_tr         = models["X_train"]
    train_states = models[key].predict(X_tr.values)
    train_regime = pd.Series(train_states, index=X_tr.index).map(models[f"{key}_map"])
    feat_means   = (
        X_tr.assign(Regime=train_regime)
            .groupby("Regime")[FEATURE_COLS]
            .mean().round(3)
    )
    st.dataframe(feat_means, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <hr style="border:none; border-top:1px solid {RULE}; margin-top:32px;">
    <div style="display:flex; justify-content:space-between; padding:8px 0;
                color:{SUBTEXT}; font-size:10px; font-weight:700;
                letter-spacing:0.6px;">
      <div>MARKET REGIME DETECTION  ·  CAPSTONE LIVE DASHBOARD</div>
      <div style="color:{GOLD};">FIRST LOAD ~60s  ·  CACHED PER SESSION</div>
    </div>
    """,
    unsafe_allow_html=True,
)
