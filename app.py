"""
S&P 500 Market Regime Dashboard — self-contained, no pkl needed.
Fetches data and retrains models on first load (~60s), then caches for the session.
Deploy to Streamlit Community Cloud with only app.py + requirements.txt.
"""
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from hmmlearn.hmm import GaussianHMM
from pandas_datareader import data as pdr
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

COLORS_K2 = {"Risk-OFF": "#D32F2F", "Risk-ON": "#388E3C"}
COLORS_K3 = {"Bear": "#D32F2F", "Neutral": "#F57C00", "Bull": "#388E3C"}
ALLOC_K2  = {"Risk-ON": 1.0, "Risk-OFF": 0.0}
ALLOC_K3  = {"Bull": 1.0, "Neutral": 0.5, "Bear": 0.0}

st.set_page_config(page_title="Market Regime Dashboard", page_icon="📈", layout="wide")


# ── Data helpers ───────────────────────────────────────────────────────────────
def yf_close(ticker, start, end, name=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(name or ticker)


def fetch_fred(series, start, end):
    s = pdr.DataReader(series, "fred", start, end).iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
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
    out["Log_Return"]  = np.log(out["Close"] / out["Close"].shift(1))
    out["Volatility"]  = out["Log_Return"].rolling(4).std()
    ma10 = out["Close"].rolling(10).mean()
    ma40 = out["Close"].rolling(40).mean()
    out["MA_Crossover"] = (ma10 - ma40) / out["Close"]
    out["VIX_Change"]  = out["VIX"].pct_change()
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
    gmm2, hmm2 = best_gmm(2), best_hmm(2)
    ret = train_features.loc[X_train.index, "Log_Return"]

    def make_map(states, labels):
        means = ret.groupby(states).mean().sort_values()
        return {s: lbl for s, lbl in zip(means.index, labels)}

    return {
        "prep": prep, "X_train": X_train, "train_features": train_features,
        "gmm3": gmm3, "gmm3_map": make_map(gmm3.predict(Xv), ["Bear", "Neutral", "Bull"]),
        "hmm3": hmm3, "hmm3_map": make_map(hmm3.predict(Xv), ["Bear", "Neutral", "Bull"]),
        "gmm2": gmm2, "gmm2_map": make_map(gmm2.predict(Xv), ["Risk-OFF", "Risk-ON"]),
        "hmm2": hmm2, "hmm2_map": make_map(hmm2.predict(Xv), ["Risk-OFF", "Risk-ON"]),
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
    for key in ("gmm2", "gmm3", "hmm2", "hmm3"):
        m      = models[key]
        lbl    = models[f"{key}_map"]
        states = m.predict(X_pred.values)
        proba  = m.predict_proba(X_pred.values)
        out[key] = {
            "K":          2 if key.endswith("2") else 3,
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


# ── Page ───────────────────────────────────────────────────────────────────────
st.title("📈 Market Regime Dashboard")
st.caption(
    f"S&P 500 weekly regime detection · "
    f"Trained {TRAIN_START} to {TRAIN_END} · "
    f"Out-of-sample {TEST_START} to {TEST_END}"
)

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Model", ["GMM", "HMM"], horizontal=True)
    k_choice     = st.radio("Regimes", ["K=2  (Risk-ON / Risk-OFF)", "K=3  (Bull / Neutral / Bear)"])
    st.markdown("---")
    st.markdown(
        "**Features**\n\n"
        "- Log return\n"
        "- 4-week volatility\n"
        "- MA crossover (10w - 40w)\n"
        "- VIX change\n"
        "- Term spread (10y - 2y)"
    )
    st.markdown("---")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

preds  = get_predictions()
models = train_models()
key    = ("gmm" if model_choice == "GMM" else "hmm") + ("2" if "K=2" in k_choice else "3")
active = preds[key]
regime, confidence, K = active["regime"], active["confidence"], active["K"]
colors = COLORS_K2 if K == 2 else COLORS_K3
alloc  = ALLOC_K2  if K == 2 else ALLOC_K3

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
                   mode="lines", name="S&P 500", line=dict(color="black", width=2)),
        row=1, col=1,
    )
    for r, c in colors.items():
        idx = regime == r
        if idx.any():
            fig.add_trace(
                go.Scatter(x=preds["price"].index[idx], y=preds["price"].values[idx],
                           mode="markers", name=r,
                           marker=dict(color=c, size=8, line=dict(width=0.5, color="white"))),
                row=1, col=1,
            )
    fig.add_trace(
        go.Bar(x=confidence.index, y=confidence.values,
               marker=dict(color=[colors[r] for r in regime]),
               showlegend=False, name="Confidence"),
        row=2, col=1,
    )
    fig.add_hline(y=0.8, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", y=1.05))
    fig.update_yaxes(title="S&P 500",    row=1, col=1)
    fig.update_yaxes(title="Confidence", range=[0, 1.05], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**Current call:** {model_choice} (K={K}) classifies the market as "
        f"**{latest_regime}** at {latest_conf:.1%} confidence, "
        f"holding for {weeks_in} consecutive week(s)."
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
                fillcolor=colors[prev_reg], opacity=0.18,
                layer="below", line_width=0,
            )
            prev_date, prev_reg = bt.index[i], cur_reg

    # Strategy & B&H (left axis)
    fig.add_trace(
        go.Scatter(x=bt.index, y=bt["cum_strat"], mode="lines",
                   name="Strategy", line=dict(color="#1565C0", width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=bt.index, y=bt["cum_bnh"], mode="lines",
                   name="Buy & Hold", line=dict(color="black", width=2, dash="dash")),
        secondary_y=False,
    )

    # S&P 500 price (right axis)
    price_bt = preds["price"].reindex(bt.index)
    fig.add_trace(
        go.Scatter(x=price_bt.index, y=price_bt.values, mode="lines",
                   name="S&P 500", line=dict(color="gray", width=1, dash="dot"), opacity=0.5),
        secondary_y=True,
    )

    # End-value labels
    final_strat = bt["cum_strat"].iloc[-1]
    final_bnh   = bt["cum_bnh"].iloc[-1]
    last_date   = bt.index[-1]
    fig.add_annotation(x=last_date, y=final_bnh,   text=f"${final_bnh:.3f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color="black", size=12))
    fig.add_annotation(x=last_date, y=final_strat, text=f"${final_strat:.3f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color="#1565C0", size=12))

    # Regime colour legend patches
    for lbl, col in colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=col, symbol="square", opacity=0.4),
            name=lbl, showlegend=True,
        ))

    fig.update_layout(
        title=(
            f"OUT-OF-SAMPLE Backtest -- {TEST_START} to {TEST_END}<br>"
            f"<sup>{model_choice} K={K} OOS ({TEST_START} to {TEST_END})</sup>"
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
                                name="Strategy",   line=dict(color="#1565C0"), fill="tozeroy"))
    fig_dd.add_trace(go.Scatter(x=dd_bnh.index,   y=dd_bnh   * 100, mode="lines",
                                name="Buy & Hold", line=dict(color="gray")))
    fig_dd.update_layout(title="Drawdown (%)", yaxis_title="Drawdown (%)",
                         height=300, hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Performance Metrics")
    rows = [
        ["CAGR",                 f"{metrics['Strategy CAGR']:.2%}",   f"{metrics['Buy & Hold CAGR']:.2%}"],
        ["Annualised Volatility", f"{metrics['Strategy Vol']:.2%}",    f"{metrics['Buy & Hold Vol']:.2%}"],
        ["Sharpe Ratio",          f"{metrics['Strategy Sharpe']:.3f}", f"{metrics['Buy & Hold Sharpe']:.3f}"],
        ["Max Drawdown",          f"{metrics['Strategy Max DD']:.2%}", f"{metrics['Buy & Hold Max DD']:.2%}"],
        ["Regime Switches",       f"{metrics['Switches']}",            "--"],
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
        labels  = ["Risk-OFF", "Risk-ON"] if K == 2 else ["Bear", "Neutral", "Bull"]
        lbl_map = models[f"{key}_map"]
        order   = sorted(lbl_map, key=lambda s: labels.index(lbl_map[s]))
        trans   = m.transmat_[np.ix_(order, order)]

        st.subheader("HMM Transition Matrix")
        fig = go.Figure(data=go.Heatmap(
            z=trans, x=[f"-> {l}" for l in labels], y=labels,
            colorscale="Blues", zmin=0, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in trans],
            texttemplate="%{text}", textfont={"size": 16},
        ))
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(K)
        for col, lbl, p in zip(cols, labels, np.diag(trans)):
            col.metric(f"Expected dwell -- {lbl}", f"{1 / (1 - p + 1e-10):.1f} weeks")
    else:
        st.info("Transition matrix is meaningful only for HMM -- switch to HMM in the sidebar.")

    st.subheader("Regime Distribution (OOS)")
    dist = regime.value_counts(normalize=True).mul(100).round(1)
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=[colors[r] for r in dist.index],
        text=[f"{v}%" for v in dist.values], textposition="outside",
    ))
    fig.update_layout(yaxis_title="% of weeks", height=350,
                      yaxis=dict(range=[0, max(dist.values) * 1.2]))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mean Feature Values per Regime (Training)")
    X_tr         = models["X_train"]
    train_states = models[key].predict(X_tr.values)
    train_regime = pd.Series(train_states, index=X_tr.index).map(models[f"{key}_map"])
    feat_means   = (
        X_tr.assign(Regime=train_regime)
            .groupby("Regime")[FEATURE_COLS]
            .mean().round(3)
    )
    st.dataframe(feat_means, use_container_width=True)

st.caption(
    "First load trains 4 models from scratch (~60s). "
    "Models are cached for the session. Click Refresh data in the sidebar for latest prices."
)
