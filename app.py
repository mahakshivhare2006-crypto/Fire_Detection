"""
EmberWatch - Fire & Smoke Detection Dashboard
=============================================
Streamlit app that serves the Logistic Regression pipeline trained in
Fire_Smoke_Detection.ipynb (StandardScaler + LogisticRegression).

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io
import re
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EmberWatch | Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_FILES = ["Fire_Detection.pki", "Fire_Detection.pkl", "model/Fire_Detection.pkl"]
DATA_FILES = ["fire_smoke_detection_logistic_regression_dataset.csv", "data/fire_smoke_detection_logistic_regression_dataset.csv"]
TRAINED_SKLEARN = "1.6.1"
MAX_UPLOAD_ROWS = 50_000

# ----------------------------------------------------------------------------
# Feature definitions
#   hard  -> physically impossible outside this range (input is rejected)
#   train -> range seen during training (outside = extrapolation warning)
#   ui    -> slider range
# ----------------------------------------------------------------------------
FEATURES: dict[str, dict] = {
    "Temperature_C": dict(label="Temperature", unit="°C", icon="🌡️", hard=(-50.0, 150.0), train=(-5.0, 111.08), ui=(-10.0, 120.0), step=0.5,
                          help="Air temperature at the sensor."),
    "Humidity_pct": dict(label="Humidity", unit="%", icon="💧", hard=(0.0, 100.0), train=(0.0, 100.0), ui=(0.0, 100.0), step=0.5,
                         help="Relative humidity. Fires tend to dry the air."),
    "Air_Quality_Index": dict(label="Air Quality Index", unit="AQI", icon="🌫️", hard=(0.0, 1000.0), train=(0.0, 500.0), ui=(0.0, 500.0), step=1.0,
                              help="AQI reading. Smoke pushes this up quickly."),
    "Oxygen_pct": dict(label="Oxygen", unit="%", icon="🫁", hard=(0.0, 30.0), train=(15.16, 23.0), ui=(15.0, 23.0), step=0.05,
                       help="Oxygen concentration. Normal air is about 20.9%."),
    "Wind_Speed_m_s": dict(label="Wind speed", unit="m/s", icon="💨", hard=(0.0, 80.0), train=(0.0, 10.0), ui=(0.0, 15.0), step=0.1,
                           help="Wind speed near the sensor."),
    "Sensor_Distance_m": dict(label="Distance from sensor", unit="m", icon="📏", hard=(0.0, 500.0), train=(0.5, 10.0), ui=(0.0, 15.0), step=0.1,
                              help="Distance between sensor and the suspected source."),
}

PRESETS: dict[str, dict] = {
    "🏠 Normal room": dict(Temperature_C=24.0, Humidity_pct=50.0, Air_Quality_Index=60.0, Oxygen_pct=20.9, Wind_Speed_m_s=2.0, Sensor_Distance_m=5.0),
    "🍳 Cooking": dict(Temperature_C=38.0, Humidity_pct=45.0, Air_Quality_Index=130.0, Oxygen_pct=20.7, Wind_Speed_m_s=1.5, Sensor_Distance_m=3.0),
    "🔥 Developing fire": dict(Temperature_C=52.0, Humidity_pct=32.0, Air_Quality_Index=190.0, Oxygen_pct=20.0, Wind_Speed_m_s=4.0, Sensor_Distance_m=4.0),
    "🚨 Severe fire": dict(Temperature_C=85.0, Humidity_pct=18.0, Air_Quality_Index=320.0, Oxygen_pct=18.5, Wind_Speed_m_s=6.0, Sensor_Distance_m=2.0),
    "🌆 Smog day": dict(Temperature_C=33.0, Humidity_pct=60.0, Air_Quality_Index=280.0, Oxygen_pct=20.8, Wind_Speed_m_s=3.0, Sensor_Distance_m=6.0),
}

RISK_LEVELS = [
    # (min probability, name, css class, colour, advice)
    (0.80, "Critical", "critical", "#FF4D2E", [
        "Treat this as a real fire. Evacuate and raise the alarm now.",
        "Call the fire service (India: 101, or 112 for any emergency).",
        "Do not use lifts. Stay low if there is smoke.",
    ]),
    (0.50, "Warning", "warning", "#FF9F1C", [
        "Send someone to inspect the area immediately.",
        "Keep exits clear and fire extinguishers within reach.",
        "Escalate to Critical if readings keep rising.",
    ]),
    (0.20, "Watch", "watch", "#F6D55C", [
        "Readings are above normal. Keep monitoring.",
        "Check for cooking, heaters, vehicles or other benign sources.",
    ]),
    (0.00, "Safe", "safe", "#2EC4A6", [
        "No sign of fire or smoke in these readings.",
        "Keep sensors clean and test them regularly.",
    ]),
]


def risk_for(p: float) -> tuple[str, str, str, list[str]]:
    for lo, name, css, colour, advice in RISK_LEVELS:
        if p >= lo:
            return name, css, colour, advice
    return RISK_LEVELS[-1][1:5]  # pragma: no cover


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

:root{
  --ink:#10151C; --panel:#18202B; --panel-2:#1F2937; --line:#2B3645;
  --text:#E8ECF1; --muted:#8F9BAD;
  --ember:#FF5A1F; --amber:#FF9F1C; --yellow:#F6D55C; --teal:#2EC4A6; --red:#FF4D2E;
}
html, body, [class*="css"], .stMarkdown, .stText { font-family:'DM Sans', system-ui, sans-serif; }
h1,h2,h3,h4,.hero-title,.kpi-value,.risk-name { font-family:'Sora', system-ui, sans-serif !important; letter-spacing:-0.01em; }

.block-container{ padding-top:1.6rem; max-width:1280px; }
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }

/* Hero */
.hero{
  position:relative; overflow:hidden; border-radius:18px; padding:28px 32px 30px;
  background:linear-gradient(135deg,#1B2430 0%, #141B24 60%);
  border:1px solid var(--line); margin-bottom:18px;
}
.hero::after{ /* thermal strip */
  content:""; position:absolute; left:0; right:0; bottom:0; height:6px;
  background:linear-gradient(90deg,#2EC4A6 0%,#F6D55C 38%,#FF9F1C 66%,#FF4D2E 100%);
}
.hero-title{ font-size:2.2rem; font-weight:800; margin:0; color:var(--text); }
.hero-sub{ color:var(--muted); margin:6px 0 0; font-size:1.02rem; max-width:62ch; line-height:1.5; }
.hero-chips{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.chip{ background:rgba(255,255,255,.05); border:1px solid var(--line); color:var(--text);
       padding:4px 12px; border-radius:999px; font-size:.82rem; }

/* Risk banner */
.risk{ border-radius:16px; padding:18px 22px; border:1px solid var(--line); border-left-width:8px; background:var(--panel); }
.risk-name{ font-size:1.7rem; font-weight:700; margin:0; }
.risk-sub{ color:var(--muted); margin:2px 0 0; font-size:.95rem; }
.risk.safe{ border-left-color:var(--teal); } .risk.safe .risk-name{ color:var(--teal); }
.risk.watch{ border-left-color:var(--yellow); } .risk.watch .risk-name{ color:var(--yellow); }
.risk.warning{ border-left-color:var(--amber); } .risk.warning .risk-name{ color:var(--amber); }
.risk.critical{ border-left-color:var(--red); background:#2A1713; animation:pulse 1.6s ease-in-out infinite; }
.risk.critical .risk-name{ color:var(--red); }
@keyframes pulse{ 0%,100%{ box-shadow:0 0 0 0 rgba(255,77,46,.0);} 50%{ box-shadow:0 0 0 8px rgba(255,77,46,.18);} }
@media (prefers-reduced-motion: reduce){ .risk.critical{ animation:none; } }

/* KPI cards */
.kpi{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 18px; height:100%; }
.kpi-value{ font-size:1.7rem; font-weight:700; color:var(--text); line-height:1.2; }
.kpi-label{ color:var(--muted); font-size:.88rem; margin-top:2px; }

/* Misc */
.advice li{ margin-bottom:4px; }
.note{ color:var(--muted); font-size:.88rem; }
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{ border-radius:10px 10px 0 0; padding:8px 16px; }
.stTabs [aria-selected="true"]{ color:var(--ember) !important; }
div[data-testid="stMetricValue"]{ font-family:'Sora', sans-serif; }
.stButton>button, .stDownloadButton>button{ border-radius:10px; font-weight:600; }
.stButton>button[kind="primary"]{ background:var(--ember); border:none; color:#fff; }
.stButton>button[kind="primary"]:hover{ background:#E64A12; }
footer{ visibility:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Loading (cached)
# ----------------------------------------------------------------------------
def _first_existing(candidates: list[str]) -> Path | None:
    for name in candidates:
        p = BASE_DIR / name
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = _first_existing(MODEL_FILES)
    if path is None:
        return None, None, "Model file not found. Keep `Fire_Detection.pki` (or `.pkl`) next to app.py."
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        version_warning = any("version" in str(w.message).lower() for w in caught)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Could not load the model: {exc}"
    if not hasattr(model, "predict_proba"):
        return None, None, "The loaded object has no predict_proba(). Is this the right model file?"
    note = None
    if version_warning:
        note = (f"The model was trained with scikit-learn {TRAINED_SKLEARN} but this server runs a different "
                "version. Pin `scikit-learn==" + TRAINED_SKLEARN + "` in requirements.txt.")
    return model, path.name, note


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    path = _first_existing(DATA_FILES)
    if path is None:
        return None
    try:
        return pd.read_csv(path).drop_duplicates()
    except Exception:  # noqa: BLE001
        return None


model, model_name, model_note = load_model()
if model is None:
    st.error(model_note)
    st.stop()

FEATURE_ORDER: list[str] = list(getattr(model, "feature_names_in_", FEATURES.keys()))
missing_defs = [f for f in FEATURE_ORDER if f not in FEATURES]
if missing_defs:
    st.error(f"The model expects features this app does not define: {missing_defs}")
    st.stop()


# ----------------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------------
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURE_ORDER].astype(float))[:, 1]


def validate_reading(values: dict[str, float]) -> tuple[list[str], list[str], list[str]]:
    """Return (errors, warnings, notes) for one reading."""
    errors, warns, notes = [], [], []
    for k in FEATURE_ORDER:
        meta, v = FEATURES[k], values[k]
        if v is None or not np.isfinite(v):
            errors.append(f"{meta['label']} is missing or not a number.")
            continue
        lo, hi = meta["hard"]
        if not lo <= v <= hi:
            errors.append(f"{meta['label']} = {v:g} {meta['unit']} is outside the possible range ({lo:g} to {hi:g}).")
            continue
        tlo, thi = meta["train"]
        if not tlo <= v <= thi:
            warns.append(f"{meta['label']} = {v:g} {meta['unit']} is outside the training range ({tlo:g} to {thi:g}). "
                         "The model is extrapolating, so treat the result with caution.")
    if not errors:
        if values["Oxygen_pct"] < 19.5:
            notes.append("Oxygen is below 19.5%, which is considered oxygen-deficient air.")
        if values["Air_Quality_Index"] > 300:
            notes.append("AQI above 300 is in the hazardous band. Check for outdoor smog before assuming a fire.")
        if values["Humidity_pct"] > 90 and values["Temperature_C"] > 60:
            notes.append("Very high humidity together with high temperature is unusual. Check the sensors.")
    return errors, warns, notes


@st.cache_data(show_spinner=False)
def explain(values_tuple: tuple) -> pd.DataFrame | None:
    """Per-feature contribution to the log-odds (works for scaler + logistic regression pipelines)."""
    try:
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["classifier"]
        x = np.array(values_tuple, dtype=float)
        z = (x - scaler.mean_) / scaler.scale_
        contrib = z * clf.coef_[0]
        return pd.DataFrame({"feature": FEATURE_ORDER, "contribution": contrib, "z": z})
    except Exception:  # noqa: BLE001
        return None


def gauge(p: float, colour: str, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 46, "family": "Sora", "color": "#E8ECF1"}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8F9BAD", "tickfont": {"color": "#8F9BAD"}},
            "bar": {"color": colour, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "rgba(46,196,166,.22)"},
                {"range": [20, 50], "color": "rgba(246,213,92,.22)"},
                {"range": [50, 80], "color": "rgba(255,159,28,.22)"},
                {"range": [80, 100], "color": "rgba(255,77,46,.28)"},
            ],
            "threshold": {"line": {"color": "#FFFFFF", "width": 3}, "thickness": 0.85, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=270, margin=dict(l=24, r=24, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8ECF1"})
    return fig


def style_fig(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#E8ECF1"),
        xaxis=dict(gridcolor="#2B3645", zerolinecolor="#2B3645"),
        yaxis=dict(gridcolor="#2B3645", zerolinecolor="#2B3645"),
    )
    return fig


def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
DEFAULT_PRESET = PRESETS["🏠 Normal room"]
for k in FEATURE_ORDER:
    st.session_state.setdefault(f"n_{k}", float(DEFAULT_PRESET[k]))
    lo, hi = FEATURES[k]["ui"]
    st.session_state.setdefault(f"s_{k}", float(min(max(DEFAULT_PRESET[k], lo), hi)))
st.session_state.setdefault("history", [])


def _from_slider(k: str):
    st.session_state[f"n_{k}"] = float(st.session_state[f"s_{k}"])


def _from_number(k: str):
    lo, hi = FEATURES[k]["ui"]
    v = st.session_state.get(f"n_{k}")
    if v is not None:
        st.session_state[f"s_{k}"] = float(min(max(v, lo), hi))


def apply_preset(name: str):
    for k in FEATURE_ORDER:
        v = float(PRESETS[name][k])
        lo, hi = FEATURES[k]["ui"]
        st.session_state[f"n_{k}"] = v
        st.session_state[f"s_{k}"] = float(min(max(v, lo), hi))


def current_values() -> dict[str, float]:
    return {k: st.session_state.get(f"n_{k}") for k in FEATURE_ORDER}


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔥 EmberWatch")
    st.caption("Sensor-based fire and smoke detection")
    st.divider()

    st.markdown("**Alert sensitivity**")
    threshold = st.slider(
        "Fire is declared when probability reaches", 0.10, 0.90, 0.50, 0.05, format="%.2f",
        help="Lower = catches more fires but raises more false alarms. Higher = fewer false alarms but may miss early fires.",
    )
    if threshold < 0.4:
        st.caption("Sensitive mode: expect more false alarms.")
    elif threshold > 0.6:
        st.caption("Strict mode: early fires may be missed.")
    else:
        st.caption("Balanced mode (default).")

    st.divider()
    st.markdown("**About the model**")
    st.caption(f"File: `{model_name}`")
    st.caption("Algorithm: Logistic Regression with StandardScaler")
    st.caption("Inputs: " + ", ".join(FEATURES[f]["label"].lower() for f in FEATURE_ORDER))
    if model_note:
        st.warning(model_note)

    st.divider()
    st.caption("This is a decision-support demo trained on sensor data. It does not replace certified smoke alarms or fire safety systems.")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <p class="hero-title">EmberWatch</p>
      <p class="hero-sub">Enter sensor readings and get a fire probability, the reasons behind it, and the exact point where the alarm would trigger.</p>
      <div class="hero-chips">
        <span class="chip">Live prediction</span><span class="chip">Explainable result</span>
        <span class="chip">What-if analysis</span><span class="chip">Batch CSV</span><span class="chip">Input validation</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_live, tab_whatif, tab_batch, tab_model, tab_log = st.tabs(
    ["Live prediction", "What-if analysis", "Batch CSV", "Model insights", "Session log"]
)

# ============================================================================
# TAB 1 - LIVE PREDICTION
# ============================================================================
with tab_live:
    st.markdown("##### Quick scenarios")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), width="stretch")
    st.caption("The smog-day scenario is a good false-alarm test: high AQI with no real fire.")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("##### Sensor readings")
        for k in FEATURE_ORDER:
            meta = FEATURES[k]
            lo, hi = meta["ui"]
            c1, c2 = st.columns([3, 1.3])
            c1.slider(f"{meta['icon']} {meta['label']} ({meta['unit']})", lo, hi, step=meta["step"], key=f"s_{k}",
                      on_change=_from_slider, args=(k,), help=meta["help"])
            c2.number_input("Exact", min_value=meta["hard"][0], max_value=meta["hard"][1], step=meta["step"],
                            key=f"n_{k}", on_change=_from_number, args=(k,), label_visibility="visible",
                            format="%.2f")

    values = current_values()
    errors, warns, notes = validate_reading(values)

    with right:
        st.markdown("##### Result")
        if errors:
            for e in errors:
                st.error(e)
            st.info("Fix the values above to see a prediction.")
        else:
            x = pd.DataFrame([values])[FEATURE_ORDER]
            p = float(predict_proba(x)[0])
            level, css, colour, advice = risk_for(p)
            is_fire = p >= threshold

            st.markdown(
                f'<div class="risk {css}"><p class="risk-name">{"Fire detected" if is_fire else "No fire detected"} · {level}</p>'
                f'<p class="risk-sub">Fire probability {p*100:.1f}% against an alert threshold of {threshold*100:.0f}%</p></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(gauge(p, colour, threshold), width="stretch", config={"displayModeBar": False})

            st.markdown("**What to do**")
            st.markdown('<ul class="advice">' + "".join(f"<li>{a}</li>" for a in advice) + "</ul>", unsafe_allow_html=True)

            for w in warns:
                st.warning(w)
            for n in notes:
                st.info(n)

            if st.button("Add this reading to session log", type="primary", width="stretch"):
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"), **{k: values[k] for k in FEATURE_ORDER},
                    "fire_probability": round(p, 4), "risk": level, "alert": "Fire" if is_fire else "No fire",
                })
                st.toast("Reading added to the session log.", icon="✅")

    # ---- Explanation --------------------------------------------------------
    if not errors:
        st.divider()
        st.markdown("##### Why this result?")
        exp = explain(tuple(values[k] for k in FEATURE_ORDER))
        if exp is None:
            st.caption("Explanation is available only for scaler + logistic regression pipelines.")
        else:
            exp = exp.assign(label=[FEATURES[f]["label"] for f in exp["feature"]]).sort_values("contribution")
            fig = go.Figure(go.Bar(
                x=exp["contribution"], y=exp["label"], orientation="h",
                marker_color=["#FF5A1F" if c > 0 else "#2EC4A6" for c in exp["contribution"]],
                text=[f"{c:+.2f}" for c in exp["contribution"]], textposition="outside",
                hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Push towards fire (orange) or away from fire (teal)")
            st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})

            top = exp.reindex(exp["contribution"].abs().sort_values(ascending=False).index).iloc[0]
            direction = "towards fire" if top["contribution"] > 0 else "away from fire"
            st.markdown(
                f"The biggest driver is **{top['label']}**, pushing the result **{direction}**. "
                f"It is {abs(top['z']):.1f} standard deviations {'above' if top['z'] > 0 else 'below'} the training average."
            )
            st.caption("Each bar is the feature's contribution to the model's log-odds (scaled value × coefficient). "
                       "Bars add up, together with the intercept, to the final probability.")

# ============================================================================
# TAB 2 - WHAT-IF
# ============================================================================
with tab_whatif:
    st.markdown("##### What-if analysis")
    st.caption("Change one reading, keep the others as set in the Live prediction tab, and see how the probability moves.")
    errors_w, _, _ = validate_reading(current_values())
    if errors_w:
        st.error("Fix the invalid readings in the Live prediction tab first.")
    else:
        feat = st.selectbox("Reading to vary", FEATURE_ORDER, format_func=lambda f: f"{FEATURES[f]['icon']} {FEATURES[f]['label']} ({FEATURES[f]['unit']})")
        lo, hi = FEATURES[feat]["ui"]
        grid = np.linspace(lo, hi, 300)
        base = current_values()
        sweep = pd.DataFrame([{**base, feat: g} for g in grid])
        probs = predict_proba(sweep)
        now_p = float(predict_proba(pd.DataFrame([base]))[0])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=probs * 100, mode="lines", line=dict(color="#FF5A1F", width=3), name="Fire probability",
                                 hovertemplate=f"{FEATURES[feat]['label']}: %{{x:.2f}}<br>Probability: %{{y:.1f}}%<extra></extra>"))
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color="#FFFFFF", annotation_text="Alert threshold", annotation_position="bottom right")
        fig.add_trace(go.Scatter(x=[base[feat]], y=[now_p * 100], mode="markers", marker=dict(size=13, color="#F6D55C", line=dict(color="#fff", width=2)),
                                 name="Current reading", hoverinfo="skip"))
        tlo, thi = FEATURES[feat]["train"]
        fig.add_vrect(x0=max(lo, tlo), x1=min(hi, thi), fillcolor="rgba(255,255,255,.04)", line_width=0)
        fig.update_layout(xaxis_title=f"{FEATURES[feat]['label']} ({FEATURES[feat]['unit']})", yaxis_title="Fire probability (%)", yaxis_range=[0, 102],
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(style_fig(fig, 380), width="stretch", config={"displayModeBar": False})
        st.caption("The lighter band is the range the model saw during training. Outside it, results are extrapolated.")

        above = probs >= threshold
        if above.all():
            st.success(f"With the other readings as set, the alarm triggers across the whole {FEATURES[feat]['label'].lower()} range.")
        elif not above.any():
            st.success(f"With the other readings as set, {FEATURES[feat]['label'].lower()} alone never triggers the alarm in this range.")
        else:
            cross = int(np.argmax(above != above[0]))
            trigger = grid[cross]
            rising = above[-1]
            st.metric(
                "Trigger point",
                f"{trigger:.1f} {FEATURES[feat]['unit']}",
                delta=("alarm when it rises above this value" if rising else "alarm when it falls below this value"),
                delta_color="off",
            )
            gap = trigger - base[feat]
            st.markdown(f"Current value is **{abs(gap):.1f} {FEATURES[feat]['unit']} {'below' if gap > 0 else 'above'}** the trigger point.")

# ============================================================================
# TAB 3 - BATCH
# ============================================================================
with tab_batch:
    st.markdown("##### Batch prediction from CSV")
    st.caption(f"Upload a CSV with these columns: {', '.join(FEATURE_ORDER)}. Column names are matched ignoring case, spaces and underscores. Extra columns are kept.")

    template = pd.DataFrame([PRESETS["🏠 Normal room"], PRESETS["🍳 Cooking"], PRESETS["🚨 Severe fire"]])[FEATURE_ORDER]
    st.download_button("Download CSV template", template.to_csv(index=False).encode(), "fire_input_template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read this file as CSV: {exc}")
            raw = None

        if raw is not None:
            if raw.empty:
                st.error("The file has no rows.")
            elif len(raw) > MAX_UPLOAD_ROWS:
                st.error(f"The file has {len(raw):,} rows. The limit is {MAX_UPLOAD_ROWS:,}.")
            else:
                lookup = {norm(c): c for c in raw.columns}
                mapping = {f: lookup.get(norm(f)) for f in FEATURE_ORDER}
                missing = [f for f, c in mapping.items() if c is None]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                else:
                    df = raw.copy()
                    num = pd.DataFrame({f: pd.to_numeric(df[mapping[f]], errors="coerce") for f in FEATURE_ORDER})
                    bad = pd.Series(False, index=df.index)
                    reasons = pd.Series("", index=df.index)
                    for f in FEATURE_ORDER:
                        lo_h, hi_h = FEATURES[f]["hard"]
                        nan_mask = num[f].isna() | ~np.isfinite(num[f].fillna(np.nan))
                        range_mask = ~nan_mask & ((num[f] < lo_h) | (num[f] > hi_h))
                        reasons = reasons.where(~nan_mask, reasons + f"{f} missing/invalid; ")
                        reasons = reasons.where(~range_mask, reasons + f"{f} out of possible range; ")
                        bad |= nan_mask | range_mask
                    extrap = pd.Series(False, index=df.index)
                    for f in FEATURE_ORDER:
                        tlo, thi = FEATURES[f]["train"]
                        extrap |= (num[f] < tlo) | (num[f] > thi)

                    good = ~bad
                    out = df.copy()
                    out["fire_probability"] = np.nan
                    out.loc[good, "fire_probability"] = predict_proba(num.loc[good])
                    out["risk"] = out["fire_probability"].map(lambda p: risk_for(p)[0] if pd.notna(p) else "Invalid row")
                    out["alert"] = np.where(out["fire_probability"].isna(), "Invalid row",
                                            np.where(out["fire_probability"] >= threshold, "Fire", "No fire"))
                    out["extrapolated"] = np.where(good & extrap, "yes", np.where(good, "no", ""))
                    out["validation_note"] = reasons.str.strip()

                    n_bad, n_fire = int(bad.sum()), int((out["alert"] == "Fire").sum())
                    k1, k2, k3, k4 = st.columns(4)
                    k1.markdown(kpi(f"{len(out):,}", "Rows uploaded"), unsafe_allow_html=True)
                    k2.markdown(kpi(f"{n_fire:,}", "Fire alerts"), unsafe_allow_html=True)
                    k3.markdown(kpi(f"{int((good & extrap).sum()):,}", "Rows outside training range"), unsafe_allow_html=True)
                    k4.markdown(kpi(f"{n_bad:,}", "Invalid rows skipped"), unsafe_allow_html=True)
                    st.write("")

                    if n_bad:
                        st.warning(f"{n_bad} row(s) were skipped because of missing, non-numeric or impossible values. See the validation_note column.")

                    order = ["Safe", "Watch", "Warning", "Critical"]
                    counts = out["risk"].value_counts().reindex(order).fillna(0)
                    fig = go.Figure(go.Bar(x=order, y=counts.values, marker_color=["#2EC4A6", "#F6D55C", "#FF9F1C", "#FF4D2E"]))
                    fig.update_layout(title="Rows by risk level", yaxis_title="Rows")
                    st.plotly_chart(style_fig(fig, 280), width="stretch", config={"displayModeBar": False})

                    st.dataframe(out, width="stretch", height=320)
                    st.download_button("Download results CSV", out.to_csv(index=False).encode(), "fire_predictions.csv", "text/csv", type="primary")

# ============================================================================
# TAB 4 - MODEL INSIGHTS
# ============================================================================
with tab_model:
    st.markdown("##### Model insights")
    data = load_dataset()
    if data is None or "Fire_Smoke" not in data.columns or not set(FEATURE_ORDER).issubset(data.columns):
        st.info("Add the training CSV next to app.py to see metrics, ROC curve and data distribution.")
    else:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
        from sklearn.model_selection import train_test_split

        @st.cache_data(show_spinner="Evaluating model…")
        def evaluate(n_rows: int):
            X, y = data[FEATURE_ORDER], data["Fire_Smoke"]
            _, Xt, _, yt = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            pr = model.predict_proba(Xt)[:, 1]
            pred = (pr >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(yt, pr)
            return dict(
                acc=accuracy_score(yt, pred), prec=precision_score(yt, pred, zero_division=0), rec=recall_score(yt, pred, zero_division=0),
                f1=f1_score(yt, pred, zero_division=0), auc=roc_auc_score(yt, pr), cm=confusion_matrix(yt, pred, labels=[0, 1]),
                fpr=fpr, tpr=tpr, n=len(yt),
            )

        m = evaluate(len(data))
        c = st.columns(5)
        for col, (val, lab) in zip(c, [(m["acc"], "Accuracy"), (m["prec"], "Precision"), (m["rec"], "Recall"), (m["f1"], "F1 score"), (m["auc"], "ROC AUC")]):
            col.markdown(kpi(f"{val*100:.1f}%" if lab != "ROC AUC" else f"{val:.3f}", lab), unsafe_allow_html=True)
        st.caption(f"Measured on the same 25% hold-out split as the notebook ({m['n']:,} rows, random_state=42), at a 0.50 threshold.")
        st.write("")

        a, b = st.columns(2)
        with a:
            cm = m["cm"]
            fig = go.Figure(go.Heatmap(z=cm, x=["No fire", "Fire"], y=["No fire", "Fire"], colorscale=[[0, "#18202B"], [1, "#FF5A1F"]],
                                       text=cm, texttemplate="%{text}", textfont=dict(size=20), showscale=False))
            fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
            st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})
            st.caption(f"{cm[1][0]} fires were missed and {cm[0][1]} false alarms were raised. Missed fires matter more here, which is why the sensitivity slider exists.")
        with b:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines", line=dict(color="#FF5A1F", width=3), name="Model"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#8F9BAD", dash="dash"), name="Random guess"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})

        a, b = st.columns(2)
        with a:
            try:
                coefs = model.named_steps["classifier"].coef_[0]
                cdf = pd.DataFrame({"f": [FEATURES[f]["label"] for f in FEATURE_ORDER], "c": coefs}).sort_values("c")
                fig = go.Figure(go.Bar(x=cdf["c"], y=cdf["f"], orientation="h", marker_color=["#FF5A1F" if v > 0 else "#2EC4A6" for v in cdf["c"]]))
                fig.update_layout(title="Feature weights (on scaled inputs)")
                st.plotly_chart(style_fig(fig, 320), width="stretch", config={"displayModeBar": False})
            except Exception:  # noqa: BLE001
                st.caption("Feature weights are not available for this model type.")
        with b:
            vc = data["Fire_Smoke"].value_counts().sort_index()
            fig = go.Figure(go.Pie(labels=["No fire", "Fire"], values=[vc.get(0, 0), vc.get(1, 0)], hole=0.6, marker_colors=["#2EC4A6", "#FF5A1F"]))
            fig.update_layout(title="Training data balance")
            st.plotly_chart(style_fig(fig, 320), width="stretch", config={"displayModeBar": False})

        unused = [c for c in data.columns if c not in FEATURE_ORDER + ["Fire_Smoke"]]
        if unused:
            st.info("The dataset also contains " + ", ".join(unused) + ", which this model does not use. "
                    "Retraining with them could improve early detection.")

# ============================================================================
# TAB 5 - SESSION LOG
# ============================================================================
with tab_log:
    st.markdown("##### Session log")
    hist = st.session_state.history
    if not hist:
        st.info("Nothing logged yet. Use Add this reading to session log on the Live prediction tab.")
    else:
        hdf = pd.DataFrame(hist)
        fig = go.Figure(go.Scatter(x=list(range(1, len(hdf) + 1)), y=hdf["fire_probability"] * 100, mode="lines+markers",
                                   line=dict(color="#FF5A1F", width=3), marker=dict(size=9)))
        fig.add_hline(y=threshold * 100, line_dash="dash", line_color="#FFFFFF")
        fig.update_layout(title="Fire probability across logged readings", xaxis_title="Reading number", yaxis_title="Probability (%)", yaxis_range=[0, 102])
        st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})
        st.dataframe(hdf, width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("Download log CSV", hdf.to_csv(index=False).encode(), "fire_session_log.csv", "text/csv", type="primary", width="stretch")
        if c2.button("Clear log", width="stretch"):
            st.session_state.history = []
            st.rerun()
