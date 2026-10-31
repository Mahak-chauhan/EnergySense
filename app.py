import os
import joblib
import pandas as pd
import streamlit as st
import altair as alt
from features import create_features

st.set_page_config(
    page_title="EnergySense — AI Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
    --bg-main: #080C14;
    --bg-card: #0F172A;
    --bg-card-subtle: #131E35;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(139, 92, 246, 0.35);
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --text-dim: #64748B;
    --accent-purple: #8B5CF6;
    --accent-purple-light: #A78BFA;
    --accent-cyan: #06B6D4;
    --accent-cyan-light: #38BDF8;
}

/* Global Background and Typography */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

p, span, label {
    color: var(--text-muted);
}

/* Header Elements */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid var(--border-accent);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent-cyan-light);
    margin-bottom: 0.6rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.brand-title {
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    color: #FFFFFF;
}

.brand-gradient {
    background: linear-gradient(135deg, #C084FC 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
    margin-bottom: 1.25rem;
    max-width: 680px;
    line-height: 1.5;
}

/* Card Containers */
.saas-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 100px;
}

.metric-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.76rem;
    color: var(--text-dim);
    margin-top: 0.25rem;
}

/* Primary Action Button */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8B5CF6 0%, #4F46E5 100%) !important;
    border-color: #C084FC !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35) !important;
}

/* Radio buttons */
div[data-testid="stRadio"] label {
    color: var(--text-primary) !important;
    font-weight: 500;
    font-size: 0.9rem;
}

/* Expander Styling */
div[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    margin-top: 1rem;
}

/* Dataframe Styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
}

/* Info/Alert Callout */
div[data-testid="stAlert"] {
    background-color: var(--bg-card-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-muted) !important;
}

/* Hide Streamlit Header & Footer deco */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)




@st.cache_resource
def load_model_and_artifacts():

    model_path = os.path.join(BASE_DIR, "final_model.pkl")
    feature_cols_path = os.path.join(BASE_DIR, "feature_columns.pkl")

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)
    return model, feature_cols

@st.cache_data
def load_recent_history():

    history_path = os.path.join(BASE_DIR, "recent_history.csv")
    df = pd.read_csv(history_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df['energy_consumption']

try:
    model, feature_cols = load_model_and_artifacts()
    energy_history = load_recent_history()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()


st.markdown('<div class="badge-pill">⚡ Intelligent Grid Intelligence</div>', unsafe_allow_html=True)
st.markdown('<h1 class="brand-title">Energy<span class="brand-gradient">Sense</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="brand-subtitle">'
    'AI-powered household electricity consumption forecasting with gradient-boosted time-series modeling '
    'and recursive multi-step estimation.'
    '</p>',
    unsafe_allow_html=True
)

last_timestamp = energy_history.index[-1]
last_val = energy_history.iloc[-1]

c_stat1, c_stat2, c_stat3 = st.columns(3)
with c_stat1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Forecast Origin (T₀)</div>'
        f'<div class="metric-value">{last_timestamp.strftime("%b %d, %H:%M")}</div>'
        f'<div class="metric-sub">Latest verified actual reading</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with c_stat2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Origin Consumption</div>'
        f'<div class="metric-value">{last_val:.3f} <span style="font-size:1rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">Active power hourly average</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with c_stat3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Historical Context</div>'
        f'<div class="metric-value">{len(energy_history):,} <span style="font-size:1rem;color:#94A3B8;">hrs</span></div>'
        f'<div class="metric-sub">Seeding lag & rolling dynamics</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.write("")




st.markdown("### 🔮 Forecast Configuration")

horizon_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")

with horizon_col:
    horizon_map = {
        "1 Hour (Short-Term)": 1,
        "6 Hours (Intraday)": 6,
        "12 Hours (Half-Day)": 12,
        "24 Hours (Full-Day)": 24
    }
    selected_label = st.radio(
        "Select Forecast Horizon:",
        options=list(horizon_map.keys()),
        index=2,
        horizontal=True
    )
    selected_horizon = horizon_map[selected_label]

with btn_col:
    run_button = st.button("⚡ Run Forecast", use_container_width=True)





with st.spinner("Computing recursive multi-step forecast..."):
    current_history = energy_history.copy()
    predictions = []

    for step in range(1, selected_horizon + 1):
        future_time = current_history.index[-1] + pd.Timedelta(hours=1)


        features_df = create_features(current_history, [future_time], is_training=False)
        features_df = features_df[feature_cols]


        pred = max(0.0, float(model.predict(features_df)[0]))

        predictions.append({
            "Date & Time": future_time,
            "Estimated Consumption (kW)": pred,
            "Change from Origin (kW)": pred - last_val
        })


        current_history.loc[future_time] = pred

forecast_df = pd.DataFrame(predictions).set_index("Date & Time")




avg_pred = forecast_df["Estimated Consumption (kW)"].mean()
max_pred = forecast_df["Estimated Consumption (kW)"].max()
min_pred = forecast_df["Estimated Consumption (kW)"].min()
end_pred = forecast_df["Estimated Consumption (kW)"].iloc[-1]

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Projected Average</div>'
        f'<div class="metric-value">{avg_pred:.3f} <span style="font-size:0.95rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">Mean across {selected_horizon}h window</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Projected Peak</div>'
        f'<div class="metric-value" style="color:#38BDF8;">{max_pred:.3f} <span style="font-size:0.95rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">Maximum expected load</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Projected Minimum</div>'
        f'<div class="metric-value">{min_pred:.3f} <span style="font-size:0.95rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">Minimum expected baseline</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Horizon End Value</div>'
        f'<div class="metric-value" style="color:#A78BFA;">{end_pred:.3f} <span style="font-size:0.95rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">At +{selected_horizon}h mark</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.write("")




st.markdown("### 📈 Load Trajectory Visualization")


context_hours = 12
hist_slice = energy_history.iloc[-context_hours:]
hist_records = pd.DataFrame({
    "Timestamp": hist_slice.index,
    "Consumption (kW)": hist_slice.values,
    "Series": "Observed History (Actual)"
})

forecast_records = pd.DataFrame({
    "Timestamp": forecast_df.index,
    "Consumption (kW)": forecast_df["Estimated Consumption (kW)"].values,
    "Series": "Projected Forecast (Estimate)"
})


bridge_record = pd.DataFrame({
    "Timestamp": [last_timestamp],
    "Consumption (kW)": [last_val],
    "Series": ["Projected Forecast (Estimate)"]
})

combined_chart_df = pd.concat([hist_records, bridge_record, forecast_records]).sort_values("Timestamp")

chart = (
    alt.Chart(combined_chart_df)
    .mark_line(strokeWidth=2.8)
    .encode(
        x=alt.X(
            "Timestamp:T",
            title="Time",
            axis=alt.Axis(
                format="%b %d, %H:%M",
                labelColor="#94A3B8",
                titleColor="#F8FAFC",
                gridColor="rgba(255, 255, 255, 0.05)",
                domainColor="rgba(255, 255, 255, 0.12)"
            )
        ),
        y=alt.Y(
            "Consumption (kW):Q",
            title="Active Power Consumption (kW)",
            scale=alt.Scale(zero=False, padding=10),
            axis=alt.Axis(
                labelColor="#94A3B8",
                titleColor="#F8FAFC",
                gridColor="rgba(255, 255, 255, 0.05)",
                domainColor="rgba(255, 255, 255, 0.12)"
            )
        ),
        color=alt.Color(
            "Series:N",
            title="Data Series",
            scale=alt.Scale(
                domain=["Observed History (Actual)", "Projected Forecast (Estimate)"],
                range=["#38BDF8", "#A78BFA"]
            ),
            legend=alt.Legend(
                orient="top-left",
                titleColor="#F8FAFC",
                labelColor="#CBD5E1",
                fillColor="#0F172A",
                strokeColor="rgba(255, 255, 255, 0.1)",
                padding=8,
                cornerRadius=6
            )
        ),
        strokeDash=alt.StrokeDash(
            "Series:N",
            scale=alt.Scale(
                domain=["Observed History (Actual)", "Projected Forecast (Estimate)"],
                range=[[0, 0], [5, 4]]
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Timestamp:T", format="%Y-%m-%d %H:%M", title="Time"),
            alt.Tooltip("Consumption (kW):Q", format=".3f", title="Power (kW)")
        ]
    )
    .properties(height=340)
    .configure_view(strokeWidth=0, fill="rgba(15, 23, 42, 0.5)")
)

st.altair_chart(chart, use_container_width=True)

st.caption(
    "⚠️ Note on Recursive Estimation: Forecast values represent iterative estimates. "
    "At extended horizons (e.g., 24h), the model uses previous predictions as inputs for lag features, "
    "causing forecast uncertainty to accumulate over time."
)




with st.expander("📋 View Detailed Hourly Forecast Table", expanded=False):
    table_display = forecast_df.copy()
    table_display.index = table_display.index.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        table_display.style.format({
            "Estimated Consumption (kW)": "{:.3f} kW",
            "Change from Origin (kW)": "{:+.3f} kW"
        }),
        use_container_width=True
    )




st.markdown("### 📊 Model Architecture & Holdout Benchmark")
st.markdown(
    '<p style="font-size:0.88rem;color:#94A3B8;margin-bottom:1rem;">'
    'Rigorous out-of-sample evaluation on the 6,452-hour blind chronological test holdout (2010-01-10 to 2010-11-26).'
    '</p>',
    unsafe_allow_html=True
)

perf1, perf2, perf3, perf4 = st.columns(4)
with perf1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">R² Score (Holdout)</div>'
        f'<div class="metric-value" style="color:#38BDF8;">0.6135</div>'
        f'<div class="metric-sub">Explains ~61.4% of load variance</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with perf2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">MAE (Holdout)</div>'
        f'<div class="metric-value">0.3287 <span style="font-size:0.95rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">Mean absolute deviation</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with perf3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">RMSE (Holdout)</div>'
        f'<div class="metric-value">0.4791 <span style="font-size:0.95rem;color:#94A3B8;">kW</span></div>'
        f'<div class="metric-sub">Root mean squared error</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with perf4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">WAPE (Holdout)</div>'
        f'<div class="metric-value">31.68%</div>'
        f'<div class="metric-sub">Weighted percentage error</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div style="background:#0F172A;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.8rem;font-size:0.82rem;color:#94A3B8;line-height:1.5;">
        <strong style="color:#F8FAFC;">Methodology & Evaluation Disclaimer:</strong><br>
        • <strong>One-Step-Ahead Metric:</strong> Reported holdout metrics (R² = 0.6135, MAE = 0.3287 kW) reflect 1-step-ahead evaluations where genuine historical ground truth (t−1) is known.<br>
        • <strong>Statistical Meaning of R²:</strong> The R² score represents the proportion of variance explained by the model relative to a horizontal mean baseline. It is not an "accuracy percentage".<br>
        • <strong>Operational Distinction:</strong> Real-time forecasts over multi-hour horizons (6h, 12h, 24h) cannot access future actuals; the model recursively cycles past predictions into lag and rolling features, naturally accumulating uncertainty as the horizon expands.
    </div>
    """,
    unsafe_allow_html=True
)




st.markdown(
    """
    <div style="margin-top:2.5rem;padding-top:1.2rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.78rem;color:#64748B;display:flex;justify-content:space-between;align-items:center;">
        <span>EnergySense • Portfolio AI Project • Built with HistGradientBoostingRegressor & Streamlit</span>
        <span>Dataset: UCI Individual Household Electric Power Consumption • Target: Active Power (kW)</span>
    </div>
    """,
    unsafe_allow_html=True
)
