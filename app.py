import streamlit as st
import pandas as pd
import json
import os

from streamlit_extras.metric_cards import style_metric_cards

# Assuming visualization_module.py is in the same directory
from visualization_module import ScatterPlot, BarPlot
# Initialize session state for model selection
if 'selected_models' not in st.session_state:
    st.session_state['selected_models'] = []

st.set_page_config(page_title="Federated Research Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("Federated Learning Metrics Dashboard")
st.caption("A research project supervised by Prof. Manfred Vogel from ETH Zurich. Our main pipeline and results can be found here: [https://github.com/Aryand43/whisper-finetune-pipeline](https://github.com/Aryand43/whisper-finetune-pipeline). This dashboard provides a comprehensive WER and BLEU comparison across base and aggregated federated models.")
st.divider()

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: black !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    processed_data_path = "data/processed/processed_metrics.json"
    if not os.path.exists(processed_data_path):
        st.error(f"Error: {processed_data_path} not found. Please run preprocessing_module.py first.")
        st.stop()
    with open(processed_data_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

all_run_ids = df["run_id"].tolist()

with st.sidebar:
    st.header("Model Selection")
    st.markdown("Select the different models to see the comparator results.")
    selected_models = st.multiselect(
        "Select models to compare",
        options=all_run_ids,
        default=all_run_ids, # Default: all models selected
        key="selected_models"
    )

st.markdown("**Base models:** eager-haze, hearty-bee, smart-smoke, wandering-river  \n**Aggregated models:** all other runs")
st.divider()

# Filter DataFrame reactively
filtered_df = df[df["run_id"].isin(selected_models)] if selected_models else pd.DataFrame(columns=df.columns)

st.subheader("Comparison")
col1, col2 = st.columns(2)

# Updated CSS for clean white text and labels
st.markdown(
    """
    <style>
    /* Target the Label (Best WER / Best BLEU) */
    [data-testid="stMetricLabel"] p {
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    /* Target the Value (The actual numbers) */
    [data-testid="stMetricValue"] div {
        color: white !important;
    }

    /* Target the Delta (The Model ID text) */
    [data-testid="stMetricDelta"] div {
        color: #BBBBBB !important; /* Slightly off-white for visual hierarchy */
    }

    /* Optional: Ensure the card background is dark enough to show white text */
    div[data-testid="metric-container"] {
        background-color: #0E1117;
        border: 1px solid #31333F;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("Comparison")
col1, col2 = st.columns(2)

if not filtered_df.empty:
    best_wer_row = filtered_df.loc[filtered_df["wer"].idxmin()]
    best_bleu_row = filtered_df.loc[filtered_df["bleu"].idxmax()]

    with col1:
        st.metric(
            label="Best WER",
            value=f"{best_wer_row['wer']:.4f}", # Single quotes inside f-string
            delta=f"Model: {best_wer_row['run_id']}", # Single quotes inside f-string
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="Best BLEU",
            value=f"{best_bleu_row['bleu']:.2f}", # Single quotes inside f-string
            delta=f"Model: {best_bleu_row['run_id']}", # Single quotes inside f-string
            delta_color="normal"
        )
else:
    col1.metric("Best WER", "N/A")
    col2.metric("Best BLEU", "N/A")

style_metric_cards(background_color="#0E1117", border_left_color="#00FFAA")

st.divider()

# Charts
st.subheader("Charts")
scatter_tab, bars_tab, line_tab, table_tab = st.tabs(["Scatter", "Bars", "Line Chart", "Table"])

with scatter_tab:
    st.header("WER vs BLEU Scatter Plot")
    scatter_plot_instance = ScatterPlot(filtered_df)
    scatter_fig = scatter_plot_instance.render()
    st.plotly_chart(scatter_fig, use_container_width=True)

with bars_tab:
    st.header("Bar Chart of Metrics")
    selected_metric = st.selectbox("Select Metric", ["wer", "bleu"], key="bar_metric_select")
    
    bar_plot_instance = BarPlot(filtered_df, metric=selected_metric, title=f"{selected_metric.upper()} per Run ID")
    bar_fig = bar_plot_instance.render()
    st.plotly_chart(bar_fig, use_container_width=True)

with line_tab:
    st.header("WER and BLEU Trends")
    if not filtered_df.empty:
        # Ensure 'run_id' is suitable for x-axis, maybe sort by a metric if order matters
        df_line = filtered_df.set_index('run_id')[['wer', 'bleu']]
        st.line_chart(df_line)
    else:
        st.write("No data to display line chart.")

with table_tab:
    st.header("Table of Processed Metrics")
    st.dataframe(filtered_df, use_container_width=True)

st.divider()
