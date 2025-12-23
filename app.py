import streamlit as st
import pandas as pd
import json
import os

# Assuming visualization_module.py is in the same directory
from visualization_module import ScatterPlot, BarPlot

st.set_page_config(layout="wide")

st.title("Federated Learning Metrics Dashboard")
st.caption("Exploring WER and BLEU metrics for various federated learning runs.")

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

overview_tab, scatter_tab, bars_tab = st.tabs(["Overview", "Scatter", "Bars"])

with overview_tab:
    st.header("Overview of Processed Metrics")
    st.dataframe(df, use_container_width=True)

with scatter_tab:
    st.header("WER vs BLEU Scatter Plot")
    scatter_plot_instance = ScatterPlot(df)
    scatter_fig = scatter_plot_instance.render()
    st.plotly_chart(scatter_fig, use_container_width=True)

with bars_tab:
    st.header("Bar Chart of Metrics")
    selected_metric = st.selectbox("Select Metric", ["wer", "bleu"], key="bar_metric_select")
    
    bar_plot_instance = BarPlot(df, metric=selected_metric, title=f"{selected_metric.upper()} per Run ID")
    bar_fig = bar_plot_instance.render()
    st.plotly_chart(bar_fig, use_container_width=True)

