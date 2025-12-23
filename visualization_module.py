import plotly.express as px
import pandas as pd
import os

class BasePlot:
    def __init__(self, df: pd.DataFrame, title: str):
        self.df = df
        self.title = title
        self.fig = None

    def _create_layout(self):
        # Shared layout, theme, and titles
        self.fig.update_layout(
            title_text=self.title,
            title_x=0.5,  # Center title
            template="plotly_white"  # Clean theme
        )

    def render(self):
        raise NotImplementedError("Subclasses must implement render method")

    def export(self, filename: str = "plot.html", output_dir: str = "plots"):
        if self.fig is None:
            raise ValueError("Figure has not been rendered yet. Call render() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        self.fig.write_html(filepath)
        print(f"Plot exported to {filepath}")

class ScatterPlot(BasePlot):
    def __init__(self, df: pd.DataFrame, title: str = "WER vs BLEU per Run ID"):
        super().__init__(df, title)

    def render(self):
        self.fig = px.scatter(
            self.df,
            x="wer",
            y="bleu",
            hover_name="run_id",
            title=self.title
        )
        self._create_layout()
        return self.fig

class BarPlot(BasePlot):
    def __init__(self, df: pd.DataFrame, metric: str, title: str):
        super().__init__(df, title)
        if metric not in ["wer", "bleu"]:
            raise ValueError("Metric must be 'wer' or 'bleu'")
        self.metric = metric

    def render(self):
        self.fig = px.bar(
            self.df,
            x="run_id",
            y=self.metric,
            title=self.title,
            labels={'run_id': 'Run ID', self.metric: self.metric.upper()}
        )
        self._create_layout()
        return self.fig

if __name__ == "__main__":
    import json

    # Load preprocessed data
    with open("data/processed/processed_metrics.json", 'r') as f:
        processed_data = json.load(f)
    
    df = pd.DataFrame(processed_data)

    # Create and render Scatter Plot
    scatter_plot = ScatterPlot(df)
    scatter_plot.render()
    scatter_plot.export(filename="wer_vs_bleu_scatter.html")

    # Create and render Bar Plot for WER
    wer_bar_plot = BarPlot(df, metric="wer", title="WER per Run ID")
    wer_bar_plot.render()
    wer_bar_plot.export(filename="wer_bar_chart.html")

    # Create and render Bar Plot for BLEU
    bleu_bar_plot = BarPlot(df, metric="bleu", title="BLEU per Run ID")
    bleu_bar_plot.render()
    bleu_bar_plot.export(filename="bleu_bar_chart.html")


