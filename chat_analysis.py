import pandas as pd

def get_best_model(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    best_wer_row = df.loc[df["wer"].idxmin()]
    best_bleu_row = df.loc[df["bleu"].idxmax()]
    if best_wer_row["run_id"] == best_bleu_row["run_id"]:
        return f"The best overall model is {best_wer_row['run_id']} with WER: {best_wer_row['wer']:.4f} and BLEU: {best_bleu_row['bleu']:.2f}."
    else:
        return f"Best WER: {best_wer_row['run_id']} ({best_wer_row['wer']:.4f}). Best BLEU: {best_bleu_row['run_id']} ({best_bleu_row['bleu']:.2f})."

def get_worst_model(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    worst_wer_row = df.loc[df["wer"].idxmax()]
    worst_bleu_row = df.loc[df["bleu"].idxmin()]
    if worst_wer_row["run_id"] == worst_bleu_row["run_id"]:
        return f"The worst overall model is {worst_wer_row['run_id']} with WER: {worst_wer_row['wer']:.4f} and BLEU: {worst_bleu_row['bleu']:.2f}."
    else:
        return f"Worst WER: {worst_wer_row['run_id']} ({worst_wer_row['wer']:.4f}). Worst BLEU: {worst_bleu_row['run_id']} ({worst_bleu_row['bleu']:.2f})."

def explain_tradeoff(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"
    return (
        "In speech recognition, WER (Word Error Rate) and BLEU (Bilingual Evaluation Understudy) "
        "often have an inverse relationship. Generally, models with lower WER are better as they "
        "indicate fewer errors. Higher BLEU scores are also better, indicating higher quality "
        "transcriptions. \n\nHowever, improving one metric might sometimes lead to a slight decrease "
        "in the other, indicating a trade-off. It's important to consider both when evaluating models."
    )

def get_analysis_response(query: str, df: pd.DataFrame) -> str:
    query = query.lower()
    if "best model" in query:
        return get_best_model(df)
    elif "worst model" in query:
        return get_worst_model(df)
    elif "wer-bleu tradeoff" in query or "tradeoff" in query:
        return explain_tradeoff(df)
    else:
        return "I can help you by telling you about the 'best model', 'worst model', or the 'WER-BLEU tradeoff'."

