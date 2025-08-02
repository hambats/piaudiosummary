"""Generate daily summaries of captured speech."""

import os
from datetime import datetime

import pandas as pd
from transformers import pipeline

from model_manager import ensure_models

SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"


def load_log(log_path="data/log.csv"):
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["time", "class", "lang", "transcript"])
    df = pd.read_csv(log_path, header=None)
    if df.shape[1] == 3:
        df.columns = ["time", "class", "transcript"]
        df["lang"] = "unknown"
    else:
        df.columns = ["time", "class", "lang", "transcript"]
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_narrative(df):
    """Create a textual narrative from log rows."""
    pieces = []
    for _, row in df.iterrows():
        timestamp = row["time"].strftime("%H:%M")
        lang = row.get("lang", "unknown")
        if row["transcript"]:
            pieces.append(
                f"At {timestamp} [{lang}], someone said: '{row['transcript']}'."
            )
        else:
            pieces.append(f"At {timestamp}, event: {row['class']}")
    return " ".join(pieces)


def summarize_date(date_str):
    ensure_models()
    df = load_log()
    if df.empty:
        print("No transcripts found.")
        return

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    day_df = df[df["time"].dt.date == target_date]
    if day_df.empty:
        print(f"No transcripts for {date_str}")
        return

    text = build_narrative(day_df)
    summarizer = pipeline("summarization", model=SUMMARY_MODEL, local_files_only=True)
    prompt = (
        "Summarize the following observed audio events and conversations. "
        "Include who might be speaking, what happened, when and where it occurred, "
        "and possible reasons if implied: "
        + text
    )
    result = summarizer(prompt, max_length=200, min_length=50, do_sample=False)[0]
    summary_text = result["summary_text"]

    out_file = f"data/summary_{date_str}.txt"
    with open(out_file, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {out_file}")
    print(summary_text)


if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    summarize_date(today)
