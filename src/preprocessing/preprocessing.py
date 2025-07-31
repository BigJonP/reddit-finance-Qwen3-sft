import json
import re

import polars as pl

with open("src/preprocessing/title_patterns.txt", "r") as file:
    patterns = file.readlines()

patterns = [pattern.strip() for pattern in patterns]
pattern_regex = "|".join(re.escape(p) for p in patterns)

df = pl.read_ndjson("hf://datasets/winddude/reddit_finance_43_250k/top.jsonl")

df = df.filter(~pl.col("title").str.contains(pattern_regex))

df = df.with_columns(
    [
        pl.col("title")
        .str.replace_all(r"https?://\S+", "")
        .str.replace_all(r"u/[^\s]+", "user")
        .alias("title"),
        pl.col("selftext")
        .str.replace_all(r"https?://\S+", "")
        .str.replace_all(r"u/[^\s]+", "user")
        .alias("selftext"),
        pl.col("body")
        .str.replace_all(r"https?://\S+", "")
        .str.replace_all(r"u/[^\s]+", "user")
        .alias("body"),
    ]
)

training_data = []
for row in df.iter_rows(named=True):
    training_data.append(
        {
            "text": "### Instruction: \n"
            + row["title"]
            + row["selftext"]
            + "\n\n### Response: \n"
            + row["body"]
        }
    )

with open("src/lora/training_data.jsonl", "w") as file:
    for row in training_data:
        file.write(json.dumps(row) + "\n")
