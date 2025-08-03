"""Script to analyze the bot and dumper"""

import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH

# Load data
pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")

# Filter out 'no dumper'
pfm = pfm.loc[pfm["dumper"] != "no dumper"]
pfm = pfm.loc[
    (pfm["winner_dump"] != 0) | (pfm["loser_dump"] != 0) | (pfm["neutral_dump"] != 0)
]

CATE_NAMING_DICT = {
    "no_one_care": "no_one_care",
    "pre_trump_no_one_care": "no_one_care",
    "pumpfun": "pumpfun",
    "pre_trump_pumpfun": "pumpfun",
    "raydium": "raydium",
    "pre_trump_raydium": "raydium",
}
pfm["category"] = pfm["category"].replace(CATE_NAMING_DICT)

# Compute weighted means
bundle = pfm.groupby("category")[["winner_dump", "loser_dump", "neutral_dump"]].mean()
bundle = (bundle * 100).round(2)

# Build LaTeX tabular environment manually
lines = []
lines.append("\\begin{tabular}{lccc}")
lines.append("\\toprule")
lines.append("& No One Care & Unsucessful & Migrated \\\\")
lines.append("\\midrule")

for metric in ["winner_dump", "loser_dump", "neutral_dump"]:
    name = metric.replace("_", " ").title()
    row = (
        f"{name} & "
        f"{bundle.loc['no_one_care', metric]:.2f}\\% & "
        f"{bundle.loc['pumpfun', metric]:.2f}\\% & "
        f"{bundle.loc['raydium', metric]:.2f}\\% \\\\"
    )
    lines.append(row)

lines.append("\\bottomrule")
lines.append("\\end{tabular}")

# Save LaTeX file
with open(TABLE_PATH / "wln_dumper.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
