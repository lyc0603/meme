"""Script to analyze the bot and dumper"""

import json
import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH

# Load data
with open(f"{PROCESSED_DATA_PATH}/meme_project_ratio.json", "r", encoding="utf-8") as f:
    meme_project_ratio = json.load(f)

pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")
pfm["weight"] = pfm["category"].apply(lambda x: meme_project_ratio[x])

# Filter out 'no dumper'
pfm = pfm.loc[pfm["dumper"] != "no dumper"]
pfm["creator_dump"] = pfm["dumper"].apply(lambda x: 1 if x == "creator" else 0)
pfm["sniper_dump"] = pfm["dumper"].apply(lambda x: 1 if x == "sniper" else 0)
pfm["other_dump"] = pfm["dumper"].apply(lambda x: 1 if x == "other" else 0)

# Weight expansion
pfm = pfm.loc[pfm.index.repeat(pfm["weight"])].reset_index(drop=True)

# Compute weighted means
bundle = pfm.groupby("launch_bundle")[
    ["creator_dump", "sniper_dump", "other_dump"]
].mean()
sniper = pfm.groupby("sniper_bot")[["creator_dump", "sniper_dump", "other_dump"]].mean()

# Convert to percentage
bundle = (bundle * 100).round(2)
sniper = (sniper * 100).round(2)

# Build LaTeX tabular environment manually
lines = []
lines.append("\\begin{tabular}{lcccc}")
lines.append("\\toprule")
lines.append("& \\multicolumn{2}{c}{Rat Bot} & \\multicolumn{2}{c}{Sniper Bot} \\\\")
lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
lines.append("& No & Yes & No & Yes \\\\")
lines.append("\\midrule")

for metric in ["creator_dump", "sniper_dump", "other_dump"]:
    name = metric.replace("_", " ").title()
    row = (
        f"{name} & "
        f"{bundle.loc[0, metric]:.2f}\\% & "
        f"{bundle.loc[1, metric]:.2f}\\% & "
        f"{sniper.loc[0, metric]:.2f}\\% & "
        f"{sniper.loc[1, metric]:.2f}\\% \\\\"
    )
    lines.append(row)

lines.append("\\bottomrule")
lines.append("\\end{tabular}")

# Save LaTeX file
with open(TABLE_PATH / "bot_dumper.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
