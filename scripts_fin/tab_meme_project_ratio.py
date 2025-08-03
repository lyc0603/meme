"""
Script to generate a LaTeX table from a JSON file containing meme project ratios
showing only percentages for Pre- and Post-Trump samples with mapped names.
"""

import json
import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH

# Define naming dictionary
NAMING_DICT = {
    "no_one_care": "No one care",
    "pumpfun": "Unmigrated",
    "raydium": "Migrated",
}

# Load JSON file
with open(PROCESSED_DATA_PATH / "meme_project_ratio.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(list(data.items()), columns=["category", "count"])

# Separate pre_trump vs post_trump
df["period"] = df["category"].apply(
    lambda x: "Pre-Trump" if x.startswith("pre_trump") else "Post-Trump"
)
df["type"] = df["category"].apply(
    lambda x: x.replace("pre_trump_", "") if x.startswith("pre_trump") else x
)

# Pivot table
result = df.pivot(index="type", columns="period", values="count").fillna(0).astype(int)

# Compute column totals
col_totals = result.sum()

# Compute percentages only
result["Pre-Trump"] = (result["Pre-Trump"] / col_totals["Pre-Trump"] * 100).round(1)
result["Post-Trump"] = (result["Post-Trump"] / col_totals["Post-Trump"] * 100).round(1)

# Keep only percentage columns
result = result[["Pre-Trump", "Post-Trump"]]

# Apply name mapping
result.index = result.index.map(lambda x: NAMING_DICT.get(x, x))

# Add total row (100% each)
total_row = pd.Series(
    {"Pre-Trump": 100.0, "Post-Trump": 100.0},
    name="Total",
)
result = pd.concat([result, total_row.to_frame().T])

# ---- Generate LaTeX table ----
lines = [
    "\\begin{tabular}{lcc}",
    "\\toprule",
    "Project Type & Pre-Trump (\\%) & Post-Trump (\\%) \\\\",
    "\\midrule",
]

for row in result.index:
    if row == "Total":
        lines.append(r"\midrule")
    lines.append(
        f"{row} & {result.loc[row, 'Pre-Trump']:.1f} & {result.loc[row, 'Post-Trump']:.1f} \\\\"
    )

lines.extend(["\\bottomrule", "\\end{tabular}"])

# Write to file
with open(TABLE_PATH / "meme_project_ratio.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
