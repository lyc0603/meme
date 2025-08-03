"""
Script to create a LaTeX table summarizing the performance of traders
"""

import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH

# Load trader-level performance data
pft = pd.read_csv(PROCESSED_DATA_PATH / "pft.csv")
pft = pft.loc[(pft["winner"] == 1) | (pft["loser"] == 1) | (pft["neutral"] == 1)]
pft.drop_duplicates(subset=["token_address", "trader_address"], inplace=True)

pft = pft[
    [
        "token_address",
        "trader_address",
        "winner",
        "loser",
        "neutral",
        "creator",
        "sniper",
    ]
]

pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")
# pfm = trader_t.merge(pfm, how="left", on="token_address")
pfm = pft.merge(pfm, how="left", on="token_address")

# aggreagete by trader
pfm = (
    pfm.groupby(["trader_address"])[["winner", "loser", "neutral", "creator", "sniper"]]
    .mean()
    .reset_index()
)

# If the creator and sniper greater than 0, set to 1, else 0
pfm["creator"] = pfm["creator"].apply(lambda x: int(x > 0))
pfm["sniper"] = pfm["sniper"].apply(lambda x: int(x > 0))

wln_counts = {}

for group in ["winner", "loser", "neutral"]:
    # Count unique traders in each group
    # pfm_group = pfm[pfm[group] == 1]
    # wln_counts[group] = pfm_group["trader_address"].nunique()
    wln_counts[group] = pfm[group].sum()


def compute_group_counts(df: pd.DataFrame, group_var: str) -> pd.Series:
    """Compute raw counts of winner, loser, and neutral traders within a group."""
    sub_df = df[df[group_var] == 1]
    counts = {
        "Winner": int(sub_df["winner"].sum()),
        "Loser": int(sub_df["loser"].sum()),
        "Neutral": int(sub_df["neutral"].sum()),
    }
    return pd.Series(counts, name=group_var.capitalize())


# Compute raw counts
creator_counts = compute_group_counts(pfm, "creator")
sniper_counts = compute_group_counts(pfm, "sniper")

# Combine into a table
# Combine into a table
result = pd.concat([creator_counts, sniper_counts], axis=1)

# Add row total (use wln_counts)
result["Total"] = [
    int(wln_counts["winner"]),
    int(wln_counts["loser"]),
    int(wln_counts["neutral"]),
]

# Add column total (Creator, Sniper, Total)
column_total = pd.Series(
    {
        "Creator": int(result["Creator"].sum()),
        "Sniper": int(result["Sniper"].sum()),
        "Total": int(
            sum(wln_counts.values())
        ),  # Correct: total number of unique traders
    },
    name="Total",
)

# Append column total as new row
result.loc["Total"] = column_total

# Generate LaTeX table
lines = [
    "\\begin{tabular}{lccc}",
    "\\toprule",
    "Trader Type & Total & Creator & Sniper \\\\",
    "\\midrule",
]

for row in result.index:
    if row == "Total":
        lines.append(r"\midrule")
    lines.append(
        f"{row} & {result.loc[row, 'Total']:,} &  {result.loc[row, 'Creator']:,} & {result.loc[row, 'Sniper']:,} \\\\"
    )

lines.extend(["\\bottomrule", "\\end{tabular}"])

# Write LaTeX file
with open(TABLE_PATH / "sniper_creator.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
