"""This script compares financial metrics across Winner, Loser, and Neutral projects in a single LaTeX table with mean columns."""

import numpy as np
import pandas as pd
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
    ID_DICT,
)


def significance_stars(p):
    """Return significance stars based on p-value."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


# Define variable list and naming
X_VAR_PANEL = (
    list(NAMING_DICT.keys()) + list(PFM_NAMING_DICT.keys()) + list(ID_DICT.keys())
)
PANEL_NAMING_DICT = {**NAMING_DICT, **PFM_NAMING_DICT, **ID_DICT}


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

# for var, _ in RAW_PFM_NAMING_DICT.items():
#     pfm[var] = pfm[var] * (pfm["weight"] / 2000)

# Define columns to average
AGG_VARS = [_ for _ in X_VAR_PANEL if _ not in ["winner", "loser", "neutral"]]

# Define aggregation dict
agg_dict = {
    col: lambda x: np.average(x, weights=pfm.loc[x.index, "weight"]) for col in AGG_VARS
}
# agg_dict = {col: (lambda x: x.mean()) for col in AGG_VARS}


# Include winner/loser/neutral as mode or max (binary), weight as sum
agg_dict.update(
    {
        "winner": "max",
        "loser": "max",
        "neutral": "max",
        "creator": "max",
        "sniper": "max",
    }
)

# Perform weighted aggregation
pfm = pfm.groupby("trader_address").agg(agg_dict).reset_index()

winner = pfm.loc[pfm["winner"] == 1]
loser = pfm.loc[pfm["loser"] == 1]
neutral = pfm.loc[pfm["neutral"] == 1]


# LaTeX header
lines = [
    "\\begin{tabular}{lccc}",
    "\\toprule",
    "& \\multicolumn{3}{c}{Mean} \\\\",
    "\\cmidrule{2-4}",
    "& Winner & Neutral & Loser \\\\",
    "\\midrule",
]

# Mean values only
for var in [_ for _ in X_VAR_PANEL if _ not in ["winner", "loser", "neutral"]]:
    if var not in pfm.columns:
        continue
    mean_w = winner[var].mean()
    mean_n = neutral[var].mean()
    mean_l = loser[var].mean()
    if var in [
        "raw_pre_migration_duration",
        "raw_pump_duration",
        "raw_dump_duration",
    ]:
        lines.append(
            f"{PANEL_NAMING_DICT[var]} & {mean_w:,.0f} & {mean_n:,.0f} & {mean_l:,.0f} \\\\"
        )
    elif var in ["raw_number_of_traders"]:
        lines.append(
            f"{PANEL_NAMING_DICT[var]} & {mean_w:,.2f} & {mean_n:,.2f} & {mean_l:,.2f} \\\\"
        )
    else:
        lines.append(
            f"{PANEL_NAMING_DICT[var]} & {mean_w:.2f} & {mean_n:.2f} & {mean_l:.2f} \\\\"
        )

# Observation counts
lines.append("\\midrule")
lines.append(f"Observations & {len(winner):,} & {len(neutral):,} & {len(loser):,} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")

# Write to .tex
with open(TABLE_PATH / "wln_bot_diff.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
