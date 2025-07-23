"""This script compares financial metrics across Winner, Loser, and Neutral projects in a single LaTeX table with mean columns."""

import pandas as pd
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
    ID_DICT,
)


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

winner = pfm.loc[pfm["winner"] == 1]
loser = pfm.loc[pfm["loser"] == 1]
neutral = pfm.loc[pfm["neutral"] == 1]


# LaTeX table lines
lines = [
    "\\begin{tabular}{lccc}",
    "\\toprule",
    "& \\multicolumn{3}{c}{Mean} \\\\",
    "\\cmidrule{2-4}",
    "& Winner & Neutral & Loser \\\\",
    "\\midrule",
]

# Fill with mean values only
for var in X_VAR_PANEL:
    if var not in pfm.columns:
        continue
    mean_w = winner[var].mean()
    mean_n = neutral[var].mean()
    mean_l = loser[var].mean()
    lines.append(
        f"{PANEL_NAMING_DICT[var]} & {mean_w:.2f} & {mean_n:.2f} & {mean_l:.2f} \\\\"
    )

# Add observation counts
lines.append("\\midrule")
lines.append(f"Observations & {len(winner)} & {len(neutral)} & {len(loser)} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")

# Write to file
with open(TABLE_PATH / "wln_bot_mean.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
