"""This script compares financial metrics across Winner, Loser, and Neutral projects in a single LaTeX table with mean columns."""

import pandas as pd
from scipy.stats import ttest_ind
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
    ID_DICT,
)
from ast import literal_eval


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


# Load project performance data
# trader_t = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
# trader_t = trader_t.loc[trader_t["meme_num"] <= 1000].dropna(subset=["t_stat"])
# trader_t["winner"] = trader_t["t_stat"] > 2.576
# trader_t["loser"] = trader_t["t_stat"] < -2.576
# trader_t["neutral"] = trader_t["t_stat"].abs() <= 2.576

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
    pfm.groupby(["trader_address"])[["winner", "loser", "neutral", *X_VAR_PANEL]]
    .mean()
    .reset_index()
)

# If the creator and sniper greater than 0, set to 1, else 0
pfm["creator"] = pfm["creator"].apply(lambda x: int(x > 0))
pfm["sniper"] = pfm["sniper"].apply(lambda x: int(x > 0))

winner = pfm.loc[pfm["winner"] == 1]
loser = pfm.loc[pfm["loser"] == 1]
neutral = pfm.loc[pfm["neutral"] == 1]


def ttest_wrapper(df1, df2, var):
    """Perform t-test and return difference and t-statistic."""
    x, y = df1[var].dropna(), df2[var].dropna()
    if len(x) < 2 or len(y) < 2:
        return "", ""
    t_stat, p_val = ttest_ind(y, x, equal_var=False)
    diff = y.mean() - x.mean()
    return f"{diff:.2f}", f"{t_stat:.2f}{significance_stars(p_val)}"


# LaTeX header
lines = [
    "\\begin{tabular}{lcccccccccccc}",
    "\\hline",
    "& Neutral & Winner & & & Neutral & Loser & & & Winner & Loser & & \\\\",
    "\\cline{2-13}",
    "& Mean & Mean & Diff & t & Mean & Mean & Diff & t & Mean & Mean & Diff & t \\\\",
    "\\hline",
]


# Loop through all variables and compute comparisons
for var in X_VAR_PANEL:
    if var not in neutral.columns:
        continue
    # Means
    mean_n = neutral[var].mean()
    mean_l = loser[var].mean()
    mean_w = winner[var].mean()

    # T-tests
    diff_wn, t_wn = ttest_wrapper(neutral, winner, var)
    diff_ln, t_ln = ttest_wrapper(neutral, loser, var)
    diff_wl, t_wl = ttest_wrapper(winner, loser, var)

    lines.append(
        f"{PANEL_NAMING_DICT[var]} & "
        f"{mean_n:.2f} & {mean_w:.2f} & {diff_wn} & {t_wn} & "
        f"{mean_n:.2f} & {mean_l:.2f} & {diff_ln} & {t_ln} & "
        f"{mean_w:.2f} & {mean_l:.2f} & {diff_wl} & {t_wl} \\\\"
    )

obs_n = len(neutral)
obs_w = len(winner)
obs_l = len(loser)

lines.append(
    "\\hline",
)

lines.append(
    f"Observations & {obs_n} & {obs_w} & & & "
    f"{obs_n} & {obs_l} & & & "
    f"{obs_w} & {obs_l} & & \\\\"
)

# LaTeX footer
lines.extend(["\\hline", "\\end{tabular}"])

# Write to .tex
with open(TABLE_PATH / "wln_bot_diff.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
