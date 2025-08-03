"""Generate summary statistics for project and trader characteristics"""

import json
import pandas as pd
import numpy as np
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
    RAW_PFM_NAMING_DICT,
)

# Define variable groups
X_VAR_PANEL = (
    list(NAMING_DICT.keys())
    + list(PFM_NAMING_DICT.keys())
    + list(RAW_PFM_NAMING_DICT.keys())
)

CHAINS = [
    "raydium",
    "pre_trump_raydium",
    "pumpfun",
    "pre_trump_pumpfun",
    "no_one_care",
    "pre_trump_no_one_care",
]

with open(PROCESSED_DATA_PATH / "meme_project_ratio.json", "r", encoding="utf-8") as f:
    meme_project_ratio = json.load(f)

# Load performance data
pfm = []
for chain in CHAINS:
    pfm_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm_{chain}.csv")
    pfm_df["category"] = chain
    pfm_df["weight"] = meme_project_ratio[chain]
    pfm.append(pfm_df)
pfm = pd.concat(pfm, ignore_index=True)

for var, _ in PFM_NAMING_DICT.items():
    pfm["raw_" + var] = (
        round(np.exp(pfm[var]) - 1, 0) if var != "max_ret" else np.exp(pfm[var]) - 1
    )
    pfm[var] = pfm[var].clip(upper=pfm[var].quantile(0.99))
    pfm[f"raw_{var}"] = pfm[f"raw_{var}"].clip(upper=pfm[f"raw_{var}"].quantile(0.99))

pfm.to_csv(f"{PROCESSED_DATA_PATH}/pfm.csv", index=False)

# Load profit data
pft = []
for chain in CHAINS:
    pft_df = pd.read_csv(f"{PROCESSED_DATA_PATH}/pft_{chain}.csv")
    pft_df["category"] = chain
    pft_df["weight"] = meme_project_ratio[chain]
    pft.append(pft_df)
pft = pd.concat(pft, ignore_index=True)
pft.to_csv(f"{PROCESSED_DATA_PATH}/pft.csv", index=False)


# Compute summary statistics
def compute_summary(df, var_list):
    """Compute summary statistics for a list of variables in a DataFrame."""
    summary = {}
    for var in var_list:
        if var not in df:
            continue
        values = df[var].dropna()
        summary[var] = {
            "num_obs": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "p10": values.quantile(0.1),
            "median": values.median(),
            "p90": values.quantile(0.9),
        }
    return summary


# get the inital observations number of pfm
initial_obs_num = len(pfm)
pre_migration_duration_obs_num = len(pfm["pre_migration_duration"].dropna())

# Repeat rows based on weight and compute summary
pfm = pfm.loc[pfm.index.repeat(pfm["weight"])].reset_index(drop=True)

summary = compute_summary(pfm, X_VAR_PANEL)

# Generate LaTeX code
latex_lines = [
    "\\begin{tabular}{lccccccc}",
    "\\toprule",
    "Variable & Num. Obs. & Mean & Std. Dev. & P10 & Median & P90\\\\",
    "\\midrule",
]

PANEL_A_NAMING_DICT = {
    **NAMING_DICT,
    **PFM_NAMING_DICT,
    **RAW_PFM_NAMING_DICT,
}


def format_latex_line(
    var_name: str,
    obs: int,
    s: dict,
    fmt_mean: str,
    fmt_std: str,
    fmt_p10: str,
    fmt_median: str,
    fmt_p90: str,
):
    """Helper function to format a LaTeX table line."""
    return (
        f"{var_name} & {obs:,} & "
        f"{s['mean']:{fmt_mean}} & {s['std']:{fmt_std}} & "
        f"{s['p10']:{fmt_p10}} & {s['median']:{fmt_median}} & {s['p90']:{fmt_p90}}\\\\"
    )


for var in {**NAMING_DICT, **PFM_NAMING_DICT}:
    if var in PFM_NAMING_DICT:
        s = summary[f"raw_{var}"]
        obs = (
            initial_obs_num
            if var != "pre_migration_duration"
            else pre_migration_duration_obs_num
        )

        if var == "max_ret":
            line = format_latex_line(
                PANEL_A_NAMING_DICT[f"raw_{var}"],
                obs,
                s,
                ".2f",
                ".2f",
                ".2f",
                ".2f",
                ".2f",
            )
        elif var == "number_of_traders":
            line = format_latex_line(
                PANEL_A_NAMING_DICT[f"raw_{var}"],
                obs,
                s,
                ",.2f",
                ",.2f",
                ",.0f",
                ",.0f",
                ",.0f",
            )
        else:
            line = format_latex_line(
                PANEL_A_NAMING_DICT[f"raw_{var}"],
                obs,
                s,
                ",.0f",
                ",.0f",
                ",.0f",
                ",.0f",
                ",.0f",
            )
        latex_lines.append(line)

    s = summary[var]
    obs = (
        initial_obs_num
        if var != "pre_migration_duration"
        else pre_migration_duration_obs_num
    )

    if var in NAMING_DICT:
        fmt = (".2f", ".2f", ".0f", ".0f", ".0f")
    else:
        fmt = (".2f", ".2f", ".2f", ".2f", ".2f")
    latex_lines.append(format_latex_line(PANEL_A_NAMING_DICT[var], obs, s, *fmt))

latex_lines.extend(["\\bottomrule", "\\end{tabular}"])

# Save to file
with open(TABLE_PATH / "reg_var_summary.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
