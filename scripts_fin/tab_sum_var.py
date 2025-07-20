"""Generate summary statistics for project and trader characteristics"""

import pandas as pd
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
    KOL_NAMING_DICT,
)

# Define variable groups
X_VAR_PANEL = (
    list(NAMING_DICT.keys())
    + list(PFM_NAMING_DICT.keys())
    + list(KOL_NAMING_DICT.keys())
)

# Load performance data
pfm = []
for chain in ["raydium", "pre_trump_raydium", "pumpfun", "pre_trump_pumpfun"]:
    pfm.append(pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm_{chain}.csv"))
pfm = pd.concat(pfm, ignore_index=True)
for var in ["winner", "loser"]:
    pfm[var] = pfm[var].apply(lambda x: 1 if x else 0)

# winsorize the max return
pfm["max_ret"] = pfm["max_ret"].clip(upper=pfm["max_ret"].quantile(0.99))
pfm.to_csv(f"{PROCESSED_DATA_PATH}/pfm.csv", index=False)

# Load profit data
pft = []
for chain in ["raydium", "pre_trump_raydium", "pumpfun", "pre_trump_pumpfun"]:
    pft.append(pd.read_csv(f"{PROCESSED_DATA_PATH}/pft_{chain}.csv"))
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


summary = compute_summary(pfm, X_VAR_PANEL)

# Generate LaTeX code
latex_lines = [
    "\\begin{tabular}{lcccccc}",
    "\\hline",
    "Variable & Num. Obs. & Mean & Std. Dev. & P10 & Median & P90 \\\\",
    "\\hline",
]

PANEL_A_NAMING_DICT = {
    **NAMING_DICT,
    **PFM_NAMING_DICT,
    **KOL_NAMING_DICT,
}

for var in X_VAR_PANEL:
    if var in summary:
        s = summary[var]
        latex_lines.append(
            f"{PANEL_A_NAMING_DICT[var]} & {s['num_obs']} & {s['mean']:.2f} & {s['std']:.2f} & {s['p10']:.2f} & {s['median']:.2f} & {s['p90']:.2f} \\\\"
        )

latex_lines.extend(["\\hline", "\\end{tabular}"])

# Save to file
with open(TABLE_PATH / "reg_var_summary.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
