"""Generate summary statistics for project and trader characteristics"""

import pandas as pd
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
    PROFIT_NAMING_DICT,
)

# Define variable groups
X_VAR_PANEL_A = list(NAMING_DICT.keys()) + list(PFM_NAMING_DICT.keys())
X_VAR_PANEL_B = list(PROFIT_NAMING_DICT.keys())

# Load data
pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")
pft = pd.read_csv(f"{PROCESSED_DATA_PATH}/profit.csv")


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


summary_a = compute_summary(pfm, X_VAR_PANEL_A)
summary_b = compute_summary(pft, X_VAR_PANEL_B)

# Generate LaTeX code
latex_lines = [
    "\\begin{tabular}{lcccccc}",
    "\\hline",
    "Variable & Num. Obs. & Mean & Std. Dev. & P10 & Median & P90 \\\\",
    "\\hline",
    "\\textbf{Panel A. Project characteristics} \\\\",
]

PANEL_A_NAMING_DICT = {
    **NAMING_DICT,
    **PFM_NAMING_DICT,
}

PABE_NAMING_DICT = {
    **PROFIT_NAMING_DICT,
}

for var in X_VAR_PANEL_A:
    if var in summary_a:
        s = summary_a[var]
        latex_lines.append(
            f"{PANEL_A_NAMING_DICT[var]} & {s['num_obs']} & {s['mean']:.2f} & {s['std']:.2f} & {s['p10']:.2f} & {s['median']:.2f} & {s['p90']:.2f} \\\\"
        )

latex_lines.append("\\addlinespace")
latex_lines.append("\\textbf{Panel B. Trader characteristics} \\\\")

for var in X_VAR_PANEL_B:
    if var in summary_b:
        s = summary_b[var]
        latex_lines.append(
            f"{PABE_NAMING_DICT[var]} & {s['num_obs']} & {s['mean']:.2f} & {s['std']:.2f} & {s['p10']:.2f} & {s['median']:.2f} & {s['p90']:.2f} \\\\"
        )

latex_lines.extend(["\\hline", "\\end{tabular}"])

# Save to file
with open(TABLE_PATH / "reg_var_summary.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
