"""This script compares two DataFrames containing project financial metrics"""

import pandas as pd
from scipy.stats import ttest_ind
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
)


def significance_stars(p):
    """Return asterisks based on p-value significance levels."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


# Define variable groups
X_VAR_PANEL = list(NAMING_DICT.keys()) + list(PFM_NAMING_DICT.keys())

pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")


def compare_groups(df1, df2, var_list):
    """Compare two DataFrames and return a summary of differences."""
    results = {}
    for var in var_list:
        if var not in df1 or var not in df2:
            continue
        x = df1[var].dropna()
        y = df2[var].dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        t_stat, p_val = ttest_ind(y, x, equal_var=False)
        results[var] = {
            "mean_1": x.mean(),
            "mean_2": y.mean(),
            "diff": y.mean() - x.mean(),
            "t_stat": t_stat,
            "stars": significance_stars(p_val),
        }
    return results


pfm["period"] = pfm["chain"].apply(
    lambda x: "Pre-Trump" if "pre_trump" in x else "Post-Trump"
)

summary = compare_groups(
    pfm.loc[pfm["period"] == "Pre-Trump"],
    pfm.loc[pfm["period"] == "Post-Trump"],
    X_VAR_PANEL,
)

# Prepare LaTeX table lines
latex_lines = [
    "\\begin{tabular}{lcccr}",
    "\\hline",
    "Variable & Pre-Trump & Post-Trump & Diff & t \\\\",
    "\\hline",
]

PANEL_NAMING_DICT = {**NAMING_DICT, **PFM_NAMING_DICT}

for var in X_VAR_PANEL:
    if var in summary:
        s = summary[var]
        latex_lines.append(
            f"{PANEL_NAMING_DICT[var]} & "
            f"{s['mean_1']:.2f} & {s['mean_2']:.2f} & {s['diff']:.2f} & "
            f"{s['t_stat']:.2f}{s['stars']} \\\\"
        )

latex_lines.extend(["\\hline", "\\end{tabular}"])

# Write to .tex file
with open(TABLE_PATH / "reg_var_diff.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
