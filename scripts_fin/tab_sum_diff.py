"""
Calculate the difference in means of variables before and after Trump
"""

import pandas as pd
import statsmodels.api as sm
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    PFM_NAMING_DICT,
)
from environ.utils import asterisk

# Load data
pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")
pfm["post_trump"] = pfm["chain"].apply(lambda x: 0 if "pre_trump" in x else 1)

# Define variables
X_VAR_PANEL = list(NAMING_DICT.keys()) + list(PFM_NAMING_DICT.keys())
PANEL_NAMING_DICT = {**NAMING_DICT, **PFM_NAMING_DICT}

results = {}
for var in X_VAR_PANEL:
    if var not in pfm.columns:
        continue

    # Drop NA rows
    df_var = pfm.dropna(subset=[var, "weight", "post_trump"]).copy()
    if df_var.empty:
        continue

    # Dependent variable: var
    y = df_var[var].astype(float)
    # Independent variable: post dummy
    X = sm.add_constant(df_var["post_trump"].astype(float))
    w = df_var["weight"].astype(float)

    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    df_pre = df_var[df_var["post_trump"] == 0]
    df_post = df_var[df_var["post_trump"] == 1]

    if not df_pre.empty:
        mean_1 = (df_pre[var] * df_pre["weight"]).sum() / df_pre["weight"].sum()
    else:
        mean_1 = float("nan")

    if not df_post.empty:
        mean_2 = (df_post[var] * df_post["weight"]).sum() / df_post["weight"].sum()
    else:
        mean_2 = float("nan")
    # --------------------------------------

    coef = res.params["post_trump"]  # weighted difference in means
    t_stat = res.tvalues["post_trump"]
    p_val = res.pvalues["post_trump"]

    results[var] = {
        "mean_1": mean_1,
        "mean_2": mean_2,
        "diff": coef,
        "t_stat": t_stat,
        "stars": asterisk(p_val),
    }

# Prepare LaTeX table
latex_lines = [
    "\\begin{tabular}{lcccr}",
    "\\toprule",
    "Variable & Pre-Trump & Post-Trump & Diff & t \\\\",
    "\\midrule",
]

for var in X_VAR_PANEL:
    if var in results:
        s = results[var]
        latex_lines.append(
            f"{PANEL_NAMING_DICT[var]} & "
            f"{s['mean_1']:.2f} & {s['mean_2']:.2f} & {s['diff']:.2f} & "
            f"{s['t_stat']:.2f}{s['stars']} \\\\"
        )

latex_lines.extend(["\\bottomrule", "\\end{tabular}"])

with open(TABLE_PATH / "reg_var_diff.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
