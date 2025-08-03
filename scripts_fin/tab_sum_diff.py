"""
Calculate the difference in means of variables before and after Trump
"""

import pandas as pd
import statsmodels.api as sm
from environ.constants import (
    TABLE_PATH,
    PROCESSED_DATA_PATH,
    NAMING_DICT,
    RAW_PFM_NAMING_DICT,
)
from environ.utils import asterisk

# Load data
pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")
pfm["post_trump"] = pfm["chain"].apply(lambda x: 0 if "pre_trump" in x else 1)

# Define variables
X_VAR_PANEL = list(NAMING_DICT.keys()) + list(RAW_PFM_NAMING_DICT.keys())
PANEL_NAMING_DICT = {**NAMING_DICT, **RAW_PFM_NAMING_DICT}

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


def format_latex_line(
    var_name: str,
    s: dict,
    fmt_mean: str = ".2f",
    fmt_diff: str = ".2f",
):
    """Helper function to format a LaTeX table line."""
    return (
        f"{var_name} & {s['mean_1']:{fmt_mean}} & "
        f"{s['mean_2']:{fmt_mean}} & {s['diff']:{fmt_diff}} & "
        f"{s['t_stat']:.2f}{s['stars']} \\\\"
    )


for var in X_VAR_PANEL:
    if var in results:
        s = results[var]
        if var in [
            "raw_pre_migration_duration",
            "raw_pump_duration",
            "raw_dump_duration",
        ]:
            fmt_mean = ",.0f"
            fmt_diff = ",.0f"
        else:
            fmt_mean = ".2f"
            fmt_diff = ".2f"
        latex_lines.append(
            format_latex_line(PANEL_NAMING_DICT[var], s, fmt_mean, fmt_diff)
        )

latex_lines.extend(["\\bottomrule", "\\end{tabular}"])

with open(TABLE_PATH / "reg_var_diff.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
