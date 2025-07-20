"""Script to regress factors against returns and survival."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import statsmodels.api as sm

from environ.constants import (
    NAMING_DICT,
    PROCESSED_DATA_PATH,
    TABLE_PATH,
)
from environ.utils import asterisk

# Regression column name
Y_VAR = "migration"

# Dict for labeling columns
MIGRATION_NAMING_DICT = {
    "all": "$\\text{All}$",
    "pre": "$\\text{Pre-Trump}$",
    "post": "$\\text{Post-Trump}$",
}

PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    **MIGRATION_NAMING_DICT,
    "const": "$\\text{Constant}$",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
}

X_VAR_LIST = list(NAMING_DICT.keys())


def logit_regression(
    df: pd.DataFrame, x_var_list: List[str], y_var: str = "migration"
) -> Dict[str, Any]:
    """Run regression and return result dict."""
    df_reg = df.dropna(subset=x_var_list + [y_var]).copy()
    X = sm.add_constant(df_reg[x_var_list])
    y = df_reg[y_var]
    model = sm.Logit(y, X).fit()
    return {
        "model": model,
        "params": model.params,
        "pvalues": model.pvalues,
        "bse": model.bse,
        "r2": model.prsquared,
        "nobs": int(model.nobs),
    }


def render_latex_table(
    results: Dict[str, Dict[str, Any]], x_var_list: List[str]
) -> str:
    """Render regression results into a LaTeX table."""
    keys = list(results.keys())
    col_len = len(results)
    lines = []

    lines.append("\\begin{tabular}{l" + "c" * len(keys) + "}")
    lines.append("\\hline")
    lines.append(
        r" & "
        + r"\multicolumn{"
        + str(col_len)
        + r"}{c}{"
        + r"\text{Migration}_{i}"
        + r"}"
        + r" \\"
    )
    lines.append("\cline{2-" + str(len(keys) + 1) + "}")
    lines.append(" & " + " & ".join([MIGRATION_NAMING_DICT[k] for k in keys]) + r"\\")
    lines.append(
        " & " + " & ".join([f"({i})" for i in range(1, len(keys) + 1)]) + r"\\"
    )
    lines.append("\\hline")

    for var in x_var_list + ["const"]:
        row_coef = PROFIT_NAMING_DICT.get(var, var)
        row_stderr = ""
        for key in keys:
            model_res = results[key]
            if var in model_res["params"]:
                coef = model_res["params"][var]
                stderr = model_res["bse"][var]
                pval = model_res["pvalues"][var]
                row_coef += f" & {coef:.2f}{asterisk(pval)}"
                row_stderr += f" & ({stderr:.2f})"
            else:
                row_coef += " & "
                row_stderr += " & "
        lines.append(row_coef + r" \\")
        lines.append(row_stderr + r" \\")

    # Add Obs and R²
    obs_row = (
        PROFIT_NAMING_DICT["obs"]
        + " "
        + " ".join(f"& {results[key]['nobs']}" for key in keys)
        + r" \\"
    )
    r2_row = (
        PROFIT_NAMING_DICT["r2"]
        + " "
        + " ".join(f"& {results[key]['r2']:.2f}" for key in keys)
        + r" \\"
    )

    lines.append(obs_row)
    lines.append(r2_row)
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


if __name__ == "__main__":

    # Load and combine datasets
    pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")

    # Define migration variable
    pfm["migration"] = pfm["chain"].apply(
        lambda x: 1 if x in ["raydium", "pre_trump_raydium"] else 0
    )
    pfm["period"] = pfm["chain"].apply(
        lambda x: "Pre-Trump" if "pre_trump" in x else "Post-Trump"
    )

    # Run regressions for all, pre-trump, and post-trump
    results = {
        "all": logit_regression(pfm, X_VAR_LIST),
        "pre": logit_regression(pfm[pfm["period"] == "Pre-Trump"], X_VAR_LIST),
        "post": logit_regression(pfm[pfm["period"] == "Post-Trump"], X_VAR_LIST),
    }

    latex_table = render_latex_table(results, X_VAR_LIST)
    with open(TABLE_PATH / "reg_migration.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
