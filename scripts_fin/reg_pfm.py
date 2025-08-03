"""Script to regress factors against returns and survival."""

from typing import Any, Dict, List

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from environ.constants import (
    NAMING_DICT,
    PFM_NAMING_DICT,
    PROCESSED_DATA_PATH,
    TABLE_PATH,
)
from environ.utils import asterisk

PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    **PFM_NAMING_DICT,
    "const": "$\\text{Constant}$",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
}

X_VAR_LIST = list(NAMING_DICT.keys())


def compute_vif(df: pd.DataFrame, x_var_list: List[str]) -> pd.DataFrame:
    """Compute VIF for each explanatory variable (excluding constant)."""
    df_clean = df.dropna(subset=x_var_list).copy()
    X = df_clean[x_var_list]  # exclude constant
    vif = pd.DataFrame()
    vif["Variable"] = x_var_list
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["1/VIF"] = 1 / vif["VIF"]
    return vif


def render_vif_latex_table(vif_df: pd.DataFrame) -> str:
    """Render VIF results into a LaTeX table with 1/VIF and Mean VIF."""
    lines = []
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Variable & VIF & $1/\\text{VIF}$ \\\\")
    lines.append("\\midrule")
    for _, row in vif_df.iterrows():
        var_name = PROFIT_NAMING_DICT.get(row["Variable"], row["Variable"])
        lines.append(f"{var_name} & {row['VIF']:.2f} & {row['1/VIF']:.2f} \\\\")
    lines.append("\\midrule")
    lines.append(
        f"\\textbf{{Mean}} & {vif_df['VIF'].mean():.2f} & {vif_df['1/VIF'].mean():.2f} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def reg_survive(
    df: pd.DataFrame,
    x_var_list: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Run regression analysis on return and survival data."""
    results = {}

    for y_var in PFM_NAMING_DICT:
        df_reg = df.dropna(subset=[y_var]).copy()
        X = sm.add_constant(df_reg[x_var_list])
        y = df_reg[y_var]
        model = sm.WLS(y, X, weights=df_reg["weight"]).fit()

        results[y_var] = {
            "model": model,
            "params": model.params,
            "pvalues": model.pvalues,
            "bse": model.bse,
            "r2": model.rsquared,
            "nobs": int(model.nobs),
        }

    return results


def render_latex_table(
    results: Dict[str, Dict[str, Any]], x_var_list: List[str]
) -> str:
    """Render regression results into a LaTeX table."""
    keys = list(results.keys())
    lines = []

    lines.append("\\begin{tabular}{l" + "c" * len(keys) + "}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join([PFM_NAMING_DICT[key] for key in keys]) + r" \\")
    lines.append(
        " & " + " & ".join([f"({i})" for i in range(1, len(keys) + 1)]) + r"\\"
    )
    lines.append("\\midrule")

    for var in x_var_list + ["const"]:
        row_coef = PROFIT_NAMING_DICT.get(var, var)
        row_t = ""
        for key in keys:
            model_res = results[key]
            if var in model_res["params"]:
                coef = model_res["params"][var]
                stderr = model_res["bse"][var]
                pval = model_res["pvalues"][var]
                row_coef += f" & {coef:.2f}{asterisk(pval)}"
                row_t += f" & ({coef / stderr:.2f})"
            else:
                row_coef += " & "
                row_t += " & "
        lines.append(row_coef + r" \\")
        lines.append(row_t + r" \\")
    lines.append(r"\midrule")
    # Add Obs and R²
    obs_row = (
        PROFIT_NAMING_DICT["obs"]
        + " "
        + " ".join(f"& {results[key]['nobs']:,}" for key in keys)
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
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


if __name__ == "__main__":
    pfm = pd.read_csv(f"{PROCESSED_DATA_PATH}/pfm.csv")

    vif_df = compute_vif(pfm, X_VAR_LIST)
    vif_latex = render_vif_latex_table(vif_df)

    with open(TABLE_PATH / "vif_pfm.tex", "w", encoding="utf-8") as f:
        f.write(vif_latex)

    results = reg_survive(pfm, X_VAR_LIST)
    latex_table = render_latex_table(results, X_VAR_LIST)

    with open(TABLE_PATH / "reg_pfm.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
