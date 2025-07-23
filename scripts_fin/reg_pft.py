"""Script to regress profit data using regression analysis."""

import pandas as pd
import statsmodels.api as sm

from environ.constants import (
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    NAMING_DICT,
    PROFIT_NAMING_DICT,
)

from typing import List, Dict, Any

X_VAR_LIST = ["creator"]

X_VAR_CREATOR_INTERACTION = NAMING_DICT.keys()

# Extend with additional profit-related naming
PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    "profit": "$\\text{Profit}$",
    "creator_coef": "$\\text{Creator}$",
    "creator_stderr": "",
    "creator_x_var_coef": "$\\text{Creator} \\times \\text{Bot}$",
    "creator_x_var_stderr": "",
    "non_creator_coef": "$\\text{Non-Creator}$",
    "non_creator_stderr": "",
    "non_creator_x_var_coef": "$\\text{Non-Creator} \\times \\text{Bot}$",
    "non_creator_x_var_stderr": "",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
}

Y_VAR = "profit"


def asterisk(pval: float) -> str:
    """Return asterisks based on standard significance levels."""
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    else:
        return ""


def run_regression(
    df: pd.DataFrame, x_var: str, y_var: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression for the given x_var (optionally with creator interaction).
    """
    X = pd.DataFrame(
        {
            "creator": df["creator"],
            f"creator_x_{x_var}": df["creator"] * df[x_var],
            "non_creator": 1 - df["creator"],
            f"non_creator_x_{x_var}": (1 - df["creator"]) * df[x_var],
        }
    )
    y = df[y_var]
    return sm.OLS(y, X).fit()


def render_latex_table(
    var_names: List[str], res_dict: Dict[str, List[str]], y_var: str = "profit"
) -> str:
    """Render the regression results as a LaTeX table."""
    col_len = len(res_dict["obs"])

    lines = []
    lines.append("\\begin{tabular}{l" + "c" * col_len + "}")
    lines.append(r"\hline")
    lines.append(
        r" & "
        + r"\multicolumn{"
        + str(col_len)
        + r"}{c}{"
        + PROFIT_NAMING_DICT[y_var]
        + r"}"
        + r" \\"
    )
    lines.append(
        r" & \multicolumn{4}{c}{All} & \multicolumn{4}{c}{Unmigrated} & \multicolumn{4}{c}{Migrated} \\"
    )
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}")
    lines.append(r" $\text{Bot}:$ & " + " & ".join(var_names) + r" \\")
    lines.append(" & " + " & ".join([f"({i})" for i in range(1, col_len + 1)]) + r"\\")
    lines.append(r"\hline")
    for key in [
        "creator_coef",
        "creator_stderr",
        "creator_x_var_coef",
        "creator_x_var_stderr",
        "non_creator_coef",
        "non_creator_stderr",
        "non_creator_x_var_coef",
        "non_creator_x_var_stderr",
        "obs",
        "r2",
    ]:
        if key in res_dict:
            display_name = PROFIT_NAMING_DICT.get(key, key)
            row = (
                display_name + " " + " ".join(f"& {v}" for v in res_dict[key]) + r" \\"
            )
            lines.append(row)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    pft = pd.read_csv(f"{PROCESSED_DATA_PATH}/pft.csv")
    # Ensure all group indicators exist
    unmigrated = pft.loc[pft["category"].isin(["pumpfun", "pre_trump_pumpfun"])].copy()
    migrated = pft.loc[pft["category"].isin(["raydium", "pre_trump_raydium"])].copy()

    res_dict: Dict[str, List[Any]] = {
        k: []
        for k in [
            "sample",
            "creator_coef",
            "creator_stderr",
            "creator_x_var_coef",
            "creator_x_var_stderr",
            "non_creator_coef",
            "non_creator_stderr",
            "non_creator_x_var_coef",
            "non_creator_x_var_stderr",
            "con",
            "con_stderr",
            "obs",
            "r2",
        ]
    }
    var_names = []

    for sample, df in [
        ("All", pft),
        ("Unmigrated", unmigrated),
        ("Migrated", migrated),
    ]:
        for x_var in X_VAR_CREATOR_INTERACTION:
            model = run_regression(df, x_var, Y_VAR)
            print(model.summary())  # Optional: remove if you don't want console output
            var_names.append(PROFIT_NAMING_DICT.get(x_var, x_var))

            # Main effects
            for var, dict_name in [
                ("creator", "creator"),
                (f"creator_x_{x_var}", "creator_x_var"),
                ("non_creator", "non_creator"),
                (f"non_creator_x_{x_var}", "non_creator_x_var"),
            ]:
                res_dict["sample"].append(sample)
                res_dict[f"{dict_name}_coef"].append(
                    f"{model.params[var]:.2f}{asterisk(model.pvalues[var])}"
                    if var in model.params
                    else ""
                )
                res_dict[f"{dict_name}_stderr"].append(
                    f"({model.bse[var]:.2f})" if var in model.bse else ""
                )
            res_dict["obs"].append(f"{int(model.nobs)}")
            res_dict["r2"].append(f"{model.rsquared:.2f}")

    latex_str = render_latex_table(var_names, res_dict)

    with open(TABLE_PATH / "reg_pft.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
