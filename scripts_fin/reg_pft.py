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

X_VAR_CREATOR_INTERACTION = [
    "launch_bundle",
    "bundle_buy",
    "bundle_sell",
    "volume_bot",
    "positive_bot_comment_num",
    "negative_bot_comment_num",
]

# Extend with additional profit-related naming
PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    "profit": "$\\text{Profit}$",
    "creator_x_non_var_coef": "$\\text{Creator} \\times \\text{Non-Bot}$",
    "creator_x_non_var_stderr": "",
    "creator_x_var_coef": "$\\text{Creator} \\times \\text{Bot}$",
    "creator_x_var_stderr": "",
    "non_creator_x_non_var_coef": "$\\text{Non-Creator} \\times \\text{Non-Bot}$",
    "non_creator_x_non_var_stderr": "",
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
            f"creator_x_non_{x_var}": df["creator"] * (1 - df[x_var]),
            f"creator_x_{x_var}": df["creator"] * df[x_var],
            f"non_creator_x_non_{x_var}": (1 - df["creator"]) * (1 - df[x_var]),
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
    lines.append(r" $\text{Bot}:$ & " + " & ".join(var_names) + r" \\")
    lines.append(" & " + " & ".join([f"({i})" for i in range(1, col_len + 1)]) + r"\\")
    lines.append(r"\hline")
    for key in [
        "creator_x_non_var_coef",
        "creator_x_non_var_stderr",
        "creator_x_var_coef",
        "creator_x_var_stderr",
        "non_creator_x_non_var_coef",
        "non_creator_x_non_var_stderr",
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


def main():
    """Main function to run the regression analysis and generate LaTeX table."""
    # Ensure the output directory
    # Read data
    reg_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/pft.csv")

    res_dict: Dict[str, List[Any]] = {
        k: []
        for k in [
            "creator_x_non_var_coef",
            "creator_x_non_var_stderr",
            "creator_x_var_coef",
            "creator_x_var_stderr",
            "non_creator_x_non_var_coef",
            "non_creator_x_non_var_stderr",
            "non_creator_x_var_coef",
            "non_creator_x_var_stderr",
            "con",
            "con_stderr",
            "obs",
            "r2",
        ]
    }
    var_names = []

    for x_var in X_VAR_CREATOR_INTERACTION:
        model = run_regression(reg_tab, x_var, Y_VAR)
        print(model.summary())  # Optional: remove if you don't want console output
        var_names.append(PROFIT_NAMING_DICT.get(x_var, x_var))

        # Main effects
        for var, dict_name in [
            (f"creator_x_non_{x_var}", "creator_x_non_var"),
            (f"creator_x_{x_var}", "creator_x_var"),
            (f"non_creator_x_non_{x_var}", "non_creator_x_non_var"),
            (f"non_creator_x_{x_var}", "non_creator_x_var"),
        ]:
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


if __name__ == "__main__":
    main()
