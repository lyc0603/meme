"""Script to regress profit data on individual variables using regression analysis."""

import pandas as pd
import statsmodels.api as sm
from typing import List, Dict, Any

from environ.constants import (
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    NAMING_DICT,
    PROFIT_NAMING_DICT,
)

# List of variables for individual regression
X_VAR_LIST = [
    "launch_bundle_transfer",
    "bundle_creator_buy",
    "bundle_launch",
    "bundle_buy",
    "bundle_sell",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
    "positive_bot_comment_num",
    "negative_bot_comment_num",
]


# Flatten NAMING_DICT for LaTeX column naming
def flatten_naming_dict(naming_dict: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Flatten the nested naming dictionary into a single-level dictionary."""
    return {k: v for category in naming_dict.values() for k, v in category.items()}


FLAT_NAMING_DICT = flatten_naming_dict(NAMING_DICT)

# Profit LaTeX naming dictionary
PROFIT_NAMING_DICT = {
    **FLAT_NAMING_DICT,
    "profit": "$\\text{Profit}$",
    "x_var_coef": "$\\text{Bot}$",
    "x_var_stderr": "",
    "con": "$\\text{Constant}$",
    "con_stderr": "",
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
    """Run OLS regression for the given variable (no creator, no interaction)."""
    X = sm.add_constant(df[[x_var]])
    y = df[y_var]
    return sm.OLS(y, X).fit()


def render_latex_table(
    var_names: List[str], res_dict: Dict[str, List[str]], y_var: str = "profit"
) -> str:
    """Render the results dictionary into a LaTeX table string."""
    col_len = len(res_dict["con"])
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
        "x_var_coef",
        "x_var_stderr",
        "con",
        "con_stderr",
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
    """Main function to run the regression analysis on profit data."""
    reg_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/profit.csv")
    reg_tab = reg_tab.loc[reg_tab["creator"] == 0]  # Exclude creator
    res_dict: Dict[str, List[Any]] = {
        k: []
        for k in [
            "x_var_coef",
            "x_var_stderr",
            "con",
            "con_stderr",
            "obs",
            "r2",
        ]
    }
    var_names = []

    for x_var in X_VAR_LIST:
        model = run_regression(reg_tab, x_var, Y_VAR)
        print(model.summary())  # Optional, can be commented out

        # Use LaTeX name if available
        var_names.append(PROFIT_NAMING_DICT.get(x_var, x_var))

        res_dict["x_var_coef"].append(
            f"{model.params[x_var]:.2f}{asterisk(model.pvalues[x_var])}"
        )
        res_dict["x_var_stderr"].append(f"({model.bse[x_var]:.2f})")
        res_dict["con"].append(
            f"{model.params['const']:.2f}{asterisk(model.pvalues['const'])}"
        )
        res_dict["con_stderr"].append(f"({model.bse['const']:.2f})")
        res_dict["obs"].append(f"{int(model.nobs)}")
        res_dict["r2"].append(f"{model.rsquared:.2f}")

    latex_str = render_latex_table(var_names, res_dict)

    with open(TABLE_PATH / "reg_participant_profit.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)


if __name__ == "__main__":
    main()
