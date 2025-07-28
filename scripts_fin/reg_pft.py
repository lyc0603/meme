"""Script to regress profit data using regression analysis."""

from typing import List, Dict, Any

import pandas as pd
import statsmodels.api as sm

from environ.constants import (
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    PROFIT_NAMING_DICT,
)

X_VAR_LIST = ["creator"]

NAMING_DICT = {
    # bundle bots
    "launch_bundle": "$\\text{Rat Bot}$",
    # sniper bots
    "sniper_bot": "$\\text{Sniper Bot}$",
    # volume bots
    "volume_bot": "$\\text{Wash Trading Bot}$",
    # comment bots
    "bot_comment_num": "$\\text{Comment Bot}$",
}

X_VAR_CREATOR_INTERACTION = NAMING_DICT.keys()

# Extend with additional profit-related naming
PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    "profit": "$\\text{Profit}_{i,j}$",
    "creator_coef": "$\\text{Creator}_{i,j}$",
    "creator_stderr": "",
    "creator_x_var_coef": "$\\text{Creator}_{i,j} \\times \\text{Bot}_i$",
    "creator_x_var_stderr": "",
    "non_creator_coef": "$\\text{Non-Creator}_{i,j}$",
    "non_creator_stderr": "",
    "non_creator_x_var_coef": "$\\text{Non-Creator}_{i,j} \\times \\text{Bot}_i$",
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
    return sm.WLS(y, X, weights=df["weight"]).fit(
        cov_type="cluster", cov_kwds={"groups": df["token_address"]}
    )


def render_latex_table(
    var_names: List[str], res_dict: Dict[str, List[str]], y_var: str = "profit"
) -> str:
    """Render the regression results as a LaTeX table."""
    col_len = len(res_dict["obs"])

    lines = []
    lines.append("\\begin{tabular}{l" + "c" * col_len + "}")
    lines.append(r"\toprule")
    lines.append(
        r" & "
        + r"\multicolumn{"
        + str(col_len)
        + r"}{c}{"
        + PROFIT_NAMING_DICT[y_var]
        + r"}"
        + r" \\"
    )
    lines.append(r"\cline{2-" + str(col_len + 1) + "}")
    lines.append(r" $\text{Bot}:$ & " + " & ".join(var_names) + r" \\")
    lines.append(" & " + " & ".join([f"({i})" for i in range(1, col_len + 1)]) + r"\\")
    lines.append(r"\midrule")
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
            if key == "obs":
                lines.append(r"\midrule")
            lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    pft = pd.read_csv(f"{PROCESSED_DATA_PATH}/pft.csv")
    samples = {
        "all": pft,
    }

    for sample_name, df in samples.items():
        res_dict: Dict[str, List[Any]] = {
            k: []
            for k in [
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
            ]
        }
        var_names = []

        for x_var in X_VAR_CREATOR_INTERACTION:
            model = run_regression(df, x_var, Y_VAR)
            print(model.summary())
            var_names.append(PROFIT_NAMING_DICT.get(x_var, x_var))

            # Fill results
            for var, dict_name in [
                ("creator", "creator"),
                (f"creator_x_{x_var}", "creator_x_var"),
                ("non_creator", "non_creator"),
                (f"non_creator_x_{x_var}", "non_creator_x_var"),
            ]:
                res_dict[f"{dict_name}_coef"].append(
                    f"{model.params[var]:.2f}{asterisk(model.pvalues[var])}"
                    if var in model.params
                    else ""
                )
                res_dict[f"{dict_name}_stderr"].append(
                    f"({model.params[var] / model.bse[var]:.2f})"
                    if var in model.params
                    else ""
                )
            res_dict["obs"].append(f"{int(model.nobs)}")
            res_dict["r2"].append(f"{model.rsquared:.2f}")

        # Generate and save table
        latex_str = render_latex_table(var_names, res_dict, y_var=Y_VAR)
        with open(
            TABLE_PATH / f"reg_pft_{sample_name}.tex", "w", encoding="utf-8"
        ) as f:
            f.write(latex_str)
