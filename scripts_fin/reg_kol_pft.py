"""Script to regress profit data across Winner, Loser, and Neutral trader groups."""

import pandas as pd
import statsmodels.api as sm
from typing import List, Dict, Any

from environ.constants import (
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    PROFIT_NAMING_DICT,
)

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

# Define the interaction variables
X_VAR_KOL_INTERACTION = list(NAMING_DICT.keys())
Y_VAR = "profit"

# Update the naming dictionary for rendering
PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    "profit": "$\\text{Profit}_{i,j}$",
    "winner_coef": "$\\text{Winner}_{i,j}$",
    "winner_stderr": "",
    "winner_x_var_coef": "$\\text{Winner}_{i,j} \\times \\text{Bot}_i$",
    "winner_x_var_stderr": "",
    "loser_coef": "$\\text{Loser}_{i,j}$",
    "loser_stderr": "",
    "loser_x_var_coef": "$\\text{Loser}_{i,j} \\times \\text{Bot}_i$",
    "loser_x_var_stderr": "",
    "neutral_coef": "$\\text{Neutral}_{i,j}$",
    "neutral_stderr": "",
    "neutral_x_var_coef": "$\\text{Neutral}_{i,j} \\times \\text{Bot}_i$",
    "neutral_x_var_stderr": "",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
}


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


def run_regression_three_groups(
    df: pd.DataFrame, x_var: str, y_var: str
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS regression with interactions for Winner, Loser, and Neutral."""
    X = pd.DataFrame(
        {
            "winner": df["winner"],
            f"winner_x_{x_var}": df["winner"] * df[x_var],
            "loser": df["loser"],
            f"loser_x_{x_var}": df["loser"] * df[x_var],
            "neutral": df["neutral"],
            f"neutral_x_{x_var}": df["neutral"] * df[x_var],
        }
    )
    # add constant for intercept
    y = df[y_var]
    return sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["trader_address"]},
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
    lines.append("\cline{2-" + str(col_len + 1) + "}")
    lines.append(r" $\text{Bot}:$ & " + " & ".join(var_names) + r" \\")
    lines.append(" & " + " & ".join([f"({i})" for i in range(1, col_len + 1)]) + r"\\")
    lines.append(r"\midrule")
    for key in [
        "winner_coef",
        "winner_stderr",
        "winner_x_var_coef",
        "winner_x_var_stderr",
        "loser_coef",
        "loser_stderr",
        "loser_x_var_coef",
        "loser_x_var_stderr",
        "neutral_coef",
        "neutral_stderr",
        "neutral_x_var_coef",
        "neutral_x_var_stderr",
        "obs",
        "r2",
    ]:
        if key in res_dict:
            display_name = PROFIT_NAMING_DICT.get(key, key)
            row = (
                display_name + " " + " ".join(f"& {v}" for v in res_dict[key]) + r" \\"
            )
            lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    pft = pd.read_csv(f"{PROCESSED_DATA_PATH}/pft.csv")
    # Ensure all group indicators exist
    pft = pft.loc[(pft["winner"] == 1) | (pft["neutral"] == 1) | (pft["loser"] == 1)]

    res_dict: Dict[str, List[Any]] = {
        k: []
        for k in [
            "sample",
            "winner_coef",
            "winner_stderr",
            "winner_x_var_coef",
            "winner_x_var_stderr",
            "loser_coef",
            "loser_stderr",
            "loser_x_var_coef",
            "loser_x_var_stderr",
            "neutral_coef",
            "neutral_stderr",
            "neutral_x_var_coef",
            "neutral_x_var_stderr",
            "obs",
            "r2",
        ]
    }

    var_names = []

    for sample, df in [
        ("All", pft),
    ]:
        for x_var in X_VAR_KOL_INTERACTION:
            model = run_regression_three_groups(df, x_var, Y_VAR)
            print(model.summary())  # Optional for inspection
            var_names.append(PROFIT_NAMING_DICT.get(x_var, x_var))

            for var, dict_prefix in [
                ("winner", "winner"),
                (f"winner_x_{x_var}", "winner_x_var"),
                ("loser", "loser"),
                (f"loser_x_{x_var}", "loser_x_var"),
                ("neutral", "neutral"),
                (f"neutral_x_{x_var}", "neutral_x_var"),
            ]:
                res_dict["sample"].append(sample)
                res_dict[f"{dict_prefix}_coef"].append(
                    f"{model.params[var]:.2f}{asterisk(model.pvalues[var])}"
                    if var in model.params
                    else ""
                )
                res_dict[f"{dict_prefix}_stderr"].append(
                    f"({model.params[var] / model.bse[var]:.2f})"
                    if var in model.params
                    else ""
                )

            res_dict["obs"].append(f"{int(model.nobs)}")
            res_dict["r2"].append(f"{model.rsquared:.2f}")

    latex_str = render_latex_table(var_names, res_dict)

    with open(TABLE_PATH / "reg_wln_pft.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)
