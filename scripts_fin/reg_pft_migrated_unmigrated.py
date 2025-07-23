"""Script to regress profit data using regression analysis."""

import pandas as pd
import statsmodels.api as sm

from environ.constants import (
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    PROFIT_NAMING_DICT,
)

from typing import List, Dict, Any

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
    return sm.OLS(y, X).fit()


def render_latex_two_panel_table(
    var_names: List[str],
    res_dicts: Dict[str, Dict[str, List[str]]],
    y_var: str = "profit",
) -> str:
    """Render the regression results for migrated and unmigrated as two panels in one LaTeX table."""
    panels = ["Unmigrated", "Migrated"]
    lines = []
    col_len = len(res_dicts["unmigrated"]["obs"])
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
    lines.append(" & " + " & ".join([f"({i+1})" for i in range(col_len)]) + r"\\")

    for i, panel in enumerate(panels):
        res_dict = res_dicts[panel.lower()]
        if i == 0:
            lines.append(r"\midrule")
            lines.append(r"\textbf{Panel A. Unmigrated Meme Coins}\\")
            lines.append(r"\midrule")
        else:
            lines.append(r"\midrule")
            lines.append(r"\textbf{Panel B. Migrated Meme Coins}\\")
            lines.append(r"\midrule")
        col_len = len(res_dict["obs"])
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
                    display_name
                    + " "
                    + " ".join(f"& {v}" for v in res_dict[key])
                    + r" \\"
                )
                if key == "obs":
                    lines.append(r"\midrule")
                lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    pft = pd.read_csv(f"{PROCESSED_DATA_PATH}/pft.csv")
    samples = {
        "unmigrated": pft.loc[
            pft["category"].isin(["pumpfun", "pre_trump_pumpfun"])
        ].copy(),
        "migrated": pft.loc[
            pft["category"].isin(["raydium", "pre_trump_raydium"])
        ].copy(),
    }

    res_dicts = {}
    var_names = []

    for sample_name in ["unmigrated", "migrated"]:
        df = samples[sample_name]
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
        for x_var in X_VAR_CREATOR_INTERACTION:
            model = run_regression(df, x_var, Y_VAR)
            if sample_name == "unmigrated":  # populate var_names only once
                var_names.append(PROFIT_NAMING_DICT.get(x_var, x_var))
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
        res_dicts[sample_name] = res_dict

    # Generate combined panel table
    latex_str = render_latex_two_panel_table(var_names, res_dicts, y_var=Y_VAR)
    with open(
        TABLE_PATH / "reg_pft_migrated_unmigrated.tex", "w", encoding="utf-8"
    ) as f:
        f.write(latex_str)
