"""Script to explain profit data using regression analysis."""

import pandas as pd
import statsmodels.api as sm
from pyfixest.estimation import feols

from environ.constants import (
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    NAMING_DICT,
    PROFIT_NAMING_DICT,
)

x_var_list = [
    "creator",
    # "txn_number",
]

x_var_creator_interaction = [
    "launch_bundle_transfer",
    "bundle_creator_buy",
    "bundle_launch",
    "bundle_buy",
    "bundle_sell",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
    "bot_comment_num",
    "positive_bot_comment_num",
    "negative_bot_comment_num",
]

y_var = "profit"


# Remove the grouping in NAMING_DICT to create a flat dictionary
flat_naming_dict = {
    sub_key: sub_value.strip("$")
    for category in NAMING_DICT.values()
    for sub_key, sub_value in category.items()
}

profit_naming_dict = {
    **PROFIT_NAMING_DICT,
    **{
        f"{k}{stats}": v if stats == "_coef" else ""
        for k, v in PROFIT_NAMING_DICT.items()
        for stats in ["_coef", "_stderr"]
    },
    **{
        f"creator_{k}{stats}": (
            "$\\text{Profit} \\times " + f"{flat_naming_dict[k]}$"
            if stats == "_coef"
            else ""
        )
        for k in x_var_creator_interaction
        for stats in ["_coef", "_stderr"]
    },
    **{
        f"{k}{stats}": (f"${v}$" if stats == "_coef" else "")
        for k, v in flat_naming_dict.items()
        if k in x_var_creator_interaction
        for stats in ["_coef", "_stderr"]
    },
    "con": "Constant",
    "con_stderr": "",
    "obs": "Observation",
    "r2": "Overall $R^2$",
}

reg_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/profit.csv")
reg_tab.dropna(inplace=True)


# Render the Latex Table
def render_latex_table(
    res_dict: dict[str, str],
    y_var: str = "profit",
) -> str:
    """Render the regression results as a LaTeX table."""

    col_len = len(res_dict[list(res_dict.keys())[0]])

    latex_str = "\\begin{tabular}{l" + "c" * col_len + "}\n"
    latex_str += r"\hline" + "\n"
    # dependent variable
    latex_str += (
        r" & "
        + r"\multicolumn{"
        + str(col_len - 1)
        + r"}{c}{"
        + profit_naming_dict[y_var]
        + r"}"
        + r" \\"
    )
    latex_str += (
        " & "
        + " & ".join(["(" + str(_) + ")" for _ in range(1, col_len + 1)])
        + r"\\"
        + "\n"
    )
    latex_str += r"\hline" + "\n"
    # Single variable regression
    for x_var_name, x_var_info in res_dict.items():
        latex_str += f"{profit_naming_dict[x_var_name]}"
        for stats in x_var_info:
            latex_str += f"& {stats} "
        latex_str += "\\\\\n"
    latex_str += r"\hline" + "\n"
    latex_str += "\\end{tabular}\n"

    return latex_str


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


# Run regression
res_dict = {
    **{
        f"{x_var}{stats}": []
        for x_var in x_var_list + x_var_creator_interaction
        for stats in ["_coef", "_stderr"]
    },
    **{
        f"creator_{x_var}{stats}": []
        for x_var in x_var_creator_interaction
        for stats in ["_coef", "_stderr"]
    },
    "con": [],
    "con_stderr": [],
    "obs": [],
    "r2": [],
}

## Creator's profit regression
for x_var in x_var_list + x_var_creator_interaction:

    X = (
        sm.add_constant(pd.DataFrame(reg_tab[x_var]))
        if x_var not in x_var_creator_interaction
        else sm.add_constant(
            pd.DataFrame(
                {
                    "creator": reg_tab["creator"],
                    x_var: reg_tab[x_var],
                    f"creator_{x_var}": reg_tab["creator"] * reg_tab[x_var],
                }
            )
        )
    )
    y = pd.Series(reg_tab[y_var])
    model = sm.OLS(y, X).fit()
    print(model.summary())

    reg_var_non_none_list = []
    if x_var not in x_var_creator_interaction:
        res_dict[f"{x_var}_coef"].append(
            f"{model.params[x_var]:.2f}{asterisk(model.pvalues[x_var])}"
        )
        res_dict[f"{x_var}_stderr"].append(f"({model.bse[x_var]:.2f})")
        reg_var_non_none_list.extend([f"{x_var}_coef", f"{x_var}_stderr"])
    else:
        for x_var in ["creator", x_var, f"creator_{x_var}"]:
            res_dict[f"{x_var}_coef"].append(
                f"{model.params[x_var]:.2f}{asterisk(model.pvalues[x_var])}"
            )
            res_dict[f"{x_var}_stderr"].append(f"({model.bse[x_var]:.2f})")
            reg_var_non_none_list.extend([f"{x_var}_coef", f"{x_var}_stderr"])

    res_dict["con"].append(
        f"{model.params['const']:.2f}{asterisk(model.pvalues['const'])}"
    )
    res_dict["con_stderr"].append(f"({model.bse['const']:.2f})")
    res_dict["obs"].append(f"{model.nobs:.0f}")
    res_dict["r2"].append(f"{model.rsquared:.2f}")
    reg_var_non_none_list.extend(["con", "con_stderr", "obs", "r2"])

    for _ in res_dict.keys():
        if _ not in reg_var_non_none_list:
            res_dict[_].append("")

# Render the LaTeX table
latex_str = render_latex_table(res_dict)

with open(TABLE_PATH / "reg_profit_creator.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)

# Participants' profit regression
res_dict = {
    **{
        f"{x_var}{stats}": []
        for x_var in x_var_creator_interaction
        for stats in ["_coef", "_stderr"]
    },
    "con": [],
    "con_stderr": [],
    "obs": [],
    "r2": [],
}
for x_var in x_var_creator_interaction:

    X = (
        sm.add_constant(pd.DataFrame(reg_tab[x_var]))
        if x_var not in x_var_creator_interaction
        else sm.add_constant(
            pd.DataFrame(
                {
                    x_var: reg_tab[x_var],
                }
            )
        )
    )
    y = pd.Series(reg_tab[y_var])
    model = sm.OLS(y, X).fit()
    print(model.summary())

    reg_var_non_none_list = []
    res_dict[f"{x_var}_coef"].append(
        f"{model.params[x_var]:.2f}{asterisk(model.pvalues[x_var])}"
    )
    res_dict[f"{x_var}_stderr"].append(f"({model.bse[x_var]:.2f})")
    reg_var_non_none_list.extend([f"{x_var}_coef", f"{x_var}_stderr"])

    res_dict["con"].append(
        f"{model.params['const']:.2f}{asterisk(model.pvalues['const'])}"
    )
    res_dict["con_stderr"].append(f"({model.bse['const']:.2f})")
    res_dict["obs"].append(f"{model.nobs:.0f}")
    res_dict["r2"].append(f"{model.rsquared:.2f}")
    reg_var_non_none_list.extend(["con", "con_stderr", "obs", "r2"])

    for _ in res_dict.keys():
        if _ not in reg_var_non_none_list:
            res_dict[_].append("")

# Render the LaTeX table
latex_str = render_latex_table(res_dict)

with open(TABLE_PATH / "reg_participant_profit.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
