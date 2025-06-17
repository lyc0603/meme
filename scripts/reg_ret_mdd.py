"""Script to regress factors against returns and MDD."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH, NAMING_DICT, FREQ_DICT

mdd_df = pd.read_csv(Path(PROCESSED_DATA_PATH) / "ret_mdd.csv")


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


# Render the Latex Table
def render_latex_table(
    res_dict: dict[str, float],
    tab_name: str,
    x_var_list: list[str],
) -> str:
    """Render the regression results as a LaTeX table."""
    y_ret_var_name = [FREQ_DICT[k]["ret"] for k in FREQ_DICT]
    # y_mdd_var_name = [FREQ_DICT[k]["mdd"] for k in FREQ_DICT]

    latex_str = "\\begin{tabular}{l" + "c" * len(FREQ_DICT) + "}\n"
    latex_str += "\\hline\n"
    for x_var in x_var_list:
        latex_str += " & " + " & ".join(y_ret_var_name) + " \\\\\n"
        latex_str += (
            " & "
            + " & ".join(["(" + str(_) + ")" for _ in range(1, len(FREQ_DICT) + 1)])
            + " \\\\\n"
        )
        latex_str += "\\hline\n"
        latex_str += (
            # Coefficients
            NAMING_DICT[tab_name][x_var]
            + " & "
            + " & ".join(
                [
                    f"{res_dict[x_var][f"ret_{y_var}"]['coef']:.2f}"
                    + asterisk(res_dict[x_var][f"ret_{y_var}"]["coef_pval"])
                    for y_var in FREQ_DICT
                ]
            )
            + " \\\\\n"
            # Coefficients Std Err
            + " & "
            + " & ".join(
                [
                    f"({res_dict[x_var][f"ret_{y_var}"]['coef_stderr']:.2f})"
                    for y_var in FREQ_DICT
                ]
            )
            + " \\\\\n"
            # Constants
            + "Constant & "
            + " & ".join(
                [
                    f"{res_dict[x_var][f"ret_{y_var}"]['con']:.2f}"
                    + asterisk(res_dict[x_var][f"ret_{y_var}"]["con_pval"])
                    for y_var in FREQ_DICT
                ]
            )
            + " \\\\\n"
            # Constants Std Err
            + " & "
            + " & ".join(
                [
                    f"({res_dict[x_var][f"ret_{y_var}"]['con_stderr']:.2f})"
                    for y_var in FREQ_DICT
                ]
            )
            # Observation
            + " \\\\\n"
            + "Observation & "
            + " & ".join(
                [f"{res_dict[x_var][f"ret_{y_var}"]['obs']:.0f}" for y_var in FREQ_DICT]
            )
            + " \\\\\n"
            # R-squared
            + "Overall $R^2$ & "
            + " & ".join(
                [f"{res_dict[x_var][f"ret_{y_var}"]['r2']:.2f}" for y_var in FREQ_DICT]
            )
        )
        latex_str += " \\\\\n"
        latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    return latex_str


# # Preprocess the data

mdd_df.dropna(inplace=True)
# # size
# mdd_df["duration"] = np.log(mdd_df["duration"])
# mdd_df["#trader"] = np.log(mdd_df["#trader"])
# mdd_df["#txn"] = np.log(mdd_df["#txn"])
# mdd_df["#transfer"] = np.log(mdd_df["#transfer"] + 1)

# # bots

# ## Bundle
# mdd_df["holding_herf"] = mdd_df["holding_herf"]
# mdd_df["bundle"] = mdd_df["bundle"]
# mdd_df["transfer_amount"] = mdd_df["transfer_amount"] / 206900000

# ## Volume
# mdd_df["max_same_txn"] = np.log(mdd_df["max_same_txn"] / mdd_df["#txn"])
# mdd_df["pos_to_number_of_swaps_ratio"] = np.log(mdd_df["pos_to_number_of_swaps_ratio"])

# ## Comments
# for _ in [
#     "unique_replies",
#     "unique_repliers",
#     "non_swapper_repliers",
# ]:
#     mdd_df[f"{_}_1"] = mdd_df[_]
#     mdd_df[f"{_}_2"] = mdd_df[_] ** 2

# mdd_df["unique_replies"] = np.log(mdd_df["unique_replies"] + 1)
# mdd_df["reply_interval_herf"] = mdd_df["reply_interval_herf"]
# mdd_df["unique_repliers"] = np.log(mdd_df["unique_repliers"] + 1)
# mdd_df["non_swapper_repliers"] = np.log(mdd_df["non_swapper_repliers"] + 1)

# ## Devs Behavior
# mdd_df["dev_transfer"] = mdd_df["dev_transfer"]
# mdd_df["dev_buy"] = mdd_df["dev_buy"]
# mdd_df["dev_sell"] = mdd_df["dev_sell"]

for var in [
    "bundle_launch",
    "bundle_buy",
    "bundle_sell",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
    "bot_comment_num",
    "positive_bot_comment_num",
    "negative_bot_comment_num",
]:
    mdd_df[var] = mdd_df[var].apply(lambda x: 1 if x > mdd_df[var].median() else 0)

# Dependency variables
for y_var in FREQ_DICT:
    mdd_df[f"ret_{y_var}"] = np.log(mdd_df[f"ret_{y_var}"] + 1)

# Collect regression results
for tab, x_var_info in NAMING_DICT.items():
    res_dict = {}
    for x_var, x_name in x_var_info.items():
        res_dict[x_var] = {}
        for y_var in FREQ_DICT:
            reg_df = mdd_df.loc[mdd_df[f"death_{y_var}"] == 0, :].copy()
            X = sm.add_constant(reg_df[x_var])
            y = reg_df[f"ret_{y_var}"]
            model = sm.OLS(y, X).fit()
            pval = model.pvalues[x_var]
            res_dict[x_var][f"ret_{y_var}"] = {
                "coef": model.params[x_var],
                "coef_stderr": model.bse[x_var],
                "coef_pval": model.pvalues[x_var],
                "con": model.params["const"],
                "con_stderr": model.bse["const"],
                "con_pval": model.pvalues["const"],
                "obs": model.nobs,
                "r2": model.rsquared,
            }

    # Render the LaTeX table for each tab
    latex_str = render_latex_table(res_dict, tab, list(x_var_info.keys()))
    # Save the LaTeX table to a file
    with open(Path(TABLE_PATH) / f"reg_{tab}.tex", "w") as f:
        f.write(latex_str)
