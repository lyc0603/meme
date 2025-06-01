"""Script to regress factors against returns and MDD."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from environ.constants import TABLE_PATH, PROCESSED_DATA_PATH

mdd_df = pd.read_csv(Path(PROCESSED_DATA_PATH) / "ret_mdd.csv")

FREQ_DICT = {
    "1 Min": {"ret": "${\it Ret}_{\it 1min}$", "mdd": "${\it MDD}_{\it 1min}$"},
    "5 Mins": {"ret": "${\it Ret}_{\it 5min}$", "mdd": "${\it MDD}_{\it 5min}$"},
    "10 Mins": {"ret": "${\it Ret}_{\it 10min}$", "mdd": "${\it MDD}_{\it 10min}$"},
    "15 Mins": {"ret": "${\it Ret}_{\it 15min}$", "mdd": "${\it MDD}_{\it 15min}$"},
    "30 Mins": {"ret": "${\it Ret}_{\it 30min}$", "mdd": "${\it MDD}_{\it 30min}$"},
    "1 Hour": {"ret": "${\it Ret}_{\it 1h}$", "mdd": "${\it MDD}_{\it 1h}$"},
    "6 Hours": {"ret": "${\it Ret}_{\it 6h}$", "mdd": "${\it MDD}_{\it 6h}$"},
    "12 Hours": {"ret": "${\it Ret}_{\it 12h}$", "mdd": "${\it MDD}_{\it 12h}$"},
}

NAMING_DICT = {
    "size": {
        "duration": "$\mathrm{Duration}$",
        "#trader": "$\mathrm{\#Traders}$",
        "#txn": "$\mathrm{\#Txns}$",
        "#transfer": "$\mathrm{\#Transfers}$",
    },
    "bundle_bot": {
        "holding_herf": "$\mathrm{HHI_\mathrm{Holding}}$",
        "bundle": "$\mathrm{HHI_\mathrm{Bundle}}$",
    },
    "volume_bot": {
        "transfer_amount": "$\mathrm{Transfer Amount (\%)}$",
        "max_same_txn": "$\mathrm{Max Same Txn (\%)}$",
        "pos_to_number_of_swaps_ratio": "$\mathrm{Position\#Swaps}$",
    },
    "comment_bot": {
        "unique_replies": "$\mathrm{\#Replies}$",
        "reply_interval_herf": "$\mathrm{HHI_\mathrm{Interval}}$",
        "unique_repliers": "$\mathrm{\#Repliers}$",
        "non_swapper_repliers": "$\mathrm{\#Non-Trader Repliers}$",
    },
    "dev": {
        "dev_transfer": "$\mathrm{Dev Transfer}$",
        "dev_buy": "$\mathrm{Dev Buy}$",
        "dev_sell": "$\mathrm{Dev Sell}$",
    },
}


def asterisk(pval: float) -> str:
    """Return asterisks based on p-value."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""


# Render the Latex Table
def render_latex_table(
    res_dict: dict[str, float], tab_name: str, x_var_list: list[str]
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
                    f"{res_dict[x_var][y_var]['coef']:.2f}"
                    + asterisk(res_dict[x_var][y_var]["coef_pval"])
                    for y_var in FREQ_DICT
                ]
            )
            + " \\\\\n"
            # Coefficients Std Err
            + " & "
            + " & ".join(
                [
                    f"({res_dict[x_var][y_var]['coef_stderr']:.2f})"
                    for y_var in FREQ_DICT
                ]
            )
            + " \\\\\n"
            # Constants
            + "Constant & "
            + " & ".join(
                [
                    f"{res_dict[x_var][y_var]['con']:.2f}"
                    + asterisk(res_dict[x_var][y_var]["con_pval"])
                    for y_var in FREQ_DICT
                ]
            )
            + " \\\\\n"
            # Constants Std Err
            + " & "
            + " & ".join(
                [f"({res_dict[x_var][y_var]['con_stderr']:.2f})" for y_var in FREQ_DICT]
            )
            # Observation
            + " \\\\\n"
            + "Observation & "
            + " & ".join(
                [f"{res_dict[x_var][y_var]['obs']:.0f}" for y_var in FREQ_DICT]
            )
            + " \\\\\n"
            # R-squared
            + "Overall $R^2$ & "
            + " & ".join([f"{res_dict[x_var][y_var]['r2']:.2f}" for y_var in FREQ_DICT])
        )
        latex_str += " \\\\\n"
        latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    return latex_str


# Preprocess the data
# size
mdd_df["duration"] = np.log(mdd_df["duration"])
mdd_df["#trader"] = np.log(mdd_df["#trader"])
mdd_df["#txn"] = np.log(mdd_df["#txn"] + 1)
mdd_df["#transfer"] = np.log(mdd_df["#transfer"] + 1)

# bots
## Bundle
mdd_df["holding_herf"] = mdd_df["holding_herf"]
mdd_df["bundle"] = np.log(mdd_df["bundle"] + 1)
## Volume
mdd_df["transfer_amount"] = mdd_df["transfer_amount"] / 206900000
mdd_df["max_same_txn"] = np.log(mdd_df["max_same_txn"] / mdd_df["#txn"])
mdd_df["pos_to_number_of_swaps_ratio"] = np.log(mdd_df["pos_to_number_of_swaps_ratio"])
## Comments
mdd_df["unique_replies"] = np.log(mdd_df["unique_replies"] + 1)
mdd_df["reply_interval_herf"] = mdd_df["reply_interval_herf"]
mdd_df["unique_repliers"] = np.log(mdd_df["unique_repliers"] + 1)
mdd_df["non_swapper_repliers"] = np.log(mdd_df["non_swapper_repliers"] + 1)

## Devs Behavior
mdd_df["dev_transfer"] = mdd_df["dev_transfer"]
mdd_df["dev_buy"] = mdd_df["dev_buy"]
mdd_df["dev_sell"] = mdd_df["dev_sell"]


# Dependency variables
for k in FREQ_DICT:
    mdd_df[k] = np.log(mdd_df[k] + 1)

# Collect regression results
for tab, x_var_info in NAMING_DICT.items():
    res_dict = {}
    for x_var, x_name in x_var_info.items():
        res_dict[x_var] = {}
        for y_var in FREQ_DICT:
            X = sm.add_constant(mdd_df[x_var])
            y = mdd_df[y_var]
            model = sm.OLS(y, X).fit()
            pval = model.pvalues[x_var]
            res_dict[x_var][y_var] = {
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
