"""Script to regress factors against returns and survival."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import statsmodels.api as sm

from environ.constants import FREQ_DICT, NAMING_DICT, PROCESSED_DATA_PATH, TABLE_PATH
from environ.utils import asterisk


def flatten_naming_dict(naming_dict: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Flatten the nested naming dictionary into a single-level dictionary."""
    return {k: v for category in naming_dict.values() for k, v in category.items()}


PROFIT_NAMING_DICT = {
    **flatten_naming_dict(NAMING_DICT),
    "const": "$\\text{Constant}$",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
}

Y_NAMING_DICT = {
    **{f"ret_{k}": v for k, v in FREQ_DICT.items()},
    "survive": "$\\text{Duration}$",
}

X_VAR_LIST = [
    "launch_bundle_transfer",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
]

REG_LIST = [
    ("1 Min", ["launch_bundle_transfer"]),
    (
        "1 Min",
        ["launch_bundle_transfer", "max_same_txn", "pos_to_number_of_swaps_ratio"],
    ),
    ("survive", ["launch_bundle_transfer"]),
    (
        "survive",
        ["launch_bundle_transfer", "max_same_txn", "pos_to_number_of_swaps_ratio"],
    ),
]

mdd_df = pd.read_csv(Path(PROCESSED_DATA_PATH) / "ret_mdd.csv")


def reg_survive(
    df: pd.DataFrame,
    x_var_list: List[str],
) -> List[Dict[str, Any]]:
    """Run regression analysis on return and survival data."""
    results = []

    for y_var, x_var_list in REG_LIST:
        reg_df = (
            df.loc[df[f"death_{y_var}"] == 0, :].copy()
            if y_var != "survive"
            else df.copy()
        )

        X = sm.add_constant(reg_df[x_var_list])
        y = reg_df[f"ret_{y_var}"] if y_var != "survive" else reg_df["survive"]
        model = sm.OLS(y, X).fit()

        results.append(
            (
                y_var,
                {
                    "model": model,
                    "params": model.params,
                    "pvalues": model.pvalues,
                    "bse": model.bse,
                    "r2": model.rsquared,
                    "nobs": int(model.nobs),
                },
            )
        )

    return results


def render_latex_table(results: List[Dict[str, Any]], x_var_list: List[str]) -> str:
    """Render regression results into a LaTeX table."""
    keys = ["ret_1 Min", "ret_1 Min", "survive", "survive"]
    lines = []

    lines.append("\\begin{tabular}{l" + "c" * len(keys) + "}")
    lines.append("\\hline")
    lines.append(" & " + " & ".join([Y_NAMING_DICT[key] for key in keys]) + r" \\")
    lines.append(
        " & " + " & ".join([f"({i})" for i in range(1, len(keys) + 1)]) + r"\\"
    )
    lines.append("\\hline")

    for var in x_var_list + ["const"]:
        row_coef = PROFIT_NAMING_DICT.get(var, var)
        row_stderr = ""
        for key, model_res in results:
            if var in model_res["params"]:
                coef = model_res["params"][var]
                stderr = model_res["bse"][var]
                pval = model_res["pvalues"][var]
                row_coef += f" & {coef:.2f}{asterisk(pval)}"
                row_stderr += f" & ({stderr:.2f})"
            else:
                row_coef += " & "
                row_stderr += " & "
        lines.append(row_coef + r" \\")
        lines.append(row_stderr + r" \\")

    # Add Obs and R²
    obs_row = (
        PROFIT_NAMING_DICT["obs"]
        + " "
        + " ".join(f"& {res[1]['nobs']}" for res in results)
        + r" \\"
    )
    r2_row = (
        PROFIT_NAMING_DICT["r2"]
        + " "
        + " ".join(f"& {res[1]['r2']:.2f}" for res in results)
        + r" \\"
    )

    lines.append(obs_row)
    lines.append(r2_row)
    lines.append("\\hline")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


if __name__ == "__main__":
    results = reg_survive(mdd_df, X_VAR_LIST)
    latex_table = render_latex_table(results, X_VAR_LIST)

    with open(TABLE_PATH / "reg_mask.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
