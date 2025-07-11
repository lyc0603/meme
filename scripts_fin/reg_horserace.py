from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import statsmodels.api as sm

from environ.constants import (
    NAMING_DICT,
    PROCESSED_DATA_PATH,
    TABLE_PATH,
    PFM_NAMING_DICT,
)
from environ.utils import asterisk

PROFIT_NAMING_DICT = {
    **NAMING_DICT,
    **PFM_NAMING_DICT,
    "const": "$\\text{Constant}$",
    "obs": "$\\text{Observations}$",
    "r2": "$R^2$",
}

X_REG = [
    ["volume_bot"],
    ["wash_trading_volume_frac"],
    ["volume_bot", "wash_trading_volume_frac"],
]
X_VAR_LIST = ["volume_bot", "wash_trading_volume_frac"]

pfm = pd.read_csv(Path(PROCESSED_DATA_PATH) / "pfm.csv")


def reg_survive(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    results = {}
    for y_var in PFM_NAMING_DICT:
        for idx, x_var in enumerate(X_REG):
            df_reg = df.dropna(subset=[y_var]).copy()
            X = sm.add_constant(df_reg[x_var])
            y = df_reg[y_var]
            model = sm.OLS(y, X).fit()
            results[f"{y_var}/{idx}"] = {
                "model": model,
                "params": model.params,
                "pvalues": model.pvalues,
                "bse": model.bse,
                "r2": model.rsquared,
                "nobs": int(model.nobs),
            }
    return results


def render_latex_table(
    results: Dict[str, Dict[str, Any]], x_var_list: List[str], chunk_size: int = 7
) -> str:
    """Render regression results into a LaTeX table split into chunks with hlines."""
    keys = list(results.keys())
    chunks = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]
    lines = []

    lines.append("\\begin{tabular}{l" + "c" * chunk_size + "}")
    for chunk_idx, chunk_keys in enumerate(chunks):
        lines.append("\\hline")
        # Header line
        lines.append(
            " & "
            + " & ".join([PFM_NAMING_DICT[k.split("/")[0]] for k in chunk_keys])
            + r" \\"
        )
        lines.append(
            " & "
            + " & ".join(
                [
                    f"({i+1})"
                    for i in range(
                        chunk_idx * chunk_size, chunk_idx * chunk_size + len(chunk_keys)
                    )
                ]
            )
            + r" \\"
        )
        lines.append("\\hline")

        for var in x_var_list + ["const"]:
            row_coef = PROFIT_NAMING_DICT.get(var, var)
            row_stderr = ""
            for key in chunk_keys:
                model_res = results[key]
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
            + "".join(f" & {results[k]['nobs']}" for k in chunk_keys)
            + r" \\"
        )
        r2_row = (
            PROFIT_NAMING_DICT["r2"]
            + "".join(f" & {results[k]['r2']:.2f}" for k in chunk_keys)
            + r" \\"
        )
        lines.append(obs_row)
        lines.append(r2_row)
        lines.append("\\hline")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


if __name__ == "__main__":
    results = reg_survive(pfm)
    latex_table = render_latex_table(results, X_VAR_LIST)

    with open(TABLE_PATH / "reg_horserace.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
