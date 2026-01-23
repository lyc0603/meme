"""
Generate a LaTeX feature-importance table with databar (per-column max)

- Models: LASSO, NN, XGBoost (NO RF)
- Remove the original Feature column; rename Notation -> Feature
- Emits \databarLASSO{<number>}, \databarNN{<number>}, \databarXGB{<number>}
  so each column can have its own max value in LaTeX.

NOTE:
- This script outputs PROPORTIONS in [0,1] by default (AS_PERCENT=False).
- In LaTeX, define three maxima, e.g.
    \newcommand{\maxLASSO}{...}
    \newcommand{\maxNN}{...}
    \newcommand{\maxXGB}{...}
  and corresponding \databarLASSO / \databarNN / \databarXGB macros.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from environ.constants import PROCESSED_DATA_CS_PATH, TABLE_PATH

# Configuration
INPUT_JSON = PROCESSED_DATA_CS_PATH / "ml_res.json"
OUTPUT_TEX = TABLE_PATH / "feature_importance.tex"

TOP_K: int | None = None  # None = include all features
DIGITS = 4  # databar looks nicer with 3-4 decimals
AS_PERCENT = False  # keep False for databar macro compatibility

# JSON keys in your file
MODELS = ["Lasso", "NN", "XGBoost"]

# Mapping from model name -> LaTeX databar macro name
DATABAR_MACRO = {
    "Lasso": "databarLASSO",
    "NN": "databarNN",
    "XGBoost": "databarXGB",
}

FEATURE_MAP: List[Tuple[str, str, str]] = [
    ("average_ret", "Return (all)", r"$\mathbf{\bar{R}_{\text{all}}}$"),
    ("last_ret", "Return (1th)", r"$\mathbf{\bar{R}_{\text{1th}}}$"),
    ("five_to_one_ret", "Return (1th--5th)", r"$\mathbf{\bar{R}_{\text{1th-5th}}}$"),
    ("ten_to_six_ret", "Return (6th--10th)", r"$\mathbf{\bar{R}_{\text{6th-10th}}}$"),
    (
        "fifteen_to_eleven_ret",
        "Return (11th--15th)",
        r"$\mathbf{\bar{R}_{\text{11th-15th}}}$",
    ),
    ("num_trades", "Number of Trades", r"$\#Trade$"),
    ("std_ret", "Return Standard Deviation", r"$std$"),
    ("t_stat", r"$t$-stat", r"$t$"),
    ("time_since_last_trade", "Time Since Last Trade", r"$T_{\mathrm{Last}}$"),
    ("time_since_first_trade", "Time Since First Trade", r"$T_{\mathrm{First}}$"),
    ("time_since_launch", "Time Since Launch", r"$T_{\mathrm{Launch}}$"),
    ("first_txn_price", "Trader Purchase Price", r"$P$"),
    ("first_txn_amount", "Trader Purchase Amount", r"$A$"),
    ("first_txn_quantity", "Trader Purchase Quantity", r"$Q$"),
    ("launch_bundle", "Bundle Bot", r"$\text{Bundle Bot}$"),
    ("sniper_bot", "Sniper Bot", r"$\text{Sniper Bot}$"),
    ("wash_trading_bot", "Bump Bot", r"$\text{Bump Bot}$"),
    ("comment_bot", "Comment Bot", r"$\text{Comment Bot}$"),
]


def read_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_importance(res: dict, model: str) -> Dict[str, float]:
    return {
        k: float(v) for k, v in res[model]["feature_importance"]["importance"].items()
    }


def normalize(imp: Dict[str, float]) -> Dict[str, float]:
    vals = {k: abs(v) for k, v in imp.items()}
    s = sum(vals.values())
    if s <= 0:
        return {k: 0.0 for k in vals}
    return {k: v / s for k, v in vals.items()}


def fmt_number(x: float) -> str:
    if AS_PERCENT:
        return f"{100 * x:.{DIGITS}f}"
    return f"{x:.{DIGITS}f}"


def rank_features(norm_imp: Dict[str, Dict[str, float]], feats: List[str]) -> List[str]:
    avg = {f: np.mean([norm_imp[m].get(f, 0.0) for m in MODELS]) for f in feats}
    return sorted(feats, key=lambda f: avg[f], reverse=True)


def databar_cell(value: float, model: str) -> str:
    """
    Emit per-model macro so each column can have its own max value in LaTeX:
      \databarLASSO{...}, \databarNN{...}, \databarXGB{...}
    """
    num = fmt_number(value)
    macro = DATABAR_MACRO[model]
    return rf"\multicolumn{{1}}{{|@{{}}l@{{}}|}}{{\{macro}{{{num}}}}}"


def build_latex(norm_imp: Dict[str, Dict[str, float]]) -> str:
    feats = [f for f, _, _ in FEATURE_MAP]
    ordered = rank_features(norm_imp, feats)
    if TOP_K is not None:
        ordered = ordered[:TOP_K]

    meta = {f: (name, sym) for f, name, sym in FEATURE_MAP}

    lines: List[str] = []
    lines.append(r"\begin{tabularx}{0.9\linewidth}{l*{3}{X}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{\makecell{Feature}}} & \multicolumn{3}{c}{\textbf{Normalized Importance}} \\"
    )
    lines.append(r"\cmidrule(lr){2-4}")
    lines.append(r" & \makecell{LASSO} & \makecell{NN} & \makecell{XGBoost} \\")
    lines.append(r"\midrule")

    for f in ordered:
        name, _ = meta[f]

        cell_lasso = databar_cell(norm_imp["Lasso"].get(f, 0.0), "Lasso")
        cell_nn = databar_cell(norm_imp["NN"].get(f, 0.0), "NN")
        cell_xgb = databar_cell(norm_imp["XGBoost"].get(f, 0.0), "XGBoost")

        lines.append(f"{name} & {cell_lasso} & {cell_nn} & {cell_xgb} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def main() -> None:
    res = read_results(INPUT_JSON)
    norm_imp = {m: normalize(get_importance(res, m)) for m in MODELS}
    latex = build_latex(norm_imp)

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)


if __name__ == "__main__":
    main()
