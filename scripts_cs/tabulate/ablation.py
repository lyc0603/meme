"""MAS-only ablation table: AUC + Average Return (no res.json dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from environ.constants import PROCESSED_DATA_CS_PATH, TABLE_PATH
from scripts_cs.ml_preprocess import (  # contains token_address, trader_address, label_cls, label
    X_test,
    X_val,
)

# ----------------------------- Config ----------------------------- #
MODEL = "mas_20260116"  # or "mas_zero_shot_20260116"

VAL_PATH = PROCESSED_DATA_CS_PATH / "batch_res_val" / f"{MODEL}.jsonl"
TEST_PATH = PROCESSED_DATA_CS_PATH / "batch_res_test" / f"{MODEL}.jsonl"

OUT_TEX = TABLE_PATH / "mas_ablation_auc_return.tex"

# Weight search
GRID_STEP = 0.05

# Threshold selection for "selected trades" average return
N_THRESH = 201
MIN_VAL_COVERAGE = 0.05
FALLBACK_THRESHOLD = 0.50

# LaTeX formatting
DIGITS_AUC = 4
DIGITS_RET = 4
PCT_DIGITS = 1

INC_MACRO = r"\inc"
DEC_MACRO = r"\dec"


# ---------------------------- Helpers ---------------------------- #
def simplex_grid(step: float = 0.05) -> List[Tuple[float, float, float]]:
    vals = np.arange(0.0, 1.0 + 1e-12, step)
    out: List[Tuple[float, float, float]] = []
    for a in vals:
        for b in vals:
            c = 1.0 - a - b
            if c >= -1e-12:
                out.append((float(a), float(b), float(max(c, 0.0))))
    return out


def load_agent_results(
    path: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Parse batch jsonl and return confidence scores per (token, trader) key
    for coin / wallet / timing.
    """
    coin, wallet, timing = {}, {}, {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            cid = r["custom_id"]
            msg = r["response"]["body"]["choices"][0]
            res = json.loads(msg["message"]["content"])

            # your existing convention: last-2 token logprob for "True/False" confidence
            lp = msg["logprobs"]["content"][-2]["logprob"]
            p = float(np.exp(lp))
            conf = p if res["result"] else 1.0 - p

            if cid.startswith("coin_"):
                coin[cid[5:]] = conf
            elif cid.startswith("wallet_"):
                wallet[cid[7:]] = conf
            elif cid.startswith("timing_"):
                timing[cid[7:]] = conf

    return coin, wallet, timing


def auc_with_weights(
    X, coin, wallet, timing, wc: float, ww: float, wt: float
) -> float | None:
    y, p = [], []

    for _, r in X.iterrows():
        k = f"{r['token_address']}_{r['trader_address']}"
        if k not in coin or k not in wallet or k not in timing:
            continue
        p.append(wc * coin[k] + ww * wallet[k] + wt * timing[k])
        y.append(int(r["label_cls"]))

    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def build_output(
    X, coin, wallet, timing, wc: float, ww: float, wt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns y_cls, ret, proba arrays aligned on samples where all agent scores exist.
    """
    y, ret, p = [], [], []

    for _, r in X.iterrows():
        k = f"{r['token_address']}_{r['trader_address']}"
        if k not in coin or k not in wallet or k not in timing:
            continue
        p.append(wc * coin[k] + ww * wallet[k] + wt * timing[k])
        y.append(int(r["label_cls"]))
        ret.append(float(r["label"]))  # your realized return

    return np.asarray(y), np.asarray(ret), np.asarray(p)


def choose_threshold_max_avg_return(
    proba_val: np.ndarray,
    ret_val: np.ndarray,
    n_thresh: int = 201,
    min_coverage: float = 0.05,
    fallback: float = 0.50,
) -> tuple[float, float, float]:
    thresholds = np.linspace(0.0, 1.0, n_thresh)

    best_t = None
    best_mean = -np.inf
    best_cov = 0.0

    if len(proba_val) == 0:
        return fallback, float("nan"), 0.0

    for t in thresholds:
        mask = proba_val >= t
        cov = float(mask.mean())
        if cov < min_coverage:
            continue

        mean_ret = float(np.nanmean(ret_val[mask])) if mask.any() else -np.inf
        if mean_ret > best_mean:
            best_mean = mean_ret
            best_t = float(t)
            best_cov = cov

    if best_t is None:
        return float(fallback), float("nan"), 0.0

    return best_t, float(best_mean), float(best_cov)


def safe_auc(y: np.ndarray, proba: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def _pct_change(x: float, base: float) -> float:
    if base == 0 or np.isnan(base) or np.isnan(x):
        return float("nan")
    return (x - base) / base * 100.0


def _annot(x: float, base: float) -> str:
    if np.isnan(x) or np.isnan(base) or base == 0:
        return ""
    d = _pct_change(x, base)
    if np.isnan(d):
        return ""
    if abs(d) < 1e-12:
        return rf"{DEC_MACRO}{{0.0\%}}"
    if d < 0:
        return rf"{DEC_MACRO}{{{abs(d):.{PCT_DIGITS}f}\%}}"
    return rf"{INC_MACRO}{{{abs(d):.{PCT_DIGITS}f}\%}}"


def _fmt_num(x: float, digits: int) -> str:
    return "nan" if np.isnan(x) else f"{x:.{digits}f}"


def renorm_weights(wc: float, ww: float, wt: float) -> tuple[float, float, float]:
    s = wc + ww + wt
    if s <= 0:
        # degenerate: return equal weights to avoid crash (shouldn't happen in our variants)
        return (1 / 3, 1 / 3, 1 / 3)
    return (wc / s, ww / s, wt / s)


def evaluate_one(
    name: str,
    y_val: np.ndarray,
    ret_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    ret_test: np.ndarray,
    p_test: np.ndarray,
) -> Dict:
    auc_val = safe_auc(y_val, p_val)
    auc_test = safe_auc(y_test, p_test)

    t_star, val_mean_sel, val_cov = choose_threshold_max_avg_return(
        proba_val=p_val,
        ret_val=ret_val,
        n_thresh=N_THRESH,
        min_coverage=MIN_VAL_COVERAGE,
        fallback=FALLBACK_THRESHOLD,
    )

    mask_test = p_test >= t_star
    test_cov = float(mask_test.mean()) if len(mask_test) else 0.0
    test_mean_sel = (
        float(np.nanmean(ret_test[mask_test])) if mask_test.any() else float("nan")
    )

    return {
        "name": name,
        "val_auc": auc_val,
        "test_auc": auc_test,
        "threshold": float(t_star),
        "val_mean_selected": float(val_mean_sel),
        "val_coverage": float(val_cov),
        "test_mean_selected": float(test_mean_sel),
        "test_coverage": float(test_cov),
    }


def build_latex_table(rows: List[Dict]) -> str:
    baseline = next((r for r in rows if r["name"] == "MAS (Full)"), rows[0])
    base_auc = float(baseline["test_auc"])
    base_ret = float(baseline["test_mean_selected"])

    lines: List[str] = []
    lines.append(r"\begin{tabularx}{\linewidth}{l*2{X}}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Ablation} & \textbf{AUC} & \textbf{Mean Return} \\")
    lines.append(r"    \midrule")

    # Comment baseline row (same style as your ablation tables)
    lines.append(
        rf"    % - & ${_fmt_num(base_auc, DIGITS_AUC)}${DEC_MACRO}{{0.0\%}}"
        rf" & ${_fmt_num(base_ret, DIGITS_RET)}${DEC_MACRO}{{0.0\%}} \\"
    )
    lines.append("")

    for r in rows:
        if r["name"] == baseline["name"]:
            continue

        auc = float(r["test_auc"])
        ret = float(r["test_mean_selected"])

        lines.append(
            rf"    {r['name']} & ${_fmt_num(auc, DIGITS_AUC)}$"
            + _annot(auc, base_auc)
            + rf" & ${_fmt_num(ret, DIGITS_RET)}$"
            + _annot(ret, base_ret)
            + r" \\"
        )
        lines.append("")

    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def main() -> None:
    coin_v, wallet_v, timing_v = load_agent_results(VAL_PATH)
    coin_t, wallet_t, timing_t = load_agent_results(TEST_PATH)

    # -------- learn best weights on VAL AUC (same as your script) -------- #
    best_auc = -np.inf
    best_w = (1 / 3, 1 / 3, 1 / 3)

    for wc, ww, wt in simplex_grid(GRID_STEP):
        auc = auc_with_weights(X_val, coin_v, wallet_v, timing_v, wc, ww, wt)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_w = (wc, ww, wt)

    wc0, ww0, wt0 = best_w
    print(
        f"[Best weights] coin={wc0:.2f}, wallet={ww0:.2f}, timing={wt0:.2f} | VAL AUC={best_auc:.4f}"
    )

    # -------- build FULL and ABLATION outputs (no JSON) -------- #
    def run_variant(name: str, wc: float, ww: float, wt: float) -> Dict:
        y_val, ret_val, p_val = build_output(
            X_val, coin_v, wallet_v, timing_v, wc, ww, wt
        )
        y_test, ret_test, p_test = build_output(
            X_test, coin_t, wallet_t, timing_t, wc, ww, wt
        )
        return evaluate_one(name, y_val, ret_val, p_val, y_test, ret_test, p_test)

    rows: List[Dict] = []
    rows.append(run_variant("MAS (Full)", wc0, ww0, wt0))

    # -Wallet Agent (set ww=0 then renormalize remaining)
    wc, ww, wt = renorm_weights(wc0, 0.0, wt0)
    rows.append(run_variant("w/o Wallet Agent", wc, ww, wt))

    # -Coin Agent (set wc=0 then renormalize remaining)
    wc, ww, wt = renorm_weights(0.0, ww0, wt0)
    rows.append(run_variant("w/o Coin Agent", wc, ww, wt))

    # -Timing Agent
    wc, ww, wt = renorm_weights(wc0, ww0, 0.0)
    rows.append(run_variant("w/o Timing Agent", wc, ww, wt))

    # -------- print diagnostics -------- #
    for r in rows:
        print(
            f"{r['name']}\n"
            f"  AUC: val={r['val_auc']:.4f} | test={r['test_auc']:.4f}\n"
            f"  Threshold*={r['threshold']:.3f} | "
            f"VAL mean={r['val_mean_selected']:.6f}, cov={r['val_coverage']:.2%} | "
            f"TEST mean={r['test_mean_selected']:.6f}, cov={r['test_coverage']:.2%}\n"
        )

    tex = build_latex_table(rows)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX table: {OUT_TEX.resolve()}")


if __name__ == "__main__":
    main()
