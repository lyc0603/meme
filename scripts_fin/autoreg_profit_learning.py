"""This script performs autoregressive analysis on trader profits, categorizing traders into winners, neutrals, and losers based on their performance metrics."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


CRITICAL_VAL = 2.576

pj_pft = pd.read_csv(PROCESSED_DATA_PATH / "trader_project_profits.csv")
pj_pft.drop_duplicates(subset=["trader_address", "meme"], keep="last", inplace=True)
pj_pft["date"] = pd.to_datetime(pj_pft["date"])
pj_pft = (
    pj_pft.loc[pj_pft["meme_num"] < 1000]
    .sort_values(["trader_address", "date"])
    .reset_index(drop=True)
)

winner = pj_pft[(pj_pft["meme_num"] < 1000) & (pj_pft["t_stat"] > CRITICAL_VAL)].copy()
neutral = pj_pft[
    (pj_pft["meme_num"] < 1000) & (pj_pft["t_stat"].abs() <= CRITICAL_VAL)
].copy()
loser = pj_pft[(pj_pft["meme_num"] < 1000) & (pj_pft["t_stat"] < -CRITICAL_VAL)].copy()


def get_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    """Generate lag features for returns in the group."""
    group = group.sort_values("date").copy()
    group["idx"] = range(len(group))

    def safe_mean(start, end):
        if start < 0:
            return np.nan
        return group["ret"].iloc[start:end].mean()

    n = len(group)
    group["ret_lag_15_plus"] = [
        safe_mean(max(0, i - 1000), i - 15) if i >= 15 else np.nan for i in range(n)
    ]

    group["ret_lag_10_15"] = [
        safe_mean(i - 15, i - 10) if i >= 15 else np.nan for i in range(n)
    ]

    group["ret_lag_5_10"] = [
        safe_mean(i - 10, i - 5) if i >= 10 else np.nan for i in range(n)
    ]

    group["ret_lag_1_5"] = [safe_mean(i - 5, i) if i >= 5 else np.nan for i in range(n)]

    return group


def parallel_apply(grouped_df):
    """Apply get_lag_features in parallel to each group."""
    groups = [group for _, group in grouped_df]
    with Pool(cpu_count() - 5) as pool:
        results = list(
            tqdm(pool.imap_unordered(get_lag_features, groups), total=len(groups))
        )
    return pd.concat(results, ignore_index=True)


# --- Run regression by group ---
def run_exit_regression(sub_df):
    """Run OLS regression for exit probability based on lagged returns."""
    lag_cols = [col for col in sub_df.columns if col.startswith("ret_lag_")]
    sub_df = sub_df.dropna(subset=lag_cols + ["ret"])
    X = sm.add_constant(sub_df[lag_cols])
    y = sub_df["ret"]
    return sm.OLS(y, X).fit()


df = parallel_apply(pj_pft.groupby("trader_address"))

groups = {
    "All": df,
    "Winner": df[df["trader_address"].isin(winner["trader_address"])],
    "Neutral": df[df["trader_address"].isin(neutral["trader_address"])],
    "Loser": df[df["trader_address"].isin(loser["trader_address"])],
}

models = {name: run_exit_regression(data) for name, data in groups.items()}


def render_multicolumn_latex_table(models, y_name="$\text{r}_{it}$", var_rename=None):
    """Render regression results as a LaTeX table with multiple columns."""

    def add_stars(val, pval):
        if pval < 0.01:
            return f"{val:.4f}***"
        elif pval < 0.05:
            return f"{val:.4f}**"
        elif pval < 0.1:
            return f"{val:.4f}*"
        else:
            return f"{val:.4f}"

    variables = list(models["All"].params.index)
    if "const" in variables:
        variables.remove("const")
        variables.append("const")

    tab = []
    tab.append("\\begin{tabular}{lcccc}")
    tab.append("\\toprule")
    tab.append(" & \\multicolumn{4}{c}{$\\text{r}_{i,t}$} \\\\")
    tab.append("\\cmidrule{2-5}")
    tab.append(" & All & Winner & Neutral & Loser\\\\")
    tab.append(" & (1) & (2) & (3) & (4) \\\\")
    tab.append("\\midrule")

    for var in variables:
        var_label = var_rename[var] if var_rename and var in var_rename else var
        row_coef = [var_label]
        row_stderr = [""]

        for model in models.values():
            if var in model.params:
                coef = model.params[var]
                stderr = model.bse[var]
                pval = model.pvalues[var]
                row_coef.append(add_stars(coef, pval))
                row_stderr.append(f"({stderr:.4f})")
            else:
                row_coef.append("")
                row_stderr.append("")

        tab.append(" & ".join(row_coef) + " \\\\")
        tab.append(" & ".join(row_stderr) + " \\\\")

    tab.append("\\midrule")
    tab.append(
        "Observations"
        + " & "
        + " & ".join(f"{int(m.nobs):,}" for m in models.values())
        + " \\\\"
    )
    tab.append(
        "$R^2$"
        + " & "
        + " & ".join(f"{m.rsquared:.4f}" for m in models.values())
        + " \\\\"
    )
    tab.append("\\bottomrule")
    tab.append("\\end{tabular}")
    return "\n".join(tab)


latex_output = render_multicolumn_latex_table(
    models,
    y_name="$\text{r}_{it}$",
    var_rename={
        "ret_lag_15_plus": r"$\overline{\text{r}}_{i,t-15,0}$",
        "ret_lag_10_15": r"$\overline{\text{r}}_{i,t-15,t-10}$",
        "ret_lag_5_10": r"$\overline{\text{r}}_{i,t-10,t-5}$",
        "ret_lag_1_5": r"$\overline{\text{r}}_{i,t-5,t-1}$",
        "const": r"Constant",
    },
)

with open(TABLE_PATH / "reg_learning.tex", "w", encoding="utf-8") as f:
    f.write(latex_output)
