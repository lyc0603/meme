"""Script for regression variable summary statistics"""

import pandas as pd
from environ.constants import TABLE_PATH, PROCESSED_DATA_PATH, NAMING_DICT

X_VAR = [
    "launch_bundle_transfer",
    "bundle_creator_buy",
    "bundle_launch",
    "bundle_buy",
    "bundle_sell",
    "max_same_txn",
    "pos_to_number_of_swaps_ratio",
    "positive_bot_comment_num",
    "negative_bot_comment_num",
]

project_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/ret_mdd.csv")

flat_naming_dict = {
    sub_key: sub_value
    for category in NAMING_DICT.values()
    for sub_key, sub_value in category.items()
}

# calculate descriptive stats
summary = {}
for var in X_VAR:
    values = project_tab[var].dropna()
    summary[var] = {
        "num_obs": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "p10": values.quantile(0.1),
        "median": values.median(),
        "p90": values.quantile(0.9),
    }

# generate LaTeX
latex_str = (
    "\\begin{tabular}{lcccccc}\n"
    "\\hline\n"
    "Variable & Num. Obs. & Mean & Std. Dev. & 10th & Median & 90th \\\\\n"
    "\\hline\n"
)

for var in X_VAR:
    stats = summary[var]
    latex_str += (
        f"{flat_naming_dict[var]} & "
        f"{stats['num_obs']} & "
        f"{stats['mean']:.3f} & "
        f"{stats['std']:.3f} & "
        f"{stats['p10']:.3f} & "
        f"{stats['median']:.3f} & "
        f"{stats['p90']:.3f} \\\\\n"
    )

latex_str += "\\hline\n\\end{tabular}\n"

with open(TABLE_PATH / "reg_var_summary.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
