"""Script to process the results of the wallet agent."""

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
)

from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH


merged_df = pd.read_csv(PROCESSED_DATA_PATH / "tab" / "wallet.csv")

y_pred = merged_df["prediction"].astype(bool)
y_true = merged_df["label"].astype(bool)

# get confusion matrix with raw counts
cm = confusion_matrix(y_true, y_pred)

# assign cells
tp = cm[1, 1]  # predicted TRUE, actual TRUE
fn = cm[1, 0]  # predicted FALSE, actual TRUE
fp = cm[0, 1]  # predicted TRUE, actual FALSE
tn = cm[0, 0]  # predicted FALSE, actual FALSE

# compute totals
actual_true_total = tp + fn
actual_false_total = fp + tn
predicted_true_total = tp + fp
predicted_false_total = fn + tn
grand_total = cm.sum()

# generate LaTeX
latex = f"""
\\begin{{tabular}}{{l|c|c|c}}
\\hline
           & \\multicolumn{{2}}{{c|}}{{\\textbf{{Predicted}}}} & \\\\
\\cline{{2-3}}
\\textbf{{Actual}} & TRUE & FALSE & \\textbf{{Total}} \\\\
\\hline
TRUE   & {tp} & {fn} & {actual_true_total} \\\\
FALSE  & {fp} & {tn} & {actual_false_total} \\\\
\\hline
\\textbf{{Total}} & {predicted_true_total} & {predicted_false_total} & {grand_total} \\\\
\\hline
\\end{{tabular}}
"""

with open(TABLE_PATH / "wallet.tex", "w") as f:
    f.write(latex)
