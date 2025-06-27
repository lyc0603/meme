"""Script to tabulate class information for meme coins."""

import json
from pathlib import Path

from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH


with open(
    Path(PROCESSED_DATA_PATH) / "tab" / "res_coin.json", "r", encoding="utf-8"
) as f:
    res_dict = json.load(f)


metrics = ["accuracy", "precision", "recall", "f1_score"]
freqs = res_dict["freq"]

# find maximum across all metric values for scaling
max_value = max(max(res_dict[metric]) for metric in metrics)
max_value = round(max_value, 4)

with open(f"{TABLE_PATH}/coin.tex", "w", encoding="utf-8") as f:
    f.write(r"\renewcommand{\maxnum}{" + str(max_value) + r"}" + "\n")
    f.write(r"\begin{tabularx}{\linewidth}{l*" + str(len(freqs)) + "{X}}" + "\n")
    f.write(r"\toprule" + "\n")
    # header row
    header = [""] + freqs
    f.write(" & ".join([f"\\textbf{{{h}}}" for h in header]) + r"\\" + "\n")
    f.write(r"\midrule" + "\n")

    for metric in metrics:
        rowname = metric.replace("_", " ").title()
        row = [f"\\textbf{{{rowname}}}"]
        for val in res_dict[metric]:
            cell = (
                r"\multicolumn{1}{|@{}l@{}|}{"
                + "\databar{{{:.4f}}}".format(round(val, 4))
                + "}"
            )
            row.append(cell)
        f.write(" & ".join(row) + r"\\" + "\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabularx}" + "\n")
