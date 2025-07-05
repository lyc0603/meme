import json
from pathlib import Path

from environ.constants import PROCESSED_DATA_PATH, TABLE_PATH

# Load results from JSON
with open(Path(PROCESSED_DATA_PATH) / "tab" / "res.json", "r", encoding="utf-8") as f:
    res_dict = json.load(f)

NAMING_DICT = {
    "transaction": "Transaction",
    "technical": "Technical",
    "comment": "Comment",
    "coin": "All",
}

metrics = ["accuracy", "precision", "recall", "f1_score"]
agents = ["transaction", "technical", "comment", "coin"]
freqs = res_dict[agents[0]]["freq"]

# find max value for data bar scaling
max_value = max(max(res_dict[agent][metric]) for agent in agents for metric in metrics)
max_value = round(max_value, 4)

with open(Path(TABLE_PATH) / "coin.tex", "w", encoding="utf-8") as f:
    f.write(r"\renewcommand{\maxnum}{" + str(max_value) + r"}" + "\n")
    f.write(r"\begin{tabularx}{\linewidth}{l|l*4{X}}" + "\n")
    f.write(r"\toprule" + "\n")

    header = [
        r"\textbf{Agent}",
        r"\textbf{Freq}",
    ] + [r"\textbf{" + m.title().replace("_", " ") + "}" for m in metrics]
    f.write(" & ".join(header) + r"\\" + "\n")
    f.write(r"\midrule" + "\n")

    # map each metric to its color databar macro
    metric_macro_map = {
        "accuracy": r"\databaracc",
        "precision": r"\databarprec",
        "recall": r"\databarrec",
        "f1_score": r"\databarfone",
    }

    for agent in agents:
        # one multirow per agent
        agent_multirow = (
            r"\multicolumn{1}{c|}{\multirow{"
            + str(len(freqs))
            + r"}{*}{\rotatebox[origin=c]{90}{\textbf{"
            + NAMING_DICT[agent]
            + r"}}}}"
        )

        for freq_idx, freq in enumerate(freqs):
            if freq_idx == 0:
                agent_cell = agent_multirow
            else:
                agent_cell = ""
            freq_cell = r"\textbf{" + str(freq) + "}"
            metric_cells = []
            for metric in metrics:
                val = res_dict[agent][metric][freq_idx]
                databar_macro = metric_macro_map[metric]
                cell = f"{databar_macro}{{{val:.4f}}}"
                metric_cells.append(cell)
            row = [agent_cell, freq_cell] + metric_cells
            f.write(" & ".join(row) + r"\\" + "\n")

        if agent != agents[-1]:
            f.write(r"\midrule" + "\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabularx}" + "\n")
