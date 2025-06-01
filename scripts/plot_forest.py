"""Plotting a forest plot for regression coefficients of single factor models"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH

# Plot
plt.figure(figsize=(7, 3.5))

# Load and preprocess data
processed_path = Path(PROCESSED_DATA_PATH)
mdd_path = processed_path / "ret.csv"
mdd_df = pd.read_csv(mdd_path)

mdd_df["duration"] = np.log(mdd_df["duration"])
mdd_df["unique_address"] = np.log(mdd_df["unique_address"])
mdd_df["unique_transfer"] = np.log(mdd_df["unique_transfer"] + 1)
# mdd_df["transfer_amount"] = mdd_df["transfer_amount"] / 206900000
# mdd_df["dev_transfer_amount"] = mdd_df["dev_transfer_amount"] / (
#     1_000_000_000 - 206_900_000
# )
# mdd_df["max_same_txn"] = np.log(mdd_df["max_same_txn"] / mdd_df["total_txn"])
# mdd_df["pos_to_number_of_swaps_ratio"] = np.log(mdd_df["pos_to_number_of_swaps_ratio"])
# mdd_df["Unique Repliers"] = np.log(mdd_df["Unique Repliers"] + 1)
# mdd_df["Unique Replies"] = np.log(mdd_df["Unique Replies"] + 1)
# mdd_df["Non-Swapper Repliers"] = np.log(mdd_df["Unique Repliers"] + 1)

FREQ_DICT = {
    "1 Min": {"freq": "1min", "before": 1},
    "5 Mins": {"freq": "1min", "before": 5},
    "10 Mins": {"freq": "1min", "before": 10},
    "15 Mins": {"freq": "1min", "before": 15},
    "30 Mins": {"freq": "1min", "before": 30},
    "1 Hour": {"freq": "1h", "before": 1},
    "6 Hours": {"freq": "1h", "before": 6},
    "12 Hours": {"freq": "1h", "before": 12},
}

NAMING_DICT = {
    "unique_address": "# Swapper",
    "duration": "Duration",
    "unique_transfer": "# Transfer",
    "holding_herf": "Holding HERF",
    # "out_degree_herf": "Out Degree HERF",
    # "in_degree_herf": "In Degree HERF",
    # "degree": "Degree",
    # "transfer_amount": "Transfer Amount",
    # "dev_transfer": "Transfer Dev",
    # "dev_txn": "Swap Dev",
    "dev_buy": "Buy Dev",
    "dev_sell": "Sell Dev",
    # "dev_transfer_amount": "Transfer Amount Dev",
    # "bundle": "Bundle HERF",
    # "txn_per_s": "Txn per Second",
    # "max_same_txn": "Max Same Txn",
    # "pos_to_number_of_swaps_ratio": "pos_to_number_of_swaps_ratio",
    # "Unique Replies": "Unique Replies",
    # "Reply Interval Herfindahl": "Reply Interval Herfindahl",
    # "Unique Repliers": "Unique Repliers",
    # "Non-Swapper Repliers": "Non-Swapper Repliers",
}

# plot the heatmap of the correlation matrix coefficients


IND_VARS = NAMING_DICT.keys()

for k in FREQ_DICT:
    mdd_df[k] = np.log(mdd_df[k] + 1)

# Collect regression results
results = []
for x_var in IND_VARS:
    for y_var in FREQ_DICT:
        X = sm.add_constant(mdd_df[x_var])
        y = mdd_df[y_var]
        model = sm.OLS(y, X).fit()
        coef = model.params[x_var]
        ci_low, ci_up = model.conf_int().loc[x_var]
        pval = model.pvalues[x_var]
        results.append(
            {
                "x_var": x_var,
                "y_var": y_var,
                "coef": coef,
                "ci_low": ci_low,
                "ci_up": ci_up,
                "pval": pval,
            }
        )

df = pd.DataFrame(results)

# Setup
x_labels = list(FREQ_DICT.keys())[::-1]
y_labels = IND_VARS
palette = sns.color_palette("tab10", len(x_labels))

# Mapping
y_pos_map = {var: i * 0.5 for i, var in enumerate(reversed(y_labels))}

for i, time_label in enumerate(x_labels):
    color = palette[i]
    sub = df[df["y_var"] == time_label]
    for _, row in sub.iterrows():
        y_base = y_pos_map[row["x_var"]]
        y_offset = (i - len(x_labels) / 2) * 0.05  # symmetric dodge
        y = y_base + y_offset
        plt.errorbar(
            row["coef"],
            y,
            xerr=[[row["coef"] - row["ci_low"]], [row["ci_up"] - row["coef"]]],
            fmt="o",
            color=color,
            capsize=3,
            label=time_label if _ == sub.index[0] else "",
        )

# Add alternating grey-white backgrounds for subclasses (x_vars)
for i, (x_var, y_val) in enumerate(y_pos_map.items()):
    color = "#f0f0f0" if i % 2 == 0 else "white"
    plt.axhspan(y_val - 0.25, y_val + 0.25, color=color, zorder=0)

plt.axvline(0, linestyle="--", color="black", linewidth=1)
plt.yticks(list(y_pos_map.values()), reversed([NAMING_DICT[var] for var in y_labels]))
plt.xlabel("Coefficient of Single Factor Model")
# Sort legend labels chronologically
handles, labels = plt.gca().get_legend_handles_labels()
sorted_pairs = sorted(
    zip(labels, handles), key=lambda x: list(FREQ_DICT.keys()).index(x[0])
)
sorted_labels, sorted_handles = zip(*sorted_pairs)
plt.legend(
    sorted_handles,
    sorted_labels,
    title="Time Interval",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=False,
)
plt.tight_layout()
plt.savefig(
    FIGURE_PATH / "forest_plot_ret.pdf",
    dpi=300,
)
plt.show()
