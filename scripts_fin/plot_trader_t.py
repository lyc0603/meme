"""Script to plot the distribution of t-statistics at 0.01 significance level."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH

# Constants
CRITICAL_VAL = 2.576
FONT_SIZE = 18

# Load and clean the data
df = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
data = df.loc[df["meme_num"] < 1000, "t_stat"].dropna()


# Trim outliers (1st to 99th percentile)
lower, upper = np.percentile(data, [1, 99])
trimmed_data = data[(data >= lower) & (data <= upper)]

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(trimmed_data, bins=100, kde=True, ax=ax, color="gray")

# Vertical lines at ±2.576 (two-tailed 0.01 significance)
ax.axvline(
    CRITICAL_VAL,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Statistical Significance Level at 1%",
)
ax.axvline(-CRITICAL_VAL, color="red", linestyle="--", linewidth=2)

# Labels and tick formatting
ax.set_xlabel("t-statistic", fontsize=FONT_SIZE)
ax.set_ylabel("Frequency", fontsize=FONT_SIZE)
ax.tick_params(axis="both", labelsize=FONT_SIZE)

# Offset font size
ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

# Legend (single entry for both lines)
legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=1,
    frameon=True,
    fontsize=FONT_SIZE,
)
legend.get_frame().set_edgecolor("black")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(FIGURE_PATH / "trader_t_stats.pdf", bbox_inches="tight")
plt.show()
