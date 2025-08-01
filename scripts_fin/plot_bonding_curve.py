"""Script to plot bonding curve components for a token sale on Solana."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from environ.constants import FIGURE_PATH

FONT_SIZE = 18

# bonding curve constants
initial_sol = 30
initial_tokens = 1_073_000_191
k = initial_sol * initial_tokens


def token_amount(x):
    """Calculate the total amount of tokens received based on the amount of SOL deposited."""
    return initial_tokens - (k / (initial_sol + x))


def price(x):
    """Calculate the price of the token based on the amount of SOL deposited."""
    return (initial_sol + x) ** 2 / k


# range of SOL
x = np.linspace(0, 85, 10)
y_tokens = token_amount(x)
z_price = price(x)
initial_price = price(0)
migration_price = 85

### Figure 1: Total Tokens Received ###
fig1, ax1 = plt.subplots(figsize=(10, 7))

ax1.plot(
    x,
    y_tokens,
    color="red",
    marker="o",
    markersize=4,
    linewidth=2,
    label="Total Meme Coin Received",
    markeredgewidth=5,
)

ax1.set_xlabel("Total SOL Deposited", fontsize=FONT_SIZE)
ax1.set_ylabel("Total Meme Coin Received", fontsize=FONT_SIZE)
ax1.tick_params(axis="both", labelsize=FONT_SIZE)

formatter1 = ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((0, 0))
ax1.yaxis.set_major_formatter(formatter1)
ax1.yaxis.get_offset_text().set_fontsize(FONT_SIZE)


ax1.axvline(
    x=migration_price, color="red", linestyle="--", linewidth=2, label="Migration"
)
ax1.set_xticks(np.sort(np.append(ax1.get_xticks(), migration_price))[1:-1])

legend = ax1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=True,
    fontsize=FONT_SIZE,
)
legend.get_frame().set_edgecolor("black")
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(FIGURE_PATH / "bonding_curve_tokens.pdf", bbox_inches="tight")
plt.show()


### Figure 2: Token Price ###
fig2, ax2 = plt.subplots(figsize=(10, 7))

ax2.plot(
    x,
    z_price,
    color="blue",
    marker="^",
    markersize=4,
    linewidth=2,
    label="Price",
    markeredgewidth=5,
)

ax2.set_xlabel("Total SOL Deposited", fontsize=FONT_SIZE)
ax2.set_ylabel("Price, SOL per Meme Coin", fontsize=FONT_SIZE)
ax2.tick_params(axis="both", labelsize=FONT_SIZE)
ax2.yaxis.get_offset_text().set_fontsize(FONT_SIZE)

formatter2 = ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((0, 0))
ax2.yaxis.set_major_formatter(formatter2)

ax2.axhline(
    y=initial_price, color="blue", linestyle="--", linewidth=2, label="Initial Price"
)
ax2.axvline(
    x=migration_price, color="red", linestyle="--", linewidth=2, label="Migration"
)

ax2.set_yticks(np.sort(np.append(ax2.get_yticks(), initial_price))[1:-1])
ax2.set_xticks(np.sort(np.append(ax2.get_xticks(), migration_price))[1:-1])

legend = ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=True,
    fontsize=FONT_SIZE,
)
legend.get_frame().set_edgecolor("black")

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(FIGURE_PATH / "bonding_curve_price.pdf", bbox_inches="tight")
plt.show()
