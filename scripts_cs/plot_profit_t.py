# Ridge plot of returns for three wallets with CI as shaded fill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, t
from pathlib import Path
from environ.constants import PROCESSED_DATA_PATH, FIGURE_PATH
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties


# --- Config ---
WALLETS = [
    "J8JSA7BGKmauruAD8A7fWwwz9UoPvbEKPksW1WDsqcd1",
    "E9D2wrgjfhbaopWjgkiwX4o9eVFjLQhUiCr5SBiZMpWn",
    "CcSVw6PGY655z9ava7pQhSkckmBL7rtkrjPGRVK5z1K3",
]

NAMING = ["Underperforming Trader", "Noise Trader", "KOL"]

FONT_SIZE = 14
N_GRID = 600
PERC_BOUNDS = (1, 99)  # trim global x-range to avoid extreme tails
Y_STEP = 0.5  # <1.0 overlaps; ~1.2 leaves gaps

# Alpha hierarchy (fill < CI < mean)
ALPHA_FILL = 0.30
ALPHA_CI = 0.55
ALPHA_MEAN = 0.95
ALPHA = 0.01  # 99% CI


# --- Load & clean ---
df = pd.read_csv(
    Path(PROCESSED_DATA_PATH) / "trader_project_profits.csv",
    usecols=["trader_address", "ret"],
)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret"])
df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
df = df.dropna(subset=["ret"])
df = df[df["trader_address"].isin(WALLETS)]

# Arrange series in the same order as WALLETS
series = []
for w in WALLETS:
    s = df.loc[df["trader_address"] == w, "ret"].astype(float).dropna()
    if len(s) < 2:
        raise ValueError(f"Wallet {w} has < 2 observations (need >= 2).")
    series.append((w, s))

# --- Shared x-grid ---
all_vals = pd.concat([s for _, s in series], axis=0)
x_lo, x_hi = np.percentile(all_vals, PERC_BOUNDS)
pad = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 0.1
xs = np.linspace(x_lo - pad, x_hi + pad, N_GRID)

# --- Colors: distinct per wallet ---
base_colors = (
    plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
)
colors = [base_colors[i % len(base_colors)] for i in range(len(series))]

# --- Plot ---
fig, ax = plt.subplots(figsize=(6.5, 3.8))
y_offset = 0.0

for (w, s), color, name in zip(series, colors, NAMING):
    kde = gaussian_kde(s.values)
    ys_raw = kde(xs)
    peak = ys_raw.max()
    ys = ys_raw / peak if peak > 0 else ys_raw

    # --- Ridge fill ---
    ax.fill_between(
        xs, y_offset, y_offset + ys, color=color, alpha=ALPHA_FILL, zorder=2
    )

    # Ridge outline
    ax.plot(xs, y_offset + ys, linewidth=1.0, color="black", alpha=0.8, zorder=3)

    # --- Mean & 99% CI ---
    n = len(s)
    mu = s.mean()
    sd = s.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = t.ppf(1 - ALPHA / 2, df=n - 1)
    half_width = tcrit * se
    ci_low, ci_high = mu - half_width, mu + half_width

    # Restrict xs to CI band
    ci_mask = (xs >= ci_low) & (xs <= ci_high)
    xs_ci = xs[ci_mask]
    ys_ci = ys[ci_mask]

    # CI fill (same color, stronger alpha than ridge fill)
    ax.fill_between(
        xs_ci, y_offset, y_offset + ys_ci, color=color, alpha=ALPHA_CI, zorder=4
    )

    # Mean line (dashed, highest alpha)
    mu_h = (kde(mu) / peak) if peak > 0 else 0.0
    ax.vlines(
        mu,
        y_offset,
        y_offset + mu_h,
        colors=color,
        linestyles="--",
        linewidth=1.8,
        alpha=ALPHA_MEAN,
        zorder=5,
    )

    # Wallet label
    ax.text(
        1.25,
        y_offset + 0.2,
        name,
        va="center",
        ha="right",
        fontsize=FONT_SIZE - 1,
        color="black",
        fontweight="bold",
    )

    y_offset += Y_STEP

# --- Aesthetics ---
ax.set_xlim(xs[0], -xs[0])
ax.set_yticks([])
ax.set_xlabel("Return", fontsize=FONT_SIZE, fontweight="bold")
ax.set_ylabel("Density", fontsize=FONT_SIZE, fontweight="bold")
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.tick_params(axis="x", labelsize=FONT_SIZE)
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
fig.tight_layout()

legend_handles = [
    mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.8, label="Mean"),
    mpatches.Patch(facecolor="grey", alpha=0.5, label="99% Confidence Interval"),
    mpatches.Patch(facecolor="grey", alpha=ALPHA_FILL, label="Distribution"),
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    fontsize=FONT_SIZE,
    frameon=False,
    prop=FontProperties(weight="bold"),
)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)

out = Path(FIGURE_PATH) / "ridge.pdf"
fig.savefig(out, bbox_inches="tight")
plt.show()
