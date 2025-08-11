"""Plot distribution of meme coin performance for multiple bot indicators"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from environ.constants import PROCESSED_DATA_CS_PATH, FIGURE_PATH

FONT_SIZE = 20
sns.set_theme(style="white")

pfm = pd.read_csv(f"{PROCESSED_DATA_CS_PATH}/pfm_cs.csv")

# Bot-related columns to iterate over
bot_vars = ["launch_bundle", "volume_bot", "bot_comment_num"]

for bot_col in bot_vars:
    # Map 0/1 -> labels (title case)
    order = ["Without bot", "With bot"]
    pfm["bot_label"] = pfm[bot_col].map({0: "Without bot", 1: "With bot"})
    pfm_subset = pfm[pfm["bot_label"].isin(order)].copy()
    pfm_subset["bot_label"] = pd.Categorical(
        pfm_subset["bot_label"], categories=order, ordered=True
    )

    # Compute limits from scatter points
    xy = pfm_subset[["max_ret", "dump_duration"]].dropna()
    if xy.empty:
        print(f"Skip {bot_col}: no data after filtering.")
        continue
    x_min, x_max = xy["max_ret"].min(), xy["max_ret"].max()
    y_min, y_max = xy["dump_duration"].min(), xy["dump_duration"].max()

    # Fixed colors: Without bot = blue, With bot = red
    color_map = {
        "Without bot": "tab:blue",
        "With bot": "tab:red",
    }
    levels = order

    # JointGrid base
    g = sns.JointGrid(
        data=pfm_subset,
        x="max_ret",
        y="dump_duration",
        height=5,
        space=0,
    )

    # Scatter in joint panel
    sns.scatterplot(
        data=pfm_subset,
        x="max_ret",
        y="dump_duration",
        hue="bot_label",
        palette=color_map,
        alpha=0.3,
        edgecolor=None,
        ax=g.ax_joint,
        legend=False,  # we'll build our own to avoid extra entries
    )

    # Marginal histograms (no density)
    BINS = 30
    for lvl in levels:
        sub = pfm_subset[pfm_subset["bot_label"] == lvl]
        if sub.empty:
            continue
        c = color_map[lvl]
        # X marginal
        sns.histplot(
            data=sub,
            x="max_ret",
            bins=BINS,
            element="bars",
            fill=True,
            alpha=0.45,
            ax=g.ax_marg_x,
            color=c,
            stat="count",
            common_norm=False,
        )
        # Y marginal
        sns.histplot(
            data=sub,
            y="dump_duration",
            bins=BINS,
            element="bars",
            fill=True,
            alpha=0.45,
            ax=g.ax_marg_y,
            color=c,
            stat="count",
            common_norm=False,
        )

    # Mean lines
    for lvl in levels:
        sub = pfm_subset[pfm_subset["bot_label"] == lvl]
        if sub.empty:
            continue
        x_mean = sub["max_ret"].mean()
        y_mean = sub["dump_duration"].mean()
        if not (np.isfinite(x_mean) and np.isfinite(y_mean)):
            continue
        c = color_map[lvl]
        g.ax_joint.axvline(x_mean, color=c, linestyle="--", linewidth=2)
        g.ax_joint.axhline(y_mean, color=c, linestyle="--", linewidth=2)
        g.ax_marg_x.axvline(x_mean, color=c, linestyle="--", linewidth=2)
        g.ax_marg_y.axhline(y_mean, color=c, linestyle="--", linewidth=2)

    # Limits (exact min/max of scatter)
    g.ax_joint.set_xlim(x_min, x_max)
    g.ax_joint.set_ylim(y_min, y_max)
    g.ax_marg_x.set_xlim(x_min, x_max)
    g.ax_marg_y.set_ylim(y_min, y_max)

    # Custom legend (only hue)
    hue_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=color_map[lvl],
               label=lvl, markersize=8)
        for lvl in levels
    ]
    hue_legend = g.ax_joint.legend(
        handles=hue_handles,
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),
        frameon=False,
        ncols=2,
        fontsize=FONT_SIZE + 2,
    )
    for text in hue_legend.get_texts():
        text.set_fontweight("bold")

    # Labels & ticks
    g.ax_joint.set_xlabel("Ln(Max Return)", fontsize=FONT_SIZE + 2, fontweight="bold")
    g.ax_joint.set_ylabel("Ln(Dump Duration)", fontsize=FONT_SIZE + 2, fontweight="bold")
    g.ax_joint.xaxis.set_tick_params(labelsize=FONT_SIZE + 2)
    g.ax_joint.yaxis.set_tick_params(labelsize=FONT_SIZE + 2)
    for lbl in g.ax_joint.get_xticklabels() + g.ax_joint.get_yticklabels():
        lbl.set_fontweight("bold")

    # Save each plot
    g.figure.savefig(f"{FIGURE_PATH}/{bot_col}_hist.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved {FIGURE_PATH}/{bot_col}_hist.pdf")
