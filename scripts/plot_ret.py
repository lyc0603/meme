"""Script to plot the returns of the meme token"""

from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from environ.constants import (
    FIGURE_PATH,
    NATIVE_ADDRESS_DICT,
    TRUMP_BLOCK,
    UNISWAP_V3_FACTORY_DICT,
)
from environ.data_class import NewTokenPool, Swap
from environ.db import fetch_native_pool_since_block
from environ.meme_analyzer import MemeAnalyzer

NUM_OF_OBSERVATIONS = 100


def plot_ret(
    chain: str, freq: str, con: dict[str, float | Callable[[float], bool]]
) -> None:
    """
    Plot the mean returns with std of the meme token
    """
    pool_num = 0
    df_ret = pd.DataFrame()

    for pool in fetch_native_pool_since_block(
        chain, TRUMP_BLOCK[chain], pool_number=1000
    ):
        try:
            if pool_num >= NUM_OF_OBSERVATIONS:
                break
            args = pool["args"]
            meme = MemeAnalyzer(
                NewTokenPool(
                    token0=args["token0"],
                    token1=args["token1"],
                    fee=args["fee"],
                    pool_add=args["pool"],
                    block_number=pool["blockNumber"],
                    chain=chain,
                    base_token=(
                        args["token0"]
                        if args["token0"] != NATIVE_ADDRESS_DICT[chain]
                        else args["token1"]
                    ),
                    quote_token=(
                        args["token1"]
                        if args["token1"] != NATIVE_ADDRESS_DICT[chain]
                        else args["token0"]
                    ),
                    txns={},
                ),
            )
            if len(meme.get_acts(Swap)) >= 2:
                pool_num += 1
                if con["condition"](meme.get_mdd(freq=freq, before=con["before"])):
                    df_log_ret = meme.get_ret(freq=freq)
                    df_log_ret["ret"] = np.log(df_log_ret["ret"] + 1)
                    df_log_ret = df_log_ret.rename(
                        columns={"ret": f"{args['pool']}_{args['fee']}"}
                    )
                    # for t_idx in df_log_ret.index:
                    #     plt.scatter(
                    #         [t_idx] * len(df_log_ret.columns),
                    #         df_log_ret.loc[t_idx].values,
                    #         color=UNISWAP_V3_FACTORY_DICT[chain]["color"],
                    #         alpha=0.2,
                    #         s=10,
                    #         marker="x",
                    #     )
                    df_ret = pd.concat(
                        [
                            df_ret,
                            df_log_ret[f"{args['pool']}_{args['fee']}"],
                        ],
                        axis=1,
                    )
        except Exception as e:
            print(f"Error in pool {pool['args']['pool']}: {e}")
            continue

    # winsorize the rows
    df_ret["mean"] = df_ret.mean(axis=1)
    df_ret["std"] = df_ret.std(axis=1)

    # plot the line chart with mean and error bars
    plt.plot(
        [0] + df_ret.index.tolist(),
        [0] + df_ret["mean"].values.tolist(),
        color=UNISWAP_V3_FACTORY_DICT[chain]["color"],
        lw=1,
        alpha=1,
    )
    plt.errorbar(
        df_ret.index,
        df_ret["mean"],
        yerr=df_ret["std"],
        fmt="x-",  # line with circular markers
        color=UNISWAP_V3_FACTORY_DICT[chain]["color"],
        ecolor=UNISWAP_V3_FACTORY_DICT[chain]["color"],
        elinewidth=1.5,
        capsize=4,
        label=UNISWAP_V3_FACTORY_DICT[chain]["name"],
        lw=1,
        alpha=1,
    )


if __name__ == "__main__":

    conditions = {
        "all": {"before": 1.0, "condition": lambda mdd: mdd < 1},
        "lower_-0.5": {
            "before": 1.0,
            "condition": lambda mdd: mdd < -0.5,
        },
        "greater_-0.5": {
            "before": 1.0,
            "condition": lambda mdd: mdd >= -0.5,
        },
    }

    for freq in ["1h"]:
        for label, con in conditions.items():
            plt.figure(figsize=(4, 3))
            for chain in ["ethereum", "base", "polygon", "bnb"]:
                plot_ret(chain, freq, con)

            # plot the horizontal dashed line at 0
            plt.axhline(0, color="black", linestyle="--")
            plt.ylabel("Log Return")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"{FIGURE_PATH}/ret_{freq}_{label}.pdf",
                dpi=300,
            )
            plt.show()
