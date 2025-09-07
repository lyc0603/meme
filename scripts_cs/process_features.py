#!/usr/bin/env python3
"""
Merged pipeline:
1) Load pfm + trader_project_profits
2) Build rolling trader features (per trade date) and label
3) Merge token-level flags from pfm
4) Build first-trade features via MemeAnalyzer (per token, per trader)
5) Merge all features into df

Outputs:
- df: pandas.DataFrame with trader, token, time-series features, token flags, and first-trade bot features.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from datetime import UTC

from environ.constants import PROCESSED_DATA_CS_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool, Swap
from environ.meme_analyzer import MemeAnalyzer
from environ.utils import handle_first_wash_bot, handle_first_comment_bot

# ----------------------------- Config -----------------------------
MIN_HISTORY = 15  # require at least this many prior trades for rolling features
PERCENT_ADD_ONE = True  # keep +1 on some return features (as in your original script)
SAVE_PATH = PROCESSED_DATA_CS_PATH / "trader_features_merged.csv"  # optional
SHOW_PROGRESS = True


def build_trader_rolling_features(
    pfm: pd.DataFrame, trader_project: pd.DataFrame
) -> pd.DataFrame:
    """
    Build rolling trader features (per trader & trade date) and attach a label (the trade's ret).
    """
    # keep only traders who ever traded tokens appearing in pfm
    token_list = pfm["token_address"].unique()
    trader_project = trader_project.copy()
    trader_project["in_token_list"] = trader_project["meme"].isin(token_list)
    trader_project["trader_in_token_list"] = trader_project.groupby("trader_address")[
        "in_token_list"
    ].transform("any")
    trader_project = trader_project[trader_project["trader_in_token_list"]].copy()
    trader_project["date"] = pd.to_datetime(trader_project["date"])

    rows = []

    # iterate through traders
    iterator = trader_project.groupby("trader_address")
    if SHOW_PROGRESS:
        iterator = tqdm(iterator, desc="Rolling features (by trader)")

    for trader_address, project_df in iterator:
        project_df = project_df.sort_values("date", ascending=True).reset_index(
            drop=True
        )

        # iterate through trades that are in token_list (your original intent)
        for _, trade in project_df.loc[project_df["in_token_list"]].iterrows():
            date = trade["date"]
            rolling_df = project_df.loc[project_df["date"] < date].copy()

            if len(rolling_df) <= MIN_HISTORY:
                continue

            # ----- features -----
            # note: original code added +1 to some features; keep behavior guarded by flag
            if PERCENT_ADD_ONE:
                average_ret = rolling_df["ret"].mean() + 1
                last_ret = rolling_df["ret"].iloc[-1] + 1
                five_to_one_ret = rolling_df["ret"].iloc[-5:].mean() + 1
                ten_to_six_ret = rolling_df["ret"].iloc[-10:-5].mean() + 1
                fifteen_to_eleven_ret = rolling_df["ret"].iloc[-15:-10].mean() + 1
            else:
                average_ret = rolling_df["ret"].mean()
                last_ret = rolling_df["ret"].iloc[-1]
                five_to_one_ret = rolling_df["ret"].iloc[-5:].mean()
                ten_to_six_ret = rolling_df["ret"].iloc[-10:-5].mean()
                fifteen_to_eleven_ret = rolling_df["ret"].iloc[-15:-10].mean()

            std_ret = rolling_df["ret"].std(ddof=1)
            t_stat, _ = stats.ttest_1samp(rolling_df["ret"], 0)
            num_trades = int(len(rolling_df))
            time_since_last_trade = float(
                (date - rolling_df["date"].iloc[-1]).total_seconds()
            )
            time_since_first_trade = float(
                (date - rolling_df["date"].iloc[0]).total_seconds()
            )

            rows.append(
                {
                    "date": date,
                    "trader_address": trader_address,
                    "token_address": trade["meme"],
                    "average_ret": average_ret,
                    "std_ret": std_ret,
                    "last_ret": last_ret,
                    "five_to_one_ret": five_to_one_ret,
                    "ten_to_six_ret": ten_to_six_ret,
                    "fifteen_to_eleven_ret": fifteen_to_eleven_ret,
                    "t_stat": t_stat,
                    "num_trades": num_trades,
                    "time_since_last_trade": time_since_last_trade,
                    "time_since_first_trade": time_since_first_trade,
                    "label": trade["ret"],
                }
            )

    df = pd.DataFrame(rows)
    # clean inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return df


def merge_token_flags(df: pd.DataFrame, pfm: pd.DataFrame) -> pd.DataFrame:
    """
    Merge token-level flags/fields from pfm into df.
    """
    keep_cols = [
        "token_address",
        "launch_bundle",
        "sniper_bot",
        "chain",
        "bot_comment_num",
        "volume_bot",
    ]
    keep_cols = [c for c in keep_cols if c in pfm.columns]
    merged = df.merge(
        pfm[keep_cols].drop_duplicates("token_address"), on="token_address", how="left"
    )
    return merged


def build_first_trade_features(df_with_chain: pd.DataFrame) -> pd.DataFrame:
    """
    For each (token_address, chain), instantiate a MemeAnalyzer, derive wash/comment bot timestamps,
    then, for each trader that appears in df_with_chain for that token, find the trader's first matching Swap
    (maker == trader_id) and record first-trade features.
    """
    # Sanity: require chain
    if "chain" not in df_with_chain.columns:
        raise ValueError("Expected 'chain' column in df; make sure pfm merge added it.")

    trader_feature_rows = []

    grouped = df_with_chain.groupby(["token_address", "chain"], sort=False)
    if SHOW_PROGRESS:
        grouped = tqdm(
            grouped,
            total=len(df_with_chain["token_address"].unique()),
            desc="First-trade features (by token)",
        )

    for (token_add, chain), group in grouped:
        # Initialize a minimal NewTokenPool (mirrors your original constructor usage)
        meme = MemeAnalyzer(
            NewTokenPool(
                token0=SOL_TOKEN_ADDRESS,
                token1=token_add,
                fee=0,
                pool_add=token_add,
                block_number=0,
                chain=chain,
                base_token=token_add,
                quote_token=SOL_TOKEN_ADDRESS,
                txns={},
            ),
        )

        # Identify first wash/comment bot trade times relative to launch
        wash_rows = handle_first_wash_bot(meme, token_add, meme.launch_time)
        comment_rows = handle_first_comment_bot(meme, token_add, meme.launch_time)

        # convenience: safely get thresholds (may be empty)
        wash_first_time = None
        if wash_rows:
            try:
                wash_first_time = wash_rows[0]["first_trade_time"]
            except Exception:
                print(
                    f"[WARN] Could not parse wash bot time for {token_add} on {chain}"
                )
                wash_first_time = None

        comment_first_time = None
        if comment_rows:
            try:
                comment_first_time = comment_rows[0]["first_comment_time"]
            except Exception:
                print(
                    f"[WARN] Could not parse comment bot time for {token_add} on {chain}"
                )
                comment_first_time = None

        # Iterate traders that appear in df for this token & chain
        for trader_add in group["trader_address"].unique():
            # Walk through meme swaps to find first maker == trader
            first_match = None
            for txn in meme.get_acts(Swap):
                try:
                    if txn["maker"] == trader_add:
                        first_match = txn
                        break
                except Exception:
                    continue

            if first_match is None:
                # Trader might not have an on-chain "maker" match in this pool (keep NaNs for merge)
                trader_feature_rows.append(
                    {
                        "trader_address": trader_add,
                        "token_address": token_add,
                        "chain": chain,
                        "first_txn_date": pd.NaT,
                        "first_txn_price": np.nan,
                        "time_since_launch": np.nan,
                        "wash_trading_bot": (
                            0 if wash_first_time is None else np.nan
                        ),  # unknown if we don't have a date
                        "comment_bot": 0 if comment_first_time is None else np.nan,
                    }
                )
                continue

            # Normalize datetime (ensure tz-aware)
            trader_txn_date = first_match["date"]
            if trader_txn_date.tzinfo is None:
                trader_txn_date = trader_txn_date.replace(tzinfo=UTC)

            # price extraction
            try:
                trader_txn_price = first_match["acts"][0].price
            except Exception:
                trader_txn_price = np.nan

            # time since launch (seconds)
            try:
                time_since_launch = float(
                    (trader_txn_date - meme.launch_time).total_seconds()
                )
            except Exception:
                time_since_launch = np.nan

            # bot indicators: 1 if trader's first trade is at/after bot's first trade
            wash_bot = 0
            if wash_first_time is not None:
                try:
                    wash_bot = int(trader_txn_date >= wash_first_time)
                except Exception:
                    wash_bot = np.nan

            comment_bot = 0
            if comment_first_time is not None:
                try:
                    comment_bot = int(trader_txn_date >= comment_first_time)
                except Exception:
                    comment_bot = np.nan

            trader_feature_rows.append(
                {
                    "trader_address": trader_add,
                    "token_address": token_add,
                    "chain": chain,
                    "first_txn_date": trader_txn_date,
                    "first_txn_price": trader_txn_price,
                    "time_since_launch": time_since_launch,
                    "wash_trading_bot": wash_bot,
                    "comment_bot": comment_bot,
                }
            )

    return pd.DataFrame(trader_feature_rows)


def main():
    """Main entry point for feature processing."""

    pfm = pd.read_csv(PROCESSED_DATA_CS_PATH / "pfm_cs.csv")
    trader_project = pd.read_csv(PROCESSED_DATA_CS_PATH / "trader_project_profits.csv")

    df = build_trader_rolling_features(pfm, trader_project)
    df = merge_token_flags(df, pfm)
    trader_first_df = build_first_trade_features(df)

    df = df.merge(
        trader_first_df,
        on=["trader_address", "token_address", "chain"],
        how="left",
    )

    # final cleanup
    df = df.replace([np.inf, -np.inf], np.nan)

    # Optional save
    try:
        df.to_csv(SAVE_PATH, index=False)
        print(f"[OK] Saved merged features to: {SAVE_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save to {SAVE_PATH}: {e}")

    print(df.head())
    print(f"Final shape: {df.shape}")

    # Expose df for importers (if this file is imported elsewhere)
    return df


if __name__ == "__main__":
    df = main()
