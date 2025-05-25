"""Class to analyze meme token"""

import datetime
import json
import pickle
from datetime import UTC, timezone
from typing import Optional

import numpy as np
import pandas as pd

from environ.constants import NATIVE_ADDRESS_DICT, PROCESSED_DATA_PATH, TRUMP_BLOCK
from environ.data_class import NewTokenPool, Swap
from environ.db import fetch_native_pool_since_block
from environ.sol_fetcher import import_pool


class MemeAnalyzer:
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        self.new_token_pool = new_token_pool
        self.txn = self._load_pickle("txn")
        self.block_created_time = self._load_creation_time()
        self.prc_date_df, self.pre_prc_date_df = self._build_price_df()

    def _load_pickle(self, attr: str):
        path = (
            f"{PROCESSED_DATA_PATH}/{attr}/"
            f"{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl"
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_creation_time(self) -> datetime.datetime:
        """Method to load the creation time of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/creation/{self.new_token_pool.chain}/"
            f"{self.new_token_pool.pool_add}.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            creation_data = json.load(f)["created_time"]
        return datetime.datetime.fromtimestamp(creation_data, UTC)

    def _build_price_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Method to build the price DataFrame of the meme token"""
        prc_date_dict = {
            "date": [],
            "price": [],
        }
        for idx, acts in enumerate(self.get_acts(Swap)):
            # if idx == 0:
            #     prc_date_dict["date"].append(acts["date"].replace(tzinfo=timezone.utc))
            #     prc_date_dict["price"].append(
            #         acts["acts"][list(acts["acts"].keys())[-1]].price
            #     )
            prc_date_dict["date"].append(acts["date"].replace(tzinfo=timezone.utc))
            prc_date_dict["price"].append(
                acts["acts"][list(acts["acts"].keys())[-1]].price
            )
        prc_date_df = pd.DataFrame(prc_date_dict)
        prc_date_df = prc_date_df.set_index("date").sort_index()
        prc_date_df["price"] = prc_date_df["price"].replace(0, np.nan)

        pre_prc_date_df = prc_date_df.loc[
            prc_date_df.index < self.block_created_time
        ].copy()
        prc_date_df = prc_date_df.loc[
            prc_date_df.index >= self.block_created_time
        ].copy()

        return prc_date_df, pre_prc_date_df

    def get_acts(self, act: type) -> list:
        """Method to get the swap transaction of the meme token"""
        acts_list = []
        for txn in self.txn:
            acts_dict = {
                k: v for k, v in sorted(txn.acts.items()) if isinstance(v, act)
            }
            if not acts_dict:
                continue
            acts_list.append(
                {
                    "date": txn.date,
                    "acts": acts_dict,
                }
            )
        acts_list.sort(key=lambda x: x["date"])
        return acts_list

    def process_price(self, freq: str) -> pd.Series:
        """Method to process the price data into different frequency"""

        # Resample the price data to the specified frequency
        # prc_resampled = price_resample(freq, self.prc_date_df.copy())

        prc_resampled = self.prc_date_df.loc[
            self.prc_date_df.index >= self.block_created_time
        ].copy()
        prc_resampled = self.prc_date_df.copy()["price"].resample(freq).last().ffill()

        # Normalize index to start at hour = 0

        match freq:
            case "1min":
                start_time = self.block_created_time.replace(second=0)
            case "1h":
                # keep the hour of start time
                start_time = self.block_created_time.replace(minute=0, second=0)

        normalized_hours = (prc_resampled.index - start_time).total_seconds() / 3600

        # Reindex to ensure a full 12-hour range even if data is shorter
        full_hour_range = pd.timedelta_range(start="0h", end="12h", freq=freq)
        full_hour_index = full_hour_range.total_seconds() / 3600  # in hours
        prc_resampled.index = normalized_hours
        prc_resampled = prc_resampled.reindex(full_hour_index).ffill()

        # append the last row of pre_prc_date_df to the resampled price data
        if not self.pre_prc_date_df.empty:
            last_pre_price = self.pre_prc_date_df["price"].iloc[-1]
            prc_resampled = pd.concat(
                [
                    pd.Series(
                        [last_pre_price],
                        index=[-1],
                        name="price",
                    ),
                    prc_resampled,
                ]
            )

        return prc_resampled

    def get_mdd(self, freq: str = "1min", before: Optional[float] = None) -> float:
        """Method to get the maximum drawdown of the meme token"""
        prc_date_df = self.process_price(freq).copy().to_frame(name="price")
        if before is not None:
            prc_date_df = prc_date_df[prc_date_df.index <= before]
        prc_date_df["running_max"] = prc_date_df["price"].cummax()
        prc_date_df["drawdown"] = (
            prc_date_df["price"] - prc_date_df["running_max"]
        ) / prc_date_df["running_max"]
        mdd = prc_date_df["drawdown"].min()
        return mdd

    def get_ret(self, freq: str) -> pd.DataFrame:
        """Method to get the return of the meme token"""

        return (
            self.process_price(freq).copy().pct_change().dropna().to_frame(name="ret")
        )


def price_resample(freq: str, df: pd.DataFrame) -> pd.Series:
    """Method to resample the price data into different frequency and keep the last value

    Args:
        freq (str): Frequency to resample the data
        df (pd.DataFrame): DataFrame containing the price data with "date" and "price" columns
    Returns:
        pd.Series: Resampled price data
    """

    return df["price"].resample(freq).last().ffill()


if __name__ == "__main__":

    SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"
    NUM_OF_OBSERVATIONS = 3

    for chain in [
        # "pumpfun",
        "raydium",
        # "ethereum",
    ]:
        for pool in (
            fetch_native_pool_since_block(
                chain, TRUMP_BLOCK[chain], pool_number=NUM_OF_OBSERVATIONS
            )
            if chain not in ["pumpfun", "raydium"]
            else import_pool(
                chain,
                NUM_OF_OBSERVATIONS,
            )
        ):
            if chain not in ["pumpfun", "raydium"]:
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
            else:
                meme = MemeAnalyzer(
                    NewTokenPool(
                        token0=SOL_TOKEN_ADDRESS,
                        token1=pool["token_address"],
                        fee=0,
                        pool_add=pool["token_address"],
                        block_number=0,
                        chain=chain,
                        base_token=pool["token_address"],
                        quote_token=SOL_TOKEN_ADDRESS,
                        txns={},
                    ),
                )
            if len(meme.get_acts(Swap)) > 2:
                # if (
                #     pool["token_address"]
                #     == "3quAiFqum25S7NtyDgzq8V3vhWB6U1v6GtWWNaNfpump"
                # ):
                print(
                    f"Pool: {pool['token_address']}, "
                    f"Max Drawdown: {meme.get_mdd(freq="1h") * 100:.2f}%, "
                )
                (meme.get_ret(freq="1h") + 1).cumprod().plot()
