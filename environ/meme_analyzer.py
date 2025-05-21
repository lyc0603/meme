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


class MemeAnalyzer:
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        # load the pool data
        self.new_token_pool = new_token_pool

        # load the transaction, transfer and smart contract data
        # for attr in ["txn", "transfer", "smart_contract"]:
        for attr in ["txn"]:
            file_path = (
                f"{PROCESSED_DATA_PATH}/{attr}/"
                f"{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl"
            )
            with open(file_path, "rb") as f:
                setattr(self, attr, pickle.load(f))

        # load the block created time
        self.block_created_time = json.load(
            open(
                f"{PROCESSED_DATA_PATH}/creation/\
{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.json",
                "r",
                encoding="utf-8",
            )
        )["created_time"]

        # extract the price data
        prc_date_dict = {
            "date": [],
            "price": [],
        }
        for idx, acts in enumerate(self.get_acts(Swap)):
            if idx == 0:
                prc_date_dict["date"].append(
                    datetime.datetime.fromtimestamp(self.block_created_time, UTC)
                )
                prc_date_dict["price"].append(
                    acts["acts"][list(acts["acts"].keys())[-1]].price
                )
            prc_date_dict["date"].append(acts["date"].replace(tzinfo=timezone.utc))
            prc_date_dict["price"].append(
                acts["acts"][list(acts["acts"].keys())[-1]].price
            )
        self.prc_date_df = pd.DataFrame(prc_date_dict)
        self.prc_date_df["date"] = pd.to_datetime(self.prc_date_df["date"])
        self.prc_date_df = self.prc_date_df.set_index("date").sort_index()
        self.prc_date_df["price"] = self.prc_date_df["price"].replace(0, np.nan)

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
        return acts_list

    def process_price(self, freq: str) -> pd.Series:
        """Method to process the price data into different frequency"""
        prc_date_df = self.prc_date_df.copy()
        prc_resampled = prc_date_df["price"].resample(freq).last().ffill()

        # Normalize index to start at hour = 0
        start_time = prc_resampled.index[0]
        normalized_hours = (prc_resampled.index - start_time).total_seconds() / 3600

        # Reindex to ensure a full 12-hour range even if data is shorter
        full_hour_range = pd.timedelta_range(start="0h", end="12h", freq=freq)
        full_hour_index = full_hour_range.total_seconds() / 3600  # in hours
        prc_resampled.index = normalized_hours
        prc_resampled = prc_resampled.reindex(full_hour_index).ffill()

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

    def get_ret(self, freq="1min") -> pd.DataFrame:
        """Method to get the return of the meme token"""

        return (
            self.process_price(freq).copy().pct_change().dropna().to_frame(name="ret")
        )


if __name__ == "__main__":
    for chain in [
        "ethereum",
    ]:
        for pool in fetch_native_pool_since_block(
            chain, TRUMP_BLOCK[chain], pool_number=100
        ):
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
            if len(meme.get_acts(Swap)) > 2:
                # if args["pool"] == "0xd6feCb0620abad24510D5192Dba1F1d931B232eB":
                print(
                    f"Max Drawdown: {meme.get_mdd(freq="1min", before=1) * 100:.2f}%, "
                )
                print(args)
                (meme.get_ret(freq="1min") + 1).cumprod().plot()
                # meme.process_price(freq="1h")
