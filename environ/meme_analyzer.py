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
from collections import defaultdict


class MemeAnalyzer:
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        self.new_token_pool = new_token_pool
        self.txn = self._load_pickle("txn")
        self.transfer = self._load_pickle("transfer")
        self.block_created_time = self._load_creation_time()
        self.prc_date_df, self.pre_prc_date_df = self._build_price_df()
        self.migration_duration = (
            self.block_created_time - self.pre_prc_date_df.index.min()
        ).total_seconds()

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
            acts_list.append({"date": txn.date, "acts": acts_dict, "maker": txn.maker})
        acts_list.sort(key=lambda x: x["date"])
        return acts_list

    def get_unique_swapers(self) -> int:
        """Method to get the unique swapers of the meme token"""
        swapers = set()
        for swap in self.get_acts(Swap):
            swapers.add(swap["maker"])
        return len(swapers)

    def get_unique_non_swap_transfers(self) -> int:
        """Method to get the unique non-swap transfers of the meme token"""
        non_swap_transfers = set()
        for transfer in self.transfer:
            if transfer.date.replace(tzinfo=timezone.utc) < self.block_created_time:
                non_swap_transfers.add(transfer.txn_hash)

        for txn in self.txn:
            non_swap_transfers.discard(txn.txn_hash)

        return len(non_swap_transfers)

    def get_holdings_herf(self) -> float:
        """Method to get the herfindahl index of the holdings of the meme token"""
        holdings = defaultdict(float)

        for swap in [
            _
            for _ in self.get_acts(Swap)
            if _["date"].replace(tzinfo=timezone.utc) < self.block_created_time
        ]:
            if swap["acts"]:
                last_act = swap["acts"][list(swap["acts"].keys())[-1]]
                if last_act.typ == "Buy":
                    holdings[swap["maker"]] += last_act.base
                elif last_act.typ == "Sell":
                    holdings[swap["maker"]] -= last_act.base

        # calculate the herfindahl index
        total_holdings = sum(holdings.values())
        if total_holdings == 0:
            return 0.0
        herf = sum((holding / total_holdings) ** 2 for holding in holdings.values())
        return herf

    def resample_price(self) -> pd.DataFrame:
        """Method to resample the price data to the specified frequency"""
        # convert the index to how many seconds since the pool was created
        prc_resampled = self.prc_date_df.loc[
            self.prc_date_df.index >= self.block_created_time
        ].copy()
        prc_resampled.index = (
            prc_resampled.index - self.block_created_time
        ).total_seconds()

        # drop the duplicate index values
        prc_resampled = prc_resampled[~prc_resampled.index.duplicated(keep="last")]

        # take the 12*3600 seconds as the maximum index value
        prc_resampled = prc_resampled.reindex(range(0, 12 * 3600 + 1, 1)).ffill()

        return prc_resampled

    def append_pre_prc_date_df(self, prc_resampled: pd.DataFrame) -> pd.DataFrame:
        """Method to append the pre price data to the resampled price data"""
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

    def process_price(self, freq: str) -> pd.DataFrame:
        """Method to process the price data into different frequency"""

        prc_resampled = self.resample_price()
        match freq:
            case "1h":
                # only keep the value can be divided by 3600
                prc_resampled = prc_resampled[
                    (prc_resampled.index % 3600 == 0) & (prc_resampled.index != 0)
                ].copy()
                prc_resampled.index = prc_resampled.index / 3600  # convert to hours
            case "1min":
                # only keep the value can be divided by 60
                prc_resampled = prc_resampled[
                    prc_resampled.index % 60 == 0 & (prc_resampled.index != 0)
                ].copy()
                prc_resampled.index = prc_resampled.index / 60  # convert to minutes

        # append the last row of pre_prc_date_df to the resampled price data
        prc_resampled = self.append_pre_prc_date_df(prc_resampled)

        return prc_resampled

    def get_mdd(self, freq: str = "1min", before: Optional[float] = None) -> float:
        """Method to get the maximum drawdown of the meme token"""
        prc_date_df = self.resample_price()
        prc_date_df = self.append_pre_prc_date_df(prc_date_df)

        if before is not None:
            match freq:
                case "1min":
                    before = before * 60
                case "1h":
                    before = before * 3600
            prc_date_df = prc_date_df[(prc_date_df.index <= before)]

        prc_date_df["running_max"] = prc_date_df["price"].cummax()
        prc_date_df["drawdown"] = (
            prc_date_df["price"] - prc_date_df["running_max"]
        ) / prc_date_df["running_max"]
        mdd = prc_date_df["drawdown"].min()
        return mdd

    def get_ret(self, freq: str) -> pd.DataFrame:
        """Method to get the return of the meme token"""

        return (
            self.process_price(freq)
            .copy()
            .pct_change()
            .dropna()
            .rename(columns={"price": "ret"})
        )


if __name__ == "__main__":

    SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"
    NUM_OF_OBSERVATIONS = 10

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
            # if len(meme.get_acts(Swap)) > 2:
            # if (
            #     pool["token_address"]
            #     == "3quAiFqum25S7NtyDgzq8V3vhWB6U1v6GtWWNaNfpump"
            # ):
            print(
                f"Pool: {pool['token_address']}, "
                f"Max Drawdown: {meme.get_mdd(freq="1min") * 100:.2f}%, "
                f"Unique Swapers: {meme.get_unique_swapers()},"
                f"Unique Non-Swap Transfers: {meme.get_unique_non_swap_transfers()}",
                f"Holdings Herfindal Index: {meme.get_holdings_herf()}",
            )
            # (meme.get_ret(freq="1min") + 1).cumprod().plot()
