"""Class to analyze meme token"""

import datetime
import json
import pickle
from collections import defaultdict, Counter
from datetime import UTC, timezone
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from environ.constants import NATIVE_ADDRESS_DICT, PROCESSED_DATA_PATH, TRUMP_BLOCK
from environ.data_class import NewTokenPool, Swap
from environ.db import fetch_native_pool_since_block
from environ.sol_fetcher import import_pool

SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"
MIGATOR = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"


def compute_herfindahl(var_dict: dict) -> float:
    """Compute Herfindahl index given a  dictionary"""
    values = np.array(list(var_dict.values()))
    if values.sum() == 0:
        return 0.0
    normalized = values / values.sum()
    return np.sum(normalized**2)


class MemeAnalyzer:
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        self.new_token_pool = new_token_pool
        self.txn = self._load_pickle("txn")
        self.transfer = self._load_pickle("transfer")
        self.block_created_time, self.launch_time, self.creator, self.pool_add = (
            self._load_creation()
        )
        self.dev_txn = 0
        self.dev_transfer = 0
        self.dev_transfer_amount = 0
        self.non_swap_transfer_hash = self._build_non_swap_transfer()
        self.prc_date_df, self.pre_prc_date_df = self._build_price_df()
        self.migration_duration = (
            self.block_created_time - self.pre_prc_date_df.index.min()
        ).total_seconds()
        self.swappers = self._build_swappers()

    def _load_pickle(self, attr: str):
        path = (
            f"{PROCESSED_DATA_PATH}/{attr}/"
            f"{self.new_token_pool.chain}/{self.new_token_pool.pool_add}.pkl"
        )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_creation(self) -> tuple[datetime.datetime, datetime.datetime, str, str]:
        """Method to load the creation time of the meme token"""
        path = (
            f"{PROCESSED_DATA_PATH}/creation/{self.new_token_pool.chain}/"
            f"{self.new_token_pool.pool_add}.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            file = json.load(f)
            creation_data = file["created_time"]
            lauch_time = file["launch_time"]
            creator_address = file["token_creator"]
            pool_adress = file["pumpfun_pool_address"]

        return (
            datetime.datetime.fromtimestamp(creation_data, UTC),
            lauch_time,
            creator_address,
            pool_adress,
        )

    def _build_non_swap_transfer(self) -> set[str]:
        """Method to get the unique non-swap transfers of the meme token"""
        non_swap_transfers = set()
        for transfer in self.transfer:
            if (
                transfer.date.replace(tzinfo=timezone.utc) < self.block_created_time
            ) & (MIGATOR not in [transfer.from_, transfer.to]):
                non_swap_transfers.add(transfer.txn_hash)

        for txn in self.txn:
            non_swap_transfers.discard(txn.txn_hash)

            if self.creator == txn.maker:
                self.dev_txn = 1

        for transfer in self.transfer:
            if transfer.txn_hash in non_swap_transfers and self.creator in [
                transfer.from_,
                transfer.to,
            ]:
                self.dev_transfer = 1

                if transfer.to == self.creator:
                    self.dev_transfer_amount += transfer.value

        return non_swap_transfers

    def _build_price_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Method to build the price DataFrame of the meme token"""
        prc_date_dict = {
            "block": [],
            "date": [],
            "price": [],
            "base": [],
        }
        for _, acts in enumerate(self.get_acts(Swap)):
            prc_date_dict["block"].append(acts["block"])
            prc_date_dict["date"].append(acts["date"].replace(tzinfo=timezone.utc))
            prc_date_dict["price"].append(
                acts["acts"][list(acts["acts"].keys())[-1]].price
            )
            prc_date_dict["base"].append(
                acts["acts"][list(acts["acts"].keys())[-1]].base
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

    def _build_swappers(self) -> dict[str, list]:
        """Method to get the swapers of the meme token"""
        swapers = defaultdict(list)
        for swap in self.get_acts(Swap):
            swapers[swap["maker"]].append(swap)
        return swapers

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
                    "maker": txn.maker,
                    "block": txn.block,
                }
            )
        acts_list.sort(key=lambda x: x["date"])
        return acts_list

    # Metrics for Meme Token Analysis
    def get_block_bundle_herf(self) -> float:
        """Method to get the Herfindahl index of the block bundle of the meme token"""
        return compute_herfindahl(
            self.pre_prc_date_df.groupby("block")["base"].sum().to_dict()
        )

    def get_unique_swapers(self) -> int:
        """Method to get the unique swapers of the meme token"""
        return len(self.swappers)

    def get_max_same_txn_per_swaper(self) -> float:
        """Method to get the max number of same transaction per swaper"""

        same_txn_list = []
        for swaper, swaps in self.swappers.items():
            txn_amount_list = [
                swap["acts"][list(swap["acts"].keys())[-1]].base for swap in swaps
            ]
            same_txn_list.append(Counter(txn_amount_list).most_common(1)[0][1])

        return max(same_txn_list) if same_txn_list else 0

    def get_non_swap_transfer_amount(self) -> int:
        """Method to get the unique non-swap transfers of the meme token"""

        transfer_amount = 0
        for transfer in self.transfer:
            if transfer.txn_hash in self.non_swap_transfer_hash:
                transfer_amount += transfer.value

        return transfer_amount

    def get_holdings_herf(self) -> float:
        """Method to get the Herfindahl index of the holdings of the meme token"""
        holdings = defaultdict(float)

        for swap in [
            s
            for s in self.get_acts(Swap)
            if s["date"].replace(tzinfo=timezone.utc) < self.block_created_time
        ]:
            last_act = swap["acts"][list(swap["acts"].keys())[-1]]
            if last_act.typ == "Buy":
                holdings[swap["maker"]] += last_act.base
            elif last_act.typ == "Sell":
                holdings[swap["maker"]] -= last_act.base

        return compute_herfindahl(holdings)

    def analyze_non_swap_transfer_graph(self):
        """Builds and visualizes the non-swap transfer graph and returns in/out Herfindahl index."""
        swap_hashes = {txn.txn_hash for txn in self.txn}
        G = nx.DiGraph()

        for transfer in self.transfer:
            if (
                transfer.date.replace(tzinfo=timezone.utc) < self.block_created_time
                and transfer.txn_hash not in swap_hashes
            ):
                G.add_edge(transfer.from_, transfer.to, weight=transfer.value)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 7))
        # color the dev transfer in red
        for node in G.nodes:
            if node == self.creator:
                G.nodes[node]["color"] = "red"
            else:
                G.nodes[node]["color"] = "skyblue"
        node_colors = [G.nodes[node]["color"] for node in G.nodes]
        edge_widths = [
            G[u][v]["weight"] / (1_000_000_00 - 206_900_00) for u, v in G.edges
        ]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            width=edge_widths,
            alpha=0.7,
        )
        plt.show()

        # out_degree_centrality = nx.out_degree_centrality(G)
        # average_out_degree_centrality = np.mean(list(out_degree_centrality.values()))

        return 1

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

    def get_ret_before(
        self, freq: str = "1min", before: Optional[float] = None
    ) -> float:
        """Method to get the return of the meme token"""

        prc_date_df = self.resample_price()
        prc_date_df = self.append_pre_prc_date_df(prc_date_df)

        if before is not None:
            match freq:
                case "1min":
                    before = before * 60
                case "1h":
                    before = before * 3600
            prc_date_df = prc_date_df[(prc_date_df.index <= before)]

        # only keep the first and last row
        starting_price = prc_date_df["price"].iloc[0]
        ending_price = prc_date_df["price"].iloc[-1]

        return (ending_price - starting_price) / starting_price

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
            .rename(columns={"price": "ret"})[["ret"]]
        )


if __name__ == "__main__":

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
            # if len(meme.get_acts(Swap)) > 2:
            # if (
            #     pool["token_address"]
            #     == "3quAiFqum25S7NtyDgzq8V3vhWB6U1v6GtWWNaNfpump"
            # ):

            # out_deg, in_deg = meme.analyze_non_swap_transfer_graph()

            print(
                # f"Pool: {pool['token_address']}, "
                f"Max Drawdown: {meme.get_mdd(freq="1min", before=1) * 100:.2f}%, "
                # f"Unique Swapers: {meme.get_unique_swapers()},"
                # f"Unique Non-Swap Transfers: {len(meme.non_swap_transfer_hash)}, ",
                # f"Holdings Herfindal Index: {meme.get_holdings_herf()}",
                # # f"Non-Swap Transfer Graph Out Herfindahl: {out_deg}, "
                # # f"Non-Swap Transfer Graph In Herfindahl: {in_deg}",
                f"Degree: {meme.analyze_non_swap_transfer_graph()}",
                # f"Transfer amount: {meme.get_non_swap_transfer_amount()}",
                # f"Dev Txn: {meme.dev_txn}, "
                # f"Dev Transfer: {meme.dev_transfer}",
                # f"Dev Transfer Amount: {meme.dev_transfer_amount}",
                f"Return: {meme.get_ret_before(freq='1min', before=1) * 100:.2f}%, ",
                f"Max Same Txn per Swaper: {meme.get_max_same_txn_per_swaper():.2f}, ",
                f"Block Bundle Herfindahl: {meme.get_block_bundle_herf()}",
            )
            # (meme.get_ret(freq="1min") + 1).cumprod().plot()
