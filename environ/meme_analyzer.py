"""Class to analyze meme token"""

import datetime
from collections import Counter, defaultdict
from datetime import timezone
from typing import Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from environ.constants import SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool, Swap
from environ.meme_base import MemeBase
from environ.sol_fetcher import import_pool
import mplfinance as mpf

MIGATOR = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"


def compute_herfindahl(var_dict: list | dict) -> float:
    """Compute Herfindahl index given a  dictionary"""
    if isinstance(var_dict, list):
        values = np.array(var_dict)
    else:
        values = np.array(list(var_dict.values()))
    if values.sum() == 0:
        return 0.0
    normalized = values / values.sum()
    return np.sum(normalized**2)


class MemeAnalyzer(MemeBase):
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        super().__init__(new_token_pool)
        self.prc_date_df, self.pre_prc_date_df = self._build_price_df()
        self.migration_duration = (
            self.migrate_time - self.pre_prc_date_df.index.min()
        ).total_seconds()
        self.comment_list = self._build_comment_list()

    # Build-in Methods
    def _build_price_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Method to build the price DataFrame of the meme token"""
        prc_date_dict = {
            "block": [],
            "date": [],
            "price": [],
            "base": [],
            "quote": [],
            "usd": [],
        }
        for _, acts in enumerate(self.get_acts(Swap)):
            last_act = acts["acts"][list(acts["acts"].keys())[-1]]

            prc_date_dict["block"].append(acts["block"])
            prc_date_dict["date"].append(acts["date"].replace(tzinfo=timezone.utc))
            for key, value in {
                "price": last_act.price,
                "base": last_act.base,
                "quote": last_act.quote,
                "usd": last_act.usd,
            }.items():
                prc_date_dict[key].append(value)

        prc_date_df = pd.DataFrame(prc_date_dict)
        prc_date_df = prc_date_df.set_index("date").sort_index()
        prc_date_df["price"] = prc_date_df["price"].replace(0, np.nan)

        pre_prc_date_df = prc_date_df.loc[prc_date_df.index < self.migrate_time].copy()
        prc_date_df = prc_date_df.loc[prc_date_df.index >= self.migrate_time].copy()

        return prc_date_df, pre_prc_date_df

    def _build_comment_list(self) -> list[dict[str, Any]]:
        """Method to build the comment dictionary of the meme token"""
        reply_list = []
        for reply in self.comment:
            time = datetime.datetime.fromtimestamp(
                reply["comment"]["timestamp"] / 1000, tz=timezone.utc
            )
            if time <= self.migrate_time:
                reply_list.append(
                    {
                        "replier": reply["comment"]["user"],
                        "time": time,
                        "bot": reply["bot"],
                        "sentiment": reply["sentiment"],
                    }
                )
        return sorted(reply_list, key=lambda x: x["time"])

    # Metrics for Bundle Bot
    def get_bundle_launch_transfer_dummy(self) -> int:
        """Method to get the launch bundle dummy variable"""
        return int(len(self.launch_bundle["bundle_launch"]) > 0)

    def get_bundle_creator_buy_dummy(self) -> int:
        """Method to get the bundle creator buy dummy variable"""
        return int(len(self.launch_bundle["bundle_creator_buy"]) > 0)

    def get_bundle_launch_buy_sell_num(self) -> tuple[int, int, int]:
        """Method to get the number of bundle buys"""
        bundle_launch = 0
        bundle_buy = 0
        bunle_sell = 0

        for block, bundle_info in self.launch_bundle["bundle"].items():
            if block == self.launch_block:
                if (
                    len(
                        [
                            row["maker"]
                            for row in bundle_info
                            if row["maker"] != self.creator
                        ]
                    )
                    > 0
                ):
                    bundle_launch += 1
            else:
                bundle_length = len(bundle_info)
                if (
                    len(
                        [
                            row["acts"][0]["typ"]
                            for row in bundle_info
                            if row["acts"][0]["typ"] == "Buy"
                        ]
                    )
                    == bundle_length
                ):
                    bundle_buy += 1

                elif (
                    len(
                        [
                            row["acts"][0]["typ"]
                            for row in bundle_info
                            if row["acts"][0]["typ"] == "Sell"
                        ]
                    )
                    == bundle_length
                ):
                    bunle_sell += 1

        return bundle_launch, bundle_buy, bunle_sell

    # Metrics for Comment Bot
    def get_comment_bot_num(self) -> int:
        """Method to get the number of bot comments"""
        return len([comment for comment in self.comment_list if comment["bot"]])

    def get_positive_comment_bot_num(self) -> int:
        """Method to get the number of positive comments"""
        return len(
            [
                comment
                for comment in self.comment_list
                if comment["bot"] and comment["sentiment"] == "positive"
            ]
        )

    def get_negative_comment_bot_num(self) -> int:
        """Method to get the number of negative comments"""
        return len(
            [
                comment
                for comment in self.comment_list
                if comment["bot"] and comment["sentiment"] == "negative"
            ]
        )

    def get_unqiue_repliers(self) -> int:
        """Method to get the unique repliers of the meme token"""
        return len(set([comment["replier"] for comment in self.comment_list]))

    def get_reply_interval_herf(self) -> float:
        """Method to get the Herfindahl index of the meme token"""

        interval_list = []

        for idx, comment_info in enumerate(self.comment_list):
            if idx == 0:
                previous_reply_time = comment_info["time"]
            else:
                interval_list.append(
                    (comment_info["time"] - previous_reply_time).total_seconds()
                )
        return compute_herfindahl(interval_list)

    def get_non_swapper_replier_num(self) -> int:
        """Method to get the number of non-swapper repliers of the meme token"""
        non_swapper_repliers = set(
            [
                comment["replier"]
                for comment in self.comment_list
                if comment["replier"] not in self.swappers
            ]
        )
        return len(non_swapper_repliers)

    # Metrics for Volume Bot
    def get_max_same_txn_per_swaper(self) -> float:
        """Method to get the max number of same transaction per swaper"""

        same_txn_list = []
        for swaper, swaps in self.swappers.items():
            txn_amount_list = [
                swap["acts"][list(swap["acts"].keys())[-1]].base for swap in swaps
            ]
            same_txn_list.append(Counter(txn_amount_list).most_common(1)[0][1])

        return max(same_txn_list) if same_txn_list else 0

    def get_pos_to_number_of_swaps_ratio(self) -> float:
        """Method to get the ratio of the number of positions to the number of swaps"""

        pos_to_number_of_swaps_dict = defaultdict(list)
        for swap in self.get_acts(Swap):
            last_act = swap["acts"][list(swap["acts"].keys())[-1]]
            if last_act.typ == "Buy":
                pos_to_number_of_swaps_dict[swap["maker"]].append(last_act.base)
            else:
                pos_to_number_of_swaps_dict[swap["maker"]].append(-last_act.base)
        pos_to_number_of_swaps = {
            k: (len(v) / (np.abs(sum(v)) + 1))
            for k, v in pos_to_number_of_swaps_dict.items()
        }
        return np.mean(list(pos_to_number_of_swaps.values()))

    def get_non_swap_transfer_amount(self) -> int:
        """Method to get the unique non-swap transfers of the meme token"""

        transfer_amount = 0
        for transfer in self.transfer:
            if transfer.txn_hash in self.non_swap_transfers:
                transfer_amount += transfer.value

        return transfer_amount

    def get_holdings_herf(self) -> float:
        """Method to get the Herfindahl index of the holdings of the meme token"""
        holdings = defaultdict(float)

        for swap in [
            s
            for s in self.get_acts(Swap)
            if s["date"].replace(tzinfo=timezone.utc) < self.migrate_time
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
                transfer.date.replace(tzinfo=timezone.utc) < self.migrate_time
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
            self.prc_date_df.index >= self.migrate_time
        ].copy()
        prc_resampled.index = (prc_resampled.index - self.migrate_time).total_seconds()

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

    # Dpendent Variables

    def get_survive(self, txn_num: int = 10) -> int:
        """Method to get the seconds since the migration time to the last N swaps"""
        swaps = self.get_acts(Swap)
        if len(swaps) < txn_num:
            return 0

        last_act_time = swaps[-txn_num]["date"].replace(tzinfo=timezone.utc)
        return int((last_act_time - self.migrate_time).total_seconds())

    def check_death(self, freq: str, before: Optional[int], txn_num: int = 10) -> int:
        """Method to check if the meme token is dead"""

        if before is not None:
            match freq:
                case "1min":
                    before = before * 60
                case "1h":
                    before = before * 3600

        swaps = self.get_acts(Swap)
        if len(swaps) < txn_num:
            return 1
        last_act_time = swaps[-txn_num]["date"].replace(tzinfo=timezone.utc)
        if last_act_time > self.migrate_time + datetime.timedelta(seconds=before):
            return 0
        else:
            return 1

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

    lst = []

    NUM_OF_OBSERVATIONS = 20

    for chain in [
        # "pumpfun",
        "raydium",
    ]:
        for pool in import_pool(
            chain,
            NUM_OF_OBSERVATIONS,
        ):
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
            print(
                # f"Pool: {pool['token_address']}, "
                f"Max Drawdown: {meme.get_mdd(freq="1min", before=1) * 100:.2f}%, "
                f"Survive: {meme.get_survive()}, "
                f"Death 1min: {meme.check_death(freq='1min', before=1)}, "
                # f"Unique Swapers: {meme.get_unique_swapers()},"
                # f"Unique Non-Swap Transfers: {len(meme.non_swap_transfer_hash)}, ",
                # # Bundle Bot Metrics
                # f"Bundle Launch Transfer Dummy: {meme.get_bundle_launch_transfer_dummy()}, ",
                # f"Bundle Creator Buy Dummy: {meme.get_bundle_creator_buy_dummy()}, ",
                # f"Bundle Launch Buy Sell Num: {meme.get_bundle_launch_buy_sell_num()}",
                # f"Holdings Herfindal Index: {meme.get_holdings_herf()}",
                # # f"Non-Swap Transfer Graph Out Herfindahl: {out_deg}, "
                # # f"Non-Swap Transfer Graph In Herfindahl: {in_deg}",
                # f"Degree: {meme.analyze_non_swap_transfer_graph()}",
                # f"Transfer amount: {meme.get_non_swap_transfer_amount()}",
                # # Dev Metrics
                # f"Dev Buy: {meme.dev_buy}, ",
                # f"Dev Sell: {meme.dev_sell}, ",
                # f"Dev Transfer: {meme.dev_transfer}",
                # f"Dev Transfer Amount: {meme.dev_transfer_amount}",
                # f"Return: {meme.get_ret_before(freq='1min', before=1) * 100:.2f}%, ",
                # f"Max Same Txn per Swaper: {meme.get_max_same_txn_per_swaper():.2f}, ",
                # f"Block Bundle Herfindahl: {meme.get_block_bundle_herf()}",
                # f"Pos to Number of Swaps Ratio: {meme.get_pos_to_number_of_swaps_ratio()}",
                # # Comment Bot Metrics
                # f"Bot Comment Num: {meme.get_comment_bot_num()}, ",
                # f"Positive Bot Comment Num: {meme.get_positive_comment_bot_num()}, ",
                # f"Negative Bot Comment Num: {meme.get_negative_comment_bot_num()}, ",
                # f"Unique Replies: {len(meme.reply_list)}, ",
                # f"Reply Interval Herfindahl: {meme.get_reply_interval_herf()}",
                # f"Unique Repliers: {meme.get_unqiue_repliers()}, ",
                # f"Non-Swapper Repliers: {meme.get_non_swapper_replier_num()}",
            )
            # (meme.get_ret(freq="1min") + 1).cumprod().plot()
            # print(meme.check_death(freq="1h", before=10))
