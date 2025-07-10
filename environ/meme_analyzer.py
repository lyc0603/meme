"""Class to analyze meme token"""

import datetime
from collections import defaultdict
from datetime import timezone
from typing import Optional, Any
from collections import Counter


import numpy as np
import pandas as pd

from environ.constants import SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool, Swap, Transfer
from environ.meme_base import MemeBase
from environ.sol_fetcher import import_pool

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


class Account:
    """Class to store the account data"""

    def __init__(self, address: tuple[str, ...]):
        self.address = address


class Trader(Account):
    """Class to store the trader data"""

    def __init__(self, address: tuple[str, ...], pool_add: str):
        super().__init__(address)
        self.token = pool_add
        self.creator: bool = False
        self.balance: float = 0.0
        self.profit: float = 0.0
        self.swaps: list[Swap] = []
        self.volume: float = 0.0
        self.non_swap_transfers: list[Transfer] = []

    def swap(self, swap: Swap) -> None:
        """Method to handle swap transactions"""
        if swap.typ == "Buy":
            self.buy(swap)
        elif swap.typ == "Sell":
            self.sell(swap)
        self.swaps.append(swap)

    def buy(self, swap) -> None:
        """Method to handle buy transactions"""
        self.balance += swap.base if swap.base else 0.0
        self.profit -= swap.usd if swap.usd else 0.0
        self.volume += swap.usd if swap.usd else 0.0

    def sell(self, swap) -> None:
        """Method to handle sell transactions"""
        self.balance -= swap.base if swap.base else 0.0
        self.profit += swap.usd if swap.usd else 0.0
        self.volume += swap.usd if swap.usd else 0.0

    def wash_trading(self) -> None:
        """Method to check the volume of the trader"""

        txn_amount_list = [swap.base for swap in self.swaps]
        if not txn_amount_list:
            return 0.0

        flip_count = sum(
            1
            for i in range(1, len(self.swaps))
            if (self.swaps[i].typ != self.swaps[i - 1].typ)
            & (self.swaps[i].base == self.swaps[i - 1].base)
        )

        return flip_count / (np.abs(self.balance) + 1)


class MemeAnalyzer(MemeBase):
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        super().__init__(new_token_pool)
        self.trading_volume = 0
        self.wash_trading_volume = 0
        self.prc_date_df, self.pre_prc_date_df = self._build_price_df()
        self.comment_list = self._build_comment_list()

        # analyze traders
        self.traders = self._load_swaps()
        self._merge_traders()
        self.trader_number = len(self.traders)
        self.volume_bot = False
        self.traders = self._wash_trade()

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

    # Dependent Variables
    def get_pre_migration_duration(self) -> int:
        """Method to get the pre-migration duration in seconds"""
        return int(self.migrate_time.timestamp() - self.launch_time.timestamp())

    def get_max_ret_and_pump_duration(self) -> tuple[float, int]:
        """Method to get the maximum return and pump duration in seconds"""
        pre_price = self.pre_prc_date_df["price"].iloc[-1]
        max_price = self.prc_date_df["price"].max()

        # Calculate max return
        max_return = (max_price - pre_price) / pre_price if pre_price else 0.0

        # Get timestamps of transactions at peak price
        max_price_ts = self.prc_date_df.loc[
            self.prc_date_df["price"] == max_price
        ].index

        # Calculate pump duration
        pump_duration = int((min(max_price_ts) - self.migrate_time).total_seconds())

        return np.log(1 + max_return), pump_duration

    def get_dump_duration(self) -> int:
        """Method to get the dump duration in seconds"""
        pre_price = self.pre_prc_date_df["price"].iloc[-1]
        min_price = 0.1 * pre_price

        # Get timestamps of transactions below 90% of pre-migration price
        dump_ts = self.prc_date_df.loc[self.prc_date_df["price"] < min_price].index
        if dump_ts.empty:
            return 12 * 3600  # Default to 12 hours if no dumps found
        # Calculate dump duration
        dump_duration = int((min(dump_ts) - self.migrate_time).total_seconds())

        return dump_duration

    def get_pre_migration_volatility(self) -> float:
        """Method to get the pre-migration volatility of the meme token"""
        pre_ret = self.get_ret(
            df_prc=self.pre_prc_date_df,
            ts=self.launch_time,
            append=False,
        )

        # Calculate standard deviation of log returns
        return np.log(pre_ret["ret"] + 1).std()

    def get_post_migration_volatility(self) -> float:
        """Method to get the post-migration volatility of the meme token"""
        post_ret = self.get_ret(
            df_prc=self.prc_date_df,
            ts=self.migrate_time,
            append=True,
        )

        # Calculate standard deviation of returns
        return np.log(post_ret["ret"] + 1).std()

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

    # Metrics for Volume Bot
    def get_volume_bot(self) -> bool:
        """Method to check if the meme token is a volume bot"""
        return int(self.volume_bot)

    def get_wash_trading_volume_frac(self) -> float:
        """Method to get the fraction of wash trading volume"""
        return (
            self.wash_trading_volume / self.trading_volume
            if self.trading_volume
            else 0.0
        )

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

    # Price & Return Processing Methods
    def resample_price(
        self, df_prc: pd.DataFrame, ts: datetime.datetime
    ) -> pd.DataFrame:
        """Method to resample the price data to the specified frequency"""
        # convert the index to how many seconds since the timestamp
        prc_resampled = df_prc.loc[df_prc.index >= ts].copy()
        prc_resampled.index = (prc_resampled.index - ts).total_seconds()

        # drop the duplicate index values
        prc_resampled = prc_resampled[~prc_resampled.index.duplicated(keep="last")]

        # prc_resampled = prc_resampled.reindex(
        #     range(0, int(prc_resampled.index.max()) + 1)
        # ).ffill()

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

    def get_ret(
        self,
        df_prc: pd.DataFrame,
        ts: datetime.datetime,
        append: bool,
    ) -> pd.DataFrame:
        """Method to get the return of the meme token"""

        df_prc.dropna(subset=["price"], inplace=True)
        prc_resampled = self.resample_price(df_prc.loc[df_prc.index >= ts], ts)
        if append:
            prc_resampled = self.append_pre_prc_date_df(prc_resampled)

        return (
            prc_resampled.copy()
            .pct_change()
            .dropna()
            .rename(columns={"price": "ret"})[["ret"]]
        )

    # trader methods
    def _load_swaps(self):
        """Method to load the profit of the traders"""
        traders = {}
        for swapper, swaps_list in self.swappers.items():
            trader = Trader((swapper,), self.pool_add)
            trader.creator = swapper == self.creator
            sorted_swaps = sorted(swaps_list, key=lambda x: x["block"])
            for swap in sorted_swaps:
                trader.swap(swap["acts"][0])
            traders[(swapper,)] = trader
        return traders

    def _merge_traders(self) -> None:
        """Method to merge traders from another TraderAnalyzer"""
        # General launch bundle
        general_launch_transfers = []
        if self.launch_block in self.launch_bundle["bundle"]:
            for _ in self.launch_bundle["bundle"][self.launch_block]:
                if _["maker"] != self.creator:
                    general_launch_transfers.append(
                        Transfer(
                            block=_["block"],
                            log_index=0,
                            from_=_["maker"],
                            to=self.creator,
                            value=_["acts"][0]["base"],
                            date=_["date"],
                            txn_hash=_["txn_hash"],
                        )
                    )

        # Creator-funded launch bundle
        sol_launch_transfers = (
            [
                transfer
                for _, bundle_info in self.launch_bundle["bundle_launch"].items()
                for transfer in bundle_info["transfer"]
            ]
            if self.launch_bundle
            else []
        )
        # Creator-funded buy bundle
        sol_bundle_creator_buy = (
            [
                transfer
                for _, bundle_info in self.launch_bundle["bundle_creator_buy"].items()
                for transfer in bundle_info["transfer"]
            ]
            if self.launch_bundle
            else []
        )
        for non_swap_transfer in (
            self.non_swap_transfers
            + general_launch_transfers
            + sol_launch_transfers
            + sol_bundle_creator_buy
        ):
            from_trader_address, from_trader = self.search_trader(
                non_swap_transfer.from_
            )
            to_trader_address, to_trader = self.search_trader(non_swap_transfer.to)

            # if the traders are the same, skip
            if from_trader_address == to_trader_address:
                continue

            merged_trader = Trader(
                from_trader.address + to_trader_address, self.pool_add
            )
            merged_trader.creator = from_trader.creator or to_trader.creator
            merged_trader.balance = from_trader.balance + to_trader.balance
            merged_trader.profit = from_trader.profit + to_trader.profit
            merged_trader.swaps = from_trader.swaps + to_trader.swaps
            merged_trader.non_swap_transfers = (
                from_trader.non_swap_transfers + to_trader.non_swap_transfers
            )
            # remove the original traders
            for address in [from_trader_address, to_trader_address]:
                del self.traders[address]

            # add the merged trader
            self.traders[merged_trader.address] = merged_trader

    def _wash_trade(self) -> float:
        """Method to get the wash trading volume of the meme token"""
        traders = {}
        for trader_add, trader in self.traders.items():
            self.trading_volume += trader.volume
            if trader.wash_trading() > 50:
                self.volume_bot = True
                self.wash_trading_volume += trader.volume
            else:
                traders[trader_add] = trader
        return traders

    def search_trader(self, address: str) -> tuple[tuple[str, ...], Trader]:
        """Method to search for a trader by address"""
        for k, v in self.traders.items():
            if address in k:
                return k, v

        self.traders[(address,)] = Trader((address,), self.pool_add)
        return (address,), self.traders[(address,)]


if __name__ == "__main__":

    lst = []

    NUM_OF_OBSERVATIONS = 10

    for chain in [
        # "pumpfun",
        "raydium",
    ]:
        for pool in import_pool(
            chain,
            NUM_OF_OBSERVATIONS,
        ):
            token_add = pool["token_address"]
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

            print(
                f"meme coin: {meme.new_token_pool.pool_add}\n",
                # f"pre_migration_duration: {meme.get_pre_migration_duration()} seconds\n",
                f"max_return and pump duration: {meme.get_max_ret_and_pump_duration()}\n",
                f"dump duration: {meme.get_dump_duration()} seconds\n",
                # f"pre_migration_volatility: {meme.get_pre_migration_volatility()}\n",
                # f"post_migration_volatility: {meme.get_post_migration_volatility()}\n",
                f"wash_trading: {max([v.wash_trading() for k,v in meme.traders.items()])}\n",
            )
            (
                meme.get_ret(meme.prc_date_df, meme.migrate_time, True) + 1
            ).cumprod().plot()

            break
