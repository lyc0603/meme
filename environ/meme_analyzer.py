"""Class to analyze meme token"""

import datetime
from collections import defaultdict
from datetime import timezone
from typing import Optional, Any

import numpy as np
import pandas as pd

from environ.constants import SOL_TOKEN_ADDRESS, PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool, Swap, Transfer
from environ.meme_base import MemeBase
from environ.sol_fetcher import import_pool

INITIAL_PRICE = 2.8e-8
MAX_INACTIVITY = pd.Timedelta(days=30)
UPPER_BOUND = 5000

trader_t = pd.read_csv(PROCESSED_DATA_PATH / "trader_t_stats.csv")
trader_t = trader_t.loc[trader_t["meme_num"] <= 1000].dropna(subset=["t_stat"])
winner = set(trader_t.loc[trader_t["t_stat"] > 2.576, "trader_address"].unique())
loser = set(trader_t.loc[trader_t["t_stat"] < -2.576, "trader_address"].unique())
neutral = set(
    trader_t.loc[
        trader_t["t_stat"].abs() <= 2.576,
        "trader_address",
    ].unique()
)


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
        self.winner: bool = False
        self.loser: bool = False
        self.neutral: bool = False
        self.sniper: bool = False
        self.balance: float = 0.0
        self.profit: float = 0.0
        self.swaps: list[Swap] = []
        self.non_swap_transfers: list[Transfer] = []
        self.wash_trading_score: Optional[float] = None
        self.first_trade_time: Optional[datetime.datetime] = None

    def swap(self, swap: Swap, date: datetime.datetime) -> None:
        """Method to handle swap transactions"""
        if swap.typ == "Buy":
            self.buy(swap)
        elif swap.typ == "Sell":
            self.sell(swap)
        self.swaps.append(swap)
        if not self.first_trade_time:
            self.first_trade_time = date.replace(tzinfo=timezone.utc)
        else:
            self.first_trade_time = min(
                self.first_trade_time, date.replace(tzinfo=timezone.utc)
            )

    def buy(self, swap) -> None:
        """Method to handle buy transactions"""
        self.balance += swap.base if swap.base else 0.0
        self.profit -= swap.usd if swap.usd else 0.0

    def sell(self, swap) -> None:
        """Method to handle sell transactions"""
        self.balance -= swap.base if swap.base else 0.0
        self.profit += swap.usd if swap.usd else 0.0

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

        self.wash_trading_score = flip_count / (np.abs(self.balance) + 1)

        return self.wash_trading_score


class MemeAnalyzer(MemeBase):
    """Class to analyze meme token"""

    def __init__(
        self,
        new_token_pool: NewTokenPool,
    ):
        super().__init__(new_token_pool)
        self.bundler = set([self.creator])
        self.prc_date_df = self._build_price_df()
        self.comment_list = self._build_comment_list()
        self.bundle = self._build_bundle()
        self.sniper = self._build_sniper()
        self.launch_bundle, self.bundle_bot_pct = self._build_bundle_bot()

        # analyze traders
        self.traders = self._load_swaps()
        self.non_bot_creator_transfer_traders = set(
            [k[0] for k, v in self.traders.items()]
        )
        self._merge_traders()
        self.volume_bot = False
        self.traders, self.bots = self._wash_trade()

    def check_migrate(self) -> str:
        """Method to check if the meme token has migrated"""

        migration = False

        for _, acts in enumerate(self.get_acts(Swap)):
            last_act = acts["acts"][list(acts["acts"].keys())[-1]]
            if last_act.dex == "Raydium Liquidity Pool V4":
                migration = True
                return migration

        return migration

    def check_max_purchase_pct(self) -> float:
        """Method to check the purchase percentage of the meme token"""
        balance = 0
        balance_list = []
        for _, acts in enumerate(self.get_acts(Swap)):
            last_act = acts["acts"][list(acts["acts"].keys())[-1]]
            if last_act.dex == "pump.fun":
                if last_act.typ == "Buy":
                    balance += last_act.base
                elif last_act.typ == "Sell":
                    balance -= last_act.base
                balance_list.append(balance)
        return max(balance_list) / 793100000

    # Build-in Methods
    def _build_bundle(self) -> dict[str, Any]:
        """Method to build the bundle data"""

        bundle = defaultdict(list)

        for _, acts in enumerate(self.get_acts(Swap)):
            if self.migrate_block:
                if acts["block"] <= self.migrate_block:
                    bundle[acts["block"]].append(acts)
            else:
                bundle[acts["block"]].append(acts)

        return {block: acts for block, acts in bundle.items() if len(acts) > 1}

    def _build_price_df(self) -> pd.DataFrame:
        """Method to build the price DataFrame of the meme token"""
        prc_date_dict = {
            "block": [],
            "date": [],
            "price": [],
            "base": [],
            "quote": [],
            "usd": [],
            "typ": [],
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
                "typ": last_act.typ,
            }.items():
                prc_date_dict[key].append(value)

        prc_date_df = pd.DataFrame(prc_date_dict)
        prc_date_df = prc_date_df.set_index("date").sort_index()
        prc_date_df["price"] = prc_date_df["price"].replace(0, np.nan)
        prc_date_df.dropna(subset=["price"], inplace=True)

        return prc_date_df

    def _build_comment_list(self) -> list[dict[str, Any]]:
        """Method to build the comment dictionary of the meme token"""
        reply_list = []
        for reply in self.comment:
            time = datetime.datetime.fromtimestamp(
                reply["comment"]["timestamp"] / 1000, tz=timezone.utc
            )
            if self.migrate_time:
                if time <= self.migrate_time:
                    reply_list.append(
                        {
                            "replier": reply["comment"]["user"],
                            "time": time,
                            "bot": reply["bot"],
                            "sentiment": reply["sentiment"],
                        }
                    )
            else:
                reply_list.append(
                    {
                        "replier": reply["comment"]["user"],
                        "time": time,
                        "bot": reply["bot"],
                        "sentiment": reply["sentiment"],
                    }
                )
        return sorted(reply_list, key=lambda x: x["time"])

    def _build_sniper(self) -> int:
        """Method to build the sniper bot"""

        sniper_txn = [
            _
            for _ in self.get_acts(Swap)
            if (_["block"] - self.launch_block < 5)
            & (_["block"] != self.launch_block)
            & (_["acts"][0]["typ"] == "Buy")
        ]

        return set([_["maker"] for _ in sniper_txn])

    def _build_bundle_bot(self) -> tuple[int, float]:
        """Method to build the bundle data"""
        bundle_launch = 0
        bundle_bot = 0

        for block, bundle_info in self.bundle.items():
            if block == self.launch_block:
                bundle_maker = [
                    row["maker"] for row in bundle_info if row["maker"] != self.creator
                ]
                self.bundler.update(bundle_maker)
                if len(bundle_maker) > 0:
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
                    bundle_bot += 1

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
                    bundle_bot += 1

        total_blocks = len([_.block for _ in self.txn])

        return bundle_launch, bundle_bot / total_blocks

    # Dependent Variables
    def get_pre_migration_duration(self) -> int:
        """Method to get the pre-migration duration in seconds"""

        if self.migrate_time:
            return int(self.migrate_time.timestamp() - self.launch_time.timestamp())
        else:
            return np.nan

    def get_max_ret_and_pump_duration(self) -> tuple[float, int]:
        """Method to get the maximum return and pump duration in seconds"""

        first_price = self.prc_date_df["price"].iloc[0]
        max_price = self.prc_date_df.loc[
            self.prc_date_df["price"] <= first_price * UPPER_BOUND, "price"
        ].max()

        # Get timestamps of transactions at peak price
        first_price_ts = self.prc_date_df.index[0]
        max_price_ts = self.prc_date_df.loc[
            self.prc_date_df["price"] == self.prc_date_df["price"].max()
        ].index
        max_price_ts = min(max_price_ts)

        pre_max_df = self.prc_date_df.loc[self.prc_date_df.index < max_price_ts].copy()
        pre_max_df["time_diff"] = pre_max_df.index.to_series().diff()

        # Find the first trade after a >1 month inactivity gap
        long_gap = pre_max_df.loc[pre_max_df["time_diff"] > MAX_INACTIVITY]
        if not long_gap.empty:
            last_gap_idx = long_gap.index[-1]
            first_price = pre_max_df.loc[last_gap_idx:].iloc[0]["price"]
            first_price_ts = pre_max_df.loc[last_gap_idx:].index[0]

        # Calculate max return
        max_return = (max_price - first_price) / first_price

        # Calculate pump duration
        pump_duration = int((max_price_ts - first_price_ts).total_seconds())

        return np.log(1 + max_return), pump_duration

    def get_dumper(self) -> tuple[str, int, int, int]:
        """Method to get the address of the dumper"""

        winner_dump, loser_dump, neutral_dump = 0, 0, 0

        max_price = self.prc_date_df["price"].max()
        max_price_ts = min(
            self.prc_date_df.loc[self.prc_date_df["price"] == max_price].index
        ).tz_localize(None)

        first_ten_sell = []
        creator_txn = []
        for swap in self.get_acts(Swap):
            if swap["date"] >= max_price_ts:
                if swap["acts"][0]["typ"] == "Sell":
                    if swap["maker"] in self.bundler:
                        creator_txn.append(swap)
                    first_ten_sell.append(swap)
                    if len(first_ten_sell) == 10:
                        break

        if len(creator_txn) != 0:
            creator_base = sum([swap["acts"][0]["base"] for swap in creator_txn])
        else:
            creator_base = 0

        if len(first_ten_sell) != 0:
            largest_sell = max(first_ten_sell, key=lambda x: x["acts"][0]["base"])
        else:
            return "no dumper", 0, 0, 0

        if largest_sell["maker"] in winner:
            winner_dump = 1

        if largest_sell["maker"] in loser:
            loser_dump = 1

        if largest_sell["maker"] in neutral:
            neutral_dump = 1

        if (largest_sell["maker"] in self.bundler) | (
            creator_base >= largest_sell["acts"][0]["base"]
        ):
            if len(winner & self.bundler) > 0:
                return "creator", 1, 0, 0
            return "creator", winner_dump, loser_dump, neutral_dump
        elif largest_sell["maker"] in self.sniper:
            return "sniper", winner_dump, loser_dump, neutral_dump
        else:
            return "other", winner_dump, loser_dump, neutral_dump

    def get_dump_duration(self) -> int:
        """Method to get the dump duration in seconds"""

        # calculate the cumulative balance
        balance_df = self.prc_date_df.copy()
        balance_df["balance"] = balance_df.apply(
            lambda row: row["base"] if row["typ"] == "Buy" else -row["base"], axis=1
        )
        balance_df["cum_balance"] = balance_df["balance"].cumsum()

        # Get the maximum price and its timestamp
        max_price = self.prc_date_df["price"].max()
        max_price_ts = min(
            self.prc_date_df.loc[self.prc_date_df["price"] == max_price].index
        )
        max_price_cum_balance = balance_df.loc[
            balance_df.index == max_price_ts, "cum_balance"
        ].values[0]
        post_max_df = balance_df.loc[balance_df.index >= max_price_ts].copy()

        # Calculate the dump threshold as 10% of the maximum cumulative balance
        dump_balance = 0.1 * max_price_cum_balance
        dump_ts = post_max_df.loc[(post_max_df["cum_balance"] < dump_balance)].index

        # Get last trade timestamp
        last_trade_ts = self.prc_date_df.index[-1]

        if dump_ts.empty:
            # return int((last_trade_ts - self.launch_time).total_seconds())
            return int((last_trade_ts - max_price_ts).total_seconds())

        dump_ts = min(dump_ts)

        # Calculate dump duration
        return int((dump_ts - max_price_ts).total_seconds())

    def get_number_of_traders(self) -> int:
        """Method to get the number of traders"""
        # non-bot-transfer trader plus the creator
        return len(self.non_bot_creator_transfer_traders) + 1

    # Metrics for Sniper Bot
    def get_sniper_bot(self) -> int:
        """Method to get the number of sniper bots"""
        return int(len(self.sniper) > 0)

    # Metrics for Bundle Bot
    def get_bundle_launch_buy_sell_num(self) -> tuple[int, float]:
        """Method to get the number of bundle buys and sells"""
        return self.launch_bundle, self.bundle_bot_pct

    # Metrics for Comment Bot
    def get_comment_bot_num(self) -> int:
        """Method to get the number of positive comments"""
        return len([comment for comment in self.comment_list if comment["bot"]])

    # Metrics for Volume Bot
    def get_volume_bot(self) -> bool:
        """Method to check if the meme token is a volume bot"""
        return int(self.volume_bot)

    # trader methods
    def _load_swaps(self):
        """Method to load the profit of the traders"""
        traders = {}
        for swapper, swaps_list in self.swappers.items():
            trader = Trader((swapper,), self.pool_add)
            trader.creator = swapper == self.creator
            trader.winner = swapper in winner
            trader.loser = swapper in loser
            trader.neutral = swapper in neutral
            trader.sniper = swapper in self.sniper
            sorted_swaps = sorted(swaps_list, key=lambda x: x["block"])
            for swap in sorted_swaps:
                trader.swap(swap["acts"][0], swap["date"])
            traders[(swapper,)] = trader
        return traders

    def _merge_traders(self) -> None:
        """Method to merge traders from another TraderAnalyzer"""

        # General launch bundle
        general_launch_transfers = []
        if self.launch_block in self.bundle:
            for _ in self.bundle[self.launch_block]:
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
                    # remove the bot trader
                    self.non_bot_creator_transfer_traders.discard(_["maker"])

        for non_swap_transfer in self.non_swap_transfers + general_launch_transfers:
            self.non_bot_creator_transfer_traders.discard(non_swap_transfer.from_)
            self.non_bot_creator_transfer_traders.discard(non_swap_transfer.to)
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
        bots = {}
        for trader_add, trader in self.traders.items():
            if trader.wash_trading() > 50:
                self.volume_bot = True
                bots[trader_add] = trader
                self.non_bot_creator_transfer_traders.discard(trader_add)
            else:
                traders[trader_add] = trader
        return traders, bots

    def search_trader(self, address: str) -> tuple[tuple[str, ...], Trader]:
        """Method to search for a trader by address"""
        for k, v in self.traders.items():
            if address in k:
                return k, v

        self.traders[(address,)] = Trader((address,), self.pool_add)
        return (address,), self.traders[(address,)]


if __name__ == "__main__":

    lst = []

    NUM_OF_OBSERVATIONS = 1

    for chain in [
        # "pre_trump_pumpfun",
        # "pre_trump_raydium",
        "pumpfun",
        # "raydium",
    ]:
        for pool in import_pool(
            chain,
            NUM_OF_OBSERVATIONS,
        ):
            token_add = pool["token_address"]
            token_add = "AE1A1M3PTWmkuhfyL4pS22eYfxLzRw28aACu4hVMpump"
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
            # bundle_launch, bundle_bot = meme.get_bundle_launch_buy_sell_num()
            print(
                # f"meme coin: {meme.new_token_pool.pool_add}\n",
                # f"pre_migration_duration: {meme.get_pre_migration_duration()} seconds\n",
                # f"max_return and pump duration: {meme.get_max_ret_and_pump_duration()}\n",
                # f"dump duration: {meme.get_dump_duration()} seconds\n",
                # f"get_bundle_launch_buy_sell_num: {meme.get_bundle_launch_buy_sell_num()}\n",
                # f"dumper: {meme.get_dumper()}\n",
                # f"wash_trading: {max([v.wash_trading() for k,v in {**meme.bots, **meme.traders}.items()])}\n",
                # f"bundle_bot: {meme.get_bundle_launch_buy_sell_num()}\n",
                # f"comment_bot_num: {meme.get_comment_bot_num()}\n",
                # f"migrate: {meme.check_migrate()}\n",
                # f"purchase_percentage: {meme.check_max_purchase_pct()}\n",
            )
            # (
            #     meme.get_ret(meme.prc_date_df, meme.migrate_time, True) + 1
            # ).cumprod().plot()

            from environ.utils import handle_first_comment_bot

            comment_rows = handle_first_comment_bot(meme, token_add, meme.launch_time)
