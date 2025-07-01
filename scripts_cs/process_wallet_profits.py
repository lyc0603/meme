"""Script to process wallet data"""

import pandas as pd
from tqdm import tqdm
import multiprocessing

from environ.constants import PROCESSED_DATA_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool, Swap, Transfer
from environ.meme_base import MemeBase
from environ.sol_fetcher import import_pool

chain = "raydium"
x_var_list = ["creator"]
y_var = "profit"


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
        self.non_swap_transfers: list[Transfer] = []
        self.profits: list[float] = []

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

    def sell(self, swap) -> None:
        """Method to handle sell transactions"""
        self.balance -= swap.base if swap.base else 0.0


class TraderAnalyzer(MemeBase):
    """Class to analyze meme traders"""

    def __init__(self, new_token_pool: NewTokenPool, after_block: int):
        super().__init__(new_token_pool)
        self.traders = self._load_swaps(after_block)
        self._merge_traders()
        self._calculate_profit()

    def _load_swaps(self, after_block: int) -> dict[tuple[str, ...], Trader]:
        """Method to load the profit of the traders"""
        traders = {}
        for swapper, swaps_list in self.swappers.items():
            if swaps_list[0]["block"] < after_block:
                continue
            trader = Trader((swapper,), self.pool_add)
            trader.creator = swapper == self.creator
            sorted_swaps = sorted(swaps_list, key=lambda x: x["block"])
            for swap in sorted_swaps:
                if swap["block"] >= after_block:
                    trader.swap(swap["acts"][0])
            traders[(swapper,)] = trader
        return traders

    def _calculate_profit(self) -> None:
        """Method to calculate the profit of the traders"""
        for trader_add, trader in self.traders.items():

            swaps = sorted(trader.swaps, key=lambda x: x.block)
            swap_list = []
            for swap in swaps:
                swap_list.append(swap)

                # get the list of buy and sell swaps
                buy_swaps = [swap for swap in swap_list if swap.typ == "Buy"]
                sell_swaps = [swap for swap in swap_list if swap.typ == "Sell"]

                # get the use and amount of the swaps
                if (len(sell_swaps)) > 0 & (len(buy_swaps) > 0):
                    buy_amount = sum(
                        swap.base for swap in buy_swaps if swap.base is not None
                    )
                    sell_amount = sum(
                        swap.base for swap in sell_swaps if swap.base is not None
                    )
                    buy_usd = sum(
                        swap.usd for swap in buy_swaps if swap.usd is not None
                    )
                    sell_usd = sum(
                        swap.usd for swap in sell_swaps if swap.usd is not None
                    )

                    # calculate the average realized profit based on the sell
                    trader.profit = (
                        ((sell_usd / sell_amount) - (buy_usd / buy_amount))
                        * sell_amount
                        if buy_amount > 0
                        else 0.0
                    )
                    trader.profits.append(trader.profit)
                else:
                    trader.profits.append(0.0)

            self.traders[trader_add] = trader

    def _merge_traders(self) -> None:
        """Method to merge traders from another TraderAnalyzer"""
        sol_launch_transfers = (
            [
                transfer
                for _, bundle_info in self.launch_bundle["bundle_launch"].items()
                for transfer in bundle_info["transfer"]
            ]
            if self.launch_bundle
            else []
        )
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
            self.non_swap_transfers + sol_launch_transfers + sol_bundle_creator_buy
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

    def search_trader(self, address: str) -> tuple[tuple[str, ...], Trader]:
        """Method to search for a trader by address"""
        for k, v in self.traders.items():
            if address in k:
                return k, v

        self.traders[(address,)] = Trader((address,), self.pool_add)
        return (address,), self.traders[(address,)]


def producer(ret_mdd_tab, task_queue, num_workers):
    """Producer function to put token information into the task queue."""
    for idx, token_info in ret_mdd_tab.iterrows():
        task_queue.put(token_info)
    # Add sentinels for consumers to exit
    for _ in range(num_workers):
        task_queue.put(None)


def consumer(task_queue, result_queue):
    """Consumer function to process token information and calculate profit."""
    while True:
        token_info = task_queue.get()
        if token_info is None:
            break
        token_address = token_info["token_address"]
        meme = TraderAnalyzer(
            NewTokenPool(
                token0=SOL_TOKEN_ADDRESS,
                token1=token_address,
                fee=0,
                pool_add=token_address,
                block_number=0,
                chain=chain,
                base_token=token_address,
                quote_token=SOL_TOKEN_ADDRESS,
                txns={},
            ),
            after_block=token_info["migration_block"],
        )
        rows = []
        for trader in meme.traders.values():
            row = {
                "token_address": token_info["token_address"],
                "migration_block": token_info["migration_block"],
                "wallet_address": trader.address,
                "creator": 1 if trader.creator else 0,
                "profits": sorted(
                    list(zip([_["block"] for _ in trader.swaps], trader.profits)),
                    key=lambda x: x[0],
                ),
            }

            rows.append(row)
        result_queue.put(rows)


if __name__ == "__main__":

    ret_mdd_tab = pd.read_csv(f"{PROCESSED_DATA_PATH}/ret_ma.csv")

    num_workers = 25
    task_queue = multiprocessing.Queue(maxsize=num_workers * 2)
    result_queue = multiprocessing.Queue()

    consumers = [
        multiprocessing.Process(target=consumer, args=(task_queue, result_queue))
        for _ in range(num_workers)
    ]
    for c in consumers:
        c.start()

    prod = multiprocessing.Process(
        target=producer, args=(ret_mdd_tab, task_queue, num_workers)
    )
    prod.start()

    reg_rows = []
    total = len(ret_mdd_tab)
    with tqdm(total=total, desc="Processing tokens") as pbar:
        finished = 0
        while finished < total:
            result = result_queue.get()
            reg_rows.extend(result)
            finished += 1
            pbar.update(1)

    prod.join()
    for c in consumers:
        c.join()

    # reg_tab = pd.DataFrame(reg_rows)
    import pickle

    with open(f"{PROCESSED_DATA_PATH}/trader_analyzer.pkl", "wb") as f:
        pickle.dump(reg_rows, f)
