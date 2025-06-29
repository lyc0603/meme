"""Class to analyze meme traders"""

from tqdm import tqdm

from environ.constants import SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool, Swap, Transfer
from environ.meme_base import MemeBase
from environ.sol_fetcher import import_pool


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
        # self.profit -= swap.usd if swap.usd else 0.0

    def sell(self, swap) -> None:
        """Method to handle sell transactions"""
        self.balance -= swap.base if swap.base else 0.0
        # self.profit += swap.usd if swap.usd else 0.0


class TraderAnalyzer(MemeBase):
    """Class to analyze meme traders"""

    def __init__(self, new_token_pool: NewTokenPool):
        super().__init__(new_token_pool)
        self.traders = self._load_swaps()
        self._merge_traders()
        self._calculate_profit()

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

    def _calculate_profit(self) -> None:
        """Method to calculate the profit of the traders"""
        for trader_add, trader in self.traders.items():
            # get the list of buy and sell swaps
            buy_swaps = [swap for swap in trader.swaps if swap.typ == "Buy"]
            sell_swaps = [swap for swap in trader.swaps if swap.typ == "Sell"]

            # get the use and amount of the swaps
            if len(sell_swaps) > 0:
                buy_amount = sum(
                    swap.base for swap in buy_swaps if swap.base is not None
                )
                sell_amount = sum(
                    swap.base for swap in sell_swaps if swap.base is not None
                )
                buy_usd = sum(swap.usd for swap in buy_swaps if swap.usd is not None)
                sell_usd = sum(swap.usd for swap in sell_swaps if swap.usd is not None)

                # calculate the average realized profit based on the sell
                trader.profit = (
                    ((sell_usd / sell_amount) - (buy_usd / buy_amount)) * sell_amount
                    if buy_amount > 0
                    else 0.0
                )

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


if __name__ == "__main__":
    NUM_OF_OBSERVATIONS = 10

    creator_profit = []

    for chain in [
        "raydium",
    ]:
        for pool in tqdm(
            import_pool(
                chain,
                NUM_OF_OBSERVATIONS,
            )
        ):
            meme = TraderAnalyzer(
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

            # print({k: v.profit for k, v in meme.traders.items() if v.creator})
            # collect creator profits
            creator_profit.append(
                {k: v.profit for k, v in meme.traders.items() if v.creator}
            )

    # plot creator profits
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.title("Creator Profit Distribution")
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    profits = [list(p.values())[0] for p in creator_profit if p]
    plt.hist(profits, bins=50, color="blue", alpha=0.7)
    plt.axvline(np.mean(profits), color="red", linestyle="dashed", linewidth=1)
    plt.axvline(np.median(profits), color="green", linestyle="dashed", linewidth=1)
    plt.legend(["Mean", "Median"])
    plt.show()
