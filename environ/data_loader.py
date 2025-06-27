"""Class to analyze meme token"""

import datetime
from datetime import timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tenacity import retry, stop_after_attempt

from environ.constants import FIGURE_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool, Swap
from environ.meme_analyzer import MemeAnalyzer
from environ.sol_fetcher import import_pool

IMAGE_URL_TEMP = "https://raw.githubusercontent.com/lyc0603/meme/\
refs/heads/main/figures/candle/{ca}.png"


class DataLoader(MemeAnalyzer):
    """Data Loader for the multi-agent system."""

    def __init__(self, *args, **kwargs):
        """Initialize the DataLoader."""
        super().__init__(*args, **kwargs)
        self.candle_url = IMAGE_URL_TEMP.format(ca=self.new_token_pool.pool_add)
        self.swap_list = self.build_txn_list()
        self.comment_list = [
            _["comment"]["user"][:6] + ": " + _["comment"]["text"]
            for _ in self.comment
            if datetime.datetime.fromtimestamp(
                _["comment"]["timestamp"] / 1000, tz=timezone.utc
            )
            <= self.migrate_time
        ]

    def build_txn_list(self) -> list:
        """Get transactions for the meme token."""
        wallet_dict = {
            self.creator: "Creator",
        }
        swap_list = []
        funded_wallet = set()
        for bundle in ["bundle_launch", "bundle_creator_buy"]:
            funded_wallet = funded_wallet.union(
                set(
                    [
                        add
                        for k, v in self.launch_bundle[bundle].items()
                        for tsf in v["transfer"]
                        for add in (tsf["from_"], tsf["to"])
                    ]
                )
            )
        funded_wallet = funded_wallet - set([self.creator])
        swap_list.append(
            (
                self.launch_block,
                f"Creator creates the meme coin at block {self.launch_block}",
            )
        )

        for swap in self.get_acts(Swap):

            if swap["maker"] not in wallet_dict:
                wallet_dict[swap["maker"]] = f"Trader {len(wallet_dict)}"

            if swap["maker"] in funded_wallet:
                funding_string = " (bundle funded by creator)"
            else:
                funding_string = ""

            if swap["acts"][0]["block"] >= self.migrate_block:
                continue

            if swap["acts"][0]["usd"] is None:
                base = "0.00"
                quote = "0.000"
            else:
                base = swap["acts"][0]["base"]
                if float(base) >= 1e9:
                    base = f"{base / 1e9:.2f}B"
                elif float(base) >= 1e6:
                    base = f"{base / 1e6:.2f}M"
                elif float(base) >= 1e3:
                    base = f"{base / 1e3:.2f}K"
                else:
                    base = f"{base:.2f}"
                quote = f"{swap['acts'][0]['quote']:.3f}"
            if swap["acts"][0]["typ"] == "Buy":
                swap_list.append(
                    (
                        swap["acts"][0]["block"],
                        f"{wallet_dict[swap['maker']]}{funding_string} buy {base} meme coin for {quote} SOL at block {swap['acts'][0]['block']}",
                    )
                )
            else:
                swap_list.append(
                    (
                        swap["acts"][0]["block"],
                        f"{wallet_dict[swap['maker']]}{funding_string} sell {base} meme coin for {quote} SOL at block {swap['acts'][0]['block']}",
                    )
                )

        return [i for _, i in sorted(swap_list, key=lambda x: x[0])]

    @retry(stop=stop_after_attempt(3))
    def plot_pre_migration_candlestick_plotly(self, freq: str = "5min") -> None:
        """
        Plot a Plotly-based candlestick chart for **pre-migration** meme token price.

        Args:
            freq (str): Resampling frequency for the pre-migration data (e.g., '5min', '15min').
        """
        df = self.pre_prc_date_df.copy()
        df = df.resample(freq).agg(
            {"price": ["first", "max", "min", "last"], "quote": "sum"}
        )
        df.columns = ["open", "high", "low", "close", "volume"]
        df.dropna(inplace=True)
        # except for the first row, the open price is the same as the previous close price
        df.loc[df.index[1:], "open"] = df["close"].iloc[:-1].values

        # Recompute high and low to ensure consistency if needed
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_width=[0.2, 0.8],
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker=dict(color="grey"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=400,
            width=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"rangeslider": {"visible": False}},
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_yaxes(row=1, col=1, tickformat=".1e")
        fig.update_yaxes(row=2, col=1)
        fig.update_xaxes(
            range=[
                df.index.min() - pd.Timedelta(minutes=0.5),
                df.index.max() + pd.Timedelta(minutes=0.5),
            ]
        )
        # fig.write_image(
        #     FIGURE_PATH / "candle" / f"{self.new_token_pool.token1}.png",
        #     scale=2,
        # )
        fig.show()


if __name__ == "__main__":

    import os

    from tqdm import tqdm

    os.makedirs(FIGURE_PATH / "candle", exist_ok=True)

    lst = []

    NUM_OF_OBSERVATIONS = 1000

    txn_num_swap = {}

    for chain in [
        "raydium",
    ]:
        # for pool in tqdm(
        #     import_pool(
        #         chain,
        #         NUM_OF_OBSERVATIONS,
        #     )
        # ):
        pool = {
            "token_address": "figures/candle/6QXbUVM4KmpLuFvUWP3mrTqVB4nn1idfK1PdXQaYqycv.png".split(
                "/"
            )[
                -1
            ].split(
                "."
            )[
                0
            ],
        }
        meme = DataLoader(
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

        txn_num_swap[pool["token_address"]] = len(meme.swap_list)

    meme.plot_pre_migration_candlestick_plotly(freq="1min")
    (meme.get_ret(freq="1min") + 1).cumprod().plot()
    freq = "1min"
    before = 40
    print(
        f"Max Drawdown: {meme.get_mdd(freq=freq, before=before) * 100:.2f}%, "
        f"Return: {meme.get_ret_before(freq=freq, before=before) * 100:.2f}%, "
        f"Survive: {meme.get_survive()}, "
        # f"Death 1min: {meme.check_death(freq=freq, before=before)}, "
    )
