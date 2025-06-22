"""Class to analyze meme token"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


import pandas as pd

from environ.constants import FIGURE_PATH, SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.sol_fetcher import import_pool
from environ.meme_analyzer import MemeAnalyzer
from tenacity import retry, stop_after_attempt


class DataLoader(MemeAnalyzer):
    """Data Loader for the multi-agent system."""

    def __init__(self, *args, **kwargs):
        """Initialize the DataLoader."""
        super().__init__(*args, **kwargs)

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
        fig.write_image(
            FIGURE_PATH / "candle" / f"{self.new_token_pool.token1}.png",
            scale=2,
        )
        # fig.show()


if __name__ == "__main__":

    import os
    from tqdm import tqdm

    os.makedirs(FIGURE_PATH / "candle", exist_ok=True)

    lst = []

    NUM_OF_OBSERVATIONS = 1000

    for chain in [
        "raydium",
    ]:
        #     for pool in tqdm(
        #         import_pool(
        #             chain,
        #             NUM_OF_OBSERVATIONS,
        #         )
        #     ):
        pool = {
            "token_address": "2YgruGy8W3XKr1RZUAZN3j5fF54hALNw27BEgfcGpump",
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
        meme.plot_pre_migration_candlestick_plotly(freq="1min")
