"""Script to produce CoT data"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.constants import FIGURE_PATH, SOL_TOKEN_ADDRESS
from typing import Optional


def get_cutoff_date(prc_date_df: pd.DataFrame):
    """Function to get cut-off date"""
    # calculate the cumulative balance
    balance_df = prc_date_df.copy()
    balance_df["balance"] = balance_df.apply(
        lambda row: row["base"] if row["typ"] == "Buy" else -row["base"], axis=1
    )
    balance_df["cum_balance"] = balance_df["balance"].cumsum()

    # Get the maximum price and its timestamp
    max_price = prc_date_df["price"].max()
    max_price_ts = min(prc_date_df.loc[prc_date_df["price"] == max_price].index)
    max_price_cum_balance = balance_df.loc[
        balance_df.index == max_price_ts, "cum_balance"
    ].values[0]
    post_max_df = balance_df.loc[balance_df.index >= max_price_ts].copy()

    # Calculate the dump threshold as 10% of the maximum cumulative balance
    dump_balance = 0.1 * max_price_cum_balance
    dump_ts = post_max_df.loc[(post_max_df["cum_balance"] < dump_balance)].index

    # Get last trade timestamp
    last_trade_ts = prc_date_df.index[-1]
    cutoff_date = last_trade_ts if dump_ts.empty else min(dump_ts)

    return cutoff_date


def plot_candlestick(
    prc_date_df: pd.DataFrame,
    freq: str,
    cutoff_date: pd.Timestamp,
    save_path: Optional[str] = None,
) -> any:
    """
    Plot a Plotly-based candlestick chart for **pre-migration** meme token price.

    Args:
        freq (str): Resampling frequency for the pre-migration data (e.g., '5min', '15min').
    """
    df = prc_date_df.copy()
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
        height=600,
        width=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"rangeslider": {"visible": False}},
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_yaxes(row=1, col=1, tickformat=".1e")
    fig.update_yaxes(row=2, col=1)
    fig.update_xaxes(range=[df.index.min() - pd.Timedelta(minutes=0.5), cutoff_date])
    if save_path:
        fig.write_image(save_path)
    fig.show()


if __name__ == "__main__":

    pool = {"token_address": "EjdQW3tbZHDM6RQ7nf6QPfJLNAwDp4yHHAYikRBipump"}
    meme = MemeAnalyzer(
        NewTokenPool(
            token0=SOL_TOKEN_ADDRESS,
            token1=pool["token_address"],
            fee=0,
            pool_add=pool["token_address"],
            block_number=0,
            chain="pre_trump_pumpfun",
            # chain="pumpfun",
            # chain="pre_trump_raydium",
            base_token=pool["token_address"],
            quote_token=SOL_TOKEN_ADDRESS,
            txns={},
        ),
    )
    cut_off_date = get_cutoff_date(meme.prc_date_df)
    plot_candlestick(
        prc_date_df=meme.prc_date_df, freq="1min", cutoff_date=cut_off_date
    )
