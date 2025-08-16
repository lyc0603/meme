"""Script to plot case study results"""

from environ.agent_data_loader import get_cutoff_date, plot_candlestick
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from environ.constants import FIGURE_PATH, SOL_TOKEN_ADDRESS

pool = {"token_address": "2F84uaBysP4sD7Qby33fAM9RrSzhcR9KWrUT1ixwpump"}
meme = MemeAnalyzer(
    NewTokenPool(
        token0=SOL_TOKEN_ADDRESS,
        token1=pool["token_address"],
        fee=0,
        pool_add=pool["token_address"],
        block_number=0,
        # chain="pre_trump_pumpfun",
        chain="pumpfun",
        # chain="pre_trump_raydium",
        base_token=pool["token_address"],
        quote_token=SOL_TOKEN_ADDRESS,
        txns={},
    ),
)
cutoff_date = get_cutoff_date(meme.prc_date_df)
plot_candlestick(
    prc_date_df=meme.prc_date_df,
    freq="1min",
    cutoff_date=cutoff_date,
    save_path=FIGURE_PATH / "case_study_candlestick.png",
)
