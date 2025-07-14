"""Get stats for migration."""

from environ.meme_analyzer import MemeAnalyzer
from environ.constants import SOL_TOKEN_ADDRESS
from environ.data_class import NewTokenPool
from environ.sol_fetcher import import_pool
from tqdm import tqdm
import pandas as pd

res = {
    "token_address": [],
    "migration": [],
    "max_purchase_pct": [],
}

NUM_OF_OBSERVATIONS = 1000

for chain in [
    # "pre_trump_pumpfun",
    # "pre_trump_raydium",
    "pumpfun",
    # "raydium",
]:
    for pool in tqdm(
        import_pool(
            chain,
            NUM_OF_OBSERVATIONS,
        )
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
        res["token_address"].append(token_add)
        res["migration"].append(meme.check_migrate())
        res["max_purchase_pct"].append(meme.check_max_purchase_pct())

    df = pd.DataFrame(res)
