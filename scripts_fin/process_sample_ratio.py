"""Process the ratio of different meme coin projects."""

import json
from typing import Any, Literal

from tqdm import tqdm

from environ.constants import SOL_TOKEN_ADDRESS, SOLANA_PATH_DICT, PROCESSED_DATA_PATH
from environ.data_class import NewTokenPool
from environ.meme_analyzer import MemeAnalyzer
from collections import defaultdict

NUM_OF_OBSERVATIONS = 1000
SOL_TOKEN_ADDRESS = "So11111111111111111111111111111111111111112"


PRE_POST_NAMING = {
    "pre_trump_pumpfun": "pre_trump_",
    "pumpfun": "",
}

group_num = defaultdict(int)


def import_meme(
    category: Literal["pumpfun", "pre_trump_pumpfun"],
    num: int = NUM_OF_OBSERVATIONS,
) -> None:
    """Utility function to fetch the pool list."""

    counter = 0
    with open(
        SOLANA_PATH_DICT[category],
        "r",
        encoding="utf-8",
    ) as f:
        for line in tqdm(f, desc=f"Importing {category} pools"):

            token_add = json.loads(line)["token_address"]
            meme = MemeAnalyzer(
                NewTokenPool(
                    token0=SOL_TOKEN_ADDRESS,
                    token1=token_add,
                    fee=0,
                    pool_add=token_add,
                    block_number=0,
                    chain=category,
                    base_token=token_add,
                    quote_token=SOL_TOKEN_ADDRESS,
                    txns={},
                )
            )

            if meme.check_migrate():
                group_num[f"{PRE_POST_NAMING[category]}raydium"] += 1
            elif meme.check_max_purchase_pct() < 0.2:
                group_num[f"{PRE_POST_NAMING[category]}no_one_care"] += 1
            else:
                group_num[f"{PRE_POST_NAMING[category]}pumpfun"] += 1

            counter += 1

            if counter >= num:
                break

        # for name in [
        #     f"{PRE_POST_NAMING[category]}raydium",
        #     f"{PRE_POST_NAMING[category]}no_one_care",
        #     f"{PRE_POST_NAMING[category]}pumpfun",
        # ]:
        #     group_num[name] = group_num[name] / (NUM_OF_OBSERVATIONS * 2)


if __name__ == "__main__":
    res_dict = {}
    categories = ["pumpfun", "pre_trump_pumpfun"]
    for category in categories:
        import_meme(category, NUM_OF_OBSERVATIONS)

    with open(
        PROCESSED_DATA_PATH / "meme_project_ratio.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(group_num, f, indent=4)
