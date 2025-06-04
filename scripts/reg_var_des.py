"""Script for regression variable description"""

from environ.constants import TABLE_PATH, NAMING_DICT, PROFIT_NAMING_DICT

DES_DICT = {
    # Size
    "duration": "Log number of seconds between the meme coin's launch on Pump.fun and its migration to Raydium.",
    "#trader": "Log number of unique traders of the meme coin on Pump.fun before migration.",
    "#txn": "Log number of transactions of the meme coin on Pump.fun before migration.",
    "#transfer": "Log number of transfers of the meme coin between wallets before migration.",
    # Bundle Bot
    "holding_herf": "Herfindahl index of the meme coin's holdings before migration.",
    "bundle": "Herfindahl index of the transactions number per block before migration.",
    "transfer_amount": "Log total amount of meme coin transfer between wallets divided by the total supply before migration.",
    # Volume Bot / Wash Trading Bot
    "max_same_txn": "Log maximum number of transactions with same token quantity made by a single trader of this meme coin before migration.",
    "pos_to_number_of_swaps_ratio": "Average ratio of net position plus one to the number of swaps made by traders of this meme coin before migration.",
    # Comments Bot
    "unique_replies": "Log number of unique comments to the meme coin on Pump.fun before migration.",
    "reply_interval_herf": "Herfindahl index of the time intervals between comments to the meme coin on Pump.fun before migration.",
    "unique_repliers": "Log number of unique users who commented on the meme coin on Pump.fun before migration.",
    "non_swapper_repliers": "Log number of unique users who commented on the meme coin on Pump.fun before migration but did not trade it.",
    # Devs Behavior
    "dev_transfer": "Dummy variable equal to 1 if the meme coin's creator made transfers with other wallet, 0 otherwise.",
    "dev_buy": "Dummy variable equal to 1 if the meme coin's creator made buy transactions, 0 otherwise.",
    "dev_sell": "Dummy variable equal to 1 if the meme coin's creator made sell transactions, 0 otherwise.",
    # Profit
    "profit": "Each trader's profit from a given meme coin within 12 hours after its migration",
    "creator": "Dummy variable equal to 1 if the trader is the meme coin's creator, 0 otherwise.",
}

flat_naming_dict = {
    sub_key: sub_value
    for category in NAMING_DICT.values()
    for sub_key, sub_value in category.items()
}

naming_dict = {
    **flat_naming_dict,
    **PROFIT_NAMING_DICT,
}

latex_str = "\\begin{tabularx}{\\textwidth}{lX}\\hline\n"
latex_str += "Variable & Description \\\\\n\\hline\n"

for var, var_name in naming_dict.items():
    latex_str += f"{var_name} & {DES_DICT[var]} \\\\\n"

latex_str += "\\hline\n\\end{tabularx}\n"

with open(TABLE_PATH / "reg_var_des.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
