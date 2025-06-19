"""Script for regression variable description"""

from environ.constants import TABLE_PATH, NAMING_DICT, PROFIT_NAMING_DICT

DES_DICT = {
    # Bundle Bot
    "launch_bundle_transfer": "Dummy variable that equals 1 if the meme coin has creator wallet funded launch bundle, 0 otherwise.",
    "bundle_creator_buy": "Dummy variable that equals 1 if the meme coin has any creator wallet funded bundle buy, 0 otherwise.",
    "bundle_launch": "Dummy variable that equals 1 if the meme coin has any launch bundle, 0 otherwise.",
    "bundle_buy": "Dummy variable that equals 1 if the buy bundle number is above the sample median, 0 otherwise.",
    "bundle_sell": "Dummy variable that equals 1 if the sell bundle number is above the sample median, 0 otherwise.",
    # Volume Bot / Wash Trading Bot
    "max_same_txn": "Dummy variable that equals 1 if the maximum number of transactions with the same token quantity made by a single trader of this meme coin before migration is above the sample median, 0 otherwise.",
    "pos_to_number_of_swaps_ratio": "Dummy variable that equals 1 if the average ratio of net position plus one to the number of swaps made by traders of this meme coin before migration is above the sample median, 0 otherwise.",
    # Comments Bot
    "positive_bot_comment_num": "Dummy variable that equals 1 if the number of positive comments made by bots is above the sample median, 0 otherwise.",
    "negative_bot_comment_num": "Dummy variable that equals 1 if the number of negative comments made by bots is above the sample median, 0 otherwise.",
    "bot_comment_num": "Dummy variable that equals 1 if the number of comments made by bots is above the sample median, 0 otherwise.",
    # Profit
    "profit": "Each trader's profit from a given meme coin within 12 hours after its migration",
    "creator": "Dummy variable that equals 1 if the trader is the meme coin's creator, 0 otherwise.",
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
