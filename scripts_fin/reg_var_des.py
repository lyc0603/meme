"""Script for regression variable description"""

from environ.constants import (
    TABLE_PATH,
    NAMING_DICT,
    PROFIT_NAMING_DICT,
    PFM_NAMING_DICT,
)

DES_DICT = {
    # Bundle Bot
    "launch_bundle": "Dummy variable that equals 1 if the meme coin has launch bundle, 0 otherwise.",
    # "launch_bundle_transfer": "Dummy variable that equals 1 if the meme coin has creator wallet funded launch bundle, 0 otherwise.",
    # "bundle_creator_buy": "Dummy variable that equals 1 if the meme coin has any creator wallet funded bundle buy, 0 otherwise.",
    # "bundle_launch": "Dummy variable that equals 1 if the meme coin has any launch bundle, 0 otherwise.",
    "bundle_bot": "Dummy variable that equals 1 if the number of buy and sell bundles is above the sample median, 0 otherwise.",
    # Volume Bot / Wash Trading Bot
    "volume_bot": "Dummy variable that equals 1 if there are volume bots in this project, 0 otherwise.",
    # Comments Bot
    "bot_comment_num": "Dummy variable that equals 1 if there are comment bots in this project, 0 otherwise.",
    # Performance Metrics
    "max_ret": "Maximum log return of the meme coin within 12 hours after its migration.",
    "pre_migration_duration": "Natural logarithm of seconds between the meme coin's launch and its migration.",
    "pump_duration": "Natural logarithm of seconds between the migration and the peak price of the meme coin.",
    "dump_duration": "Natural logarithm of seconds between the peak price and the 90\% price drop of the meme coin.",
    "number_of_traders": "Natural logarithm of the number of non-bot traders who control one or multiple wallets and trade the meme coin between the launch and 12 hours after its migration.",
    # Profit
    "profit": "Each trader's profit and loss from a given meme coin between the launch and 12 hours after its migration",
    "creator": "Dummy variable that equals 1 if the trader is the meme coin's creator, 0 otherwise.",
}


naming_dict = {
    **NAMING_DICT,
    **PROFIT_NAMING_DICT,
    **PFM_NAMING_DICT,
}

latex_str = "\\begin{tabularx}{\\textwidth}{lX}\\hline\n"
latex_str += "Variable & Description \\\\\n\\hline\n"

for var, var_name in naming_dict.items():
    latex_str += f"{var_name} & {DES_DICT[var]} \\\\\n"

latex_str += "\\hline\n\\end{tabularx}\n"

with open(TABLE_PATH / "reg_var_des.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
