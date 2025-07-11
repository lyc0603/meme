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
    "bundle_buy": "Dummy variable that equals 1 if the buy bundle number is above the sample median, 0 otherwise.",
    "bundle_sell": "Dummy variable that equals 1 if the sell bundle number is above the sample median, 0 otherwise.",
    # Volume Bot / Wash Trading Bot
    "volume_bot": "Dummy variable that equals 1 if the number of volume bots is above the sample median, 0 otherwise.",
    "wash_trading_volume_frac": "Fraction of the meme coin's volume that is attributed to volume bots.",
    # Comments Bot
    "positive_bot_comment_num": "Dummy variable that equals 1 if the number of positive comments made by bots is above the sample median, 0 otherwise.",
    "negative_bot_comment_num": "Dummy variable that equals 1 if the number of negative comments made by bots is above the sample median, 0 otherwise.",
    # Performance Metrics
    "max_ret": "Maximum log return of the meme coin within 12 hours after its migration.",
    "pre_migration_duration": "Natural logarithm of the number of seconds between the meme coin's launch and its migration.",
    "pump_duration": "Natural logarithm of the number of seconds between the migration and the peak price of the meme coin.",
    "dump_duration": "Natural logarithm of the number of seconds between the migration and the 90\% price drop of the meme coin.",
    "pre_migration_vol": "Volatility of the meme coin's log returns between its launch and the migration.",
    "post_migration_vol": "Volatility of the meme coin's log returns between its migration and 12 hours after its migration.",
    "number_of_traders": "Natural logarithm of the number of non-bot traders who traded the meme coin between the launch and 12 hours after its migration.",
    # Profit
    "profit": "Each trader's profit from a given meme coin within 12 hours after its migration",
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
