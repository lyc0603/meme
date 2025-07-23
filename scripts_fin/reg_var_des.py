"""Script for regression variable description"""

from environ.constants import (
    TABLE_PATH,
    NAMING_DICT,
    PROFIT_NAMING_DICT,
    PFM_NAMING_DICT,
    ID_DICT,
)

DES_DICT = {
    # Bundle Bot
    "launch_bundle": "Dummy variable equal to 1 if the meme coin has rat bot, 0 otherwise.",
    "sniper_bot": "Dummy variable equal to 1 if the meme coin has sniper bot, 0 otherwise.",
    # Volume Bot / Wash Trading Bot
    "volume_bot": "Dummy variable equal to 1 if the meme coin has wash trading bot, 0 otherwise.",
    # Comments Bot
    "bot_comment_num": "Dummy variable equal to 1 if the meme coin has comment bot, 0 otherwise.",
    # Performance Metrics
    "max_ret": "Maximum log return of the meme coin.",
    "pre_migration_duration": "Natural logarithm of seconds between the meme coin's launch and its migration.",
    "pump_duration": "Natural logarithm of seconds between the launch and the peak price of the meme coin.",
    "dump_duration": "Natural logarithm of seconds between the peak price and the 90\% drop of the circulating supply.",
    "number_of_traders": "Natural logarithm of the number of non-bot traders who control one or multiple wallets and trade the meme coin.",
    # Profit
    "profit": "Each trader's profit and loss from a given meme coin",
    "creator": "Dummy variable equal to 1 if the trader is the meme coin's creator, 0 otherwise.",
    "sniper": "Dummy variable equal to 1 if the trader is a sniper bot, 0 otherwise.",
    "winner": "Dummy variable equal to 1 if the t-statistic of the trader's profit is greater than 2.576, 0 otherwise.",
    "loser": "Dummy variable equal to 1 if the t-statistic of the trader's profit is less than -2.576, 0 otherwise.",
    "neutral": "Dummy variable equal to 1 if the t-statistic of the trader's profit is between -2.576 and 2.576, 0 otherwise.",
}


naming_dict = {
    **NAMING_DICT,
    **PROFIT_NAMING_DICT,
    **PFM_NAMING_DICT,
    **ID_DICT,
}

latex_str = "\\begin{tabularx}{\\textwidth}{lX}\\hline\n"
latex_str += "Variable & Description \\\\\n\\hline\n"

for var, var_des in DES_DICT.items():
    latex_str += f"{naming_dict[var]} & {var_des} \\\\\n"

latex_str += "\\hline\n\\end{tabularx}\n"

with open(TABLE_PATH / "reg_var_des.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
