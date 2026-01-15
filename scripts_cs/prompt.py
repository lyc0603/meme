"""System Instruction and Prompts"""

import json

from environ.constants import FIGURE_PATH


def load_comment(token_add: str, wallet_add: str) -> list[dict]:
    """Load comment history for a given token and wallet address"""
    with open(
        FIGURE_PATH / "comment" / f"{token_add}_{wallet_add}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        comments = f.readlines()
    return "".join(comments)


IMAGE_URL_TEMP = "https://raw.githubusercontent.com/lyc0603/meme/\
refs/heads/main/figures/candle/{ca}.png"

# JSON Schema
JSON_SCHEMA = {
    "name": "meme",
    "schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "boolean",
            },
            "reasoning": {
                "type": "string",
            },
        },
        "required": ["result", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}

JSON_SCHEMA_WALLET_AGENT = {
    "name": "wallet_copy_trading_evaluation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "statistical_significance": {
                        "type": "string",
                        "description": (
                            "Verifier-style check of whether the return t-statistic exceeds the specified "
                            "critical value, with explicit pass/fail outcome."
                        ),
                    },
                    "profitability": {
                        "type": "string",
                        "description": (
                            "Check whether the average return exceeds the profitability threshold, with "
                            "explicit comparison and result."
                        ),
                    },
                    "risk_profile": {
                        "type": "string",
                        "description": (
                            "Check whether return volatility is below the defined risk threshold."
                        ),
                    },
                    "temporal_robustness": {
                        "type": "string",
                        "description": (
                            "Check whether returns across multiple historical horizons consistently exceed "
                            "the threshold."
                        ),
                    },
                    "statistical_robustness": {
                        "type": "string",
                        "description": (
                            "Check whether the number of trades exceeds the minimum sample size threshold."
                        ),
                    },
                    "activity_and_longevity": {
                        "type": "string",
                        "description": (
                            "Check recency (time since last trade < 1 day) and longevity (time since first "
                            "trade > 1 year), with explicit pass/fail outcomes."
                        ),
                    },
                    "final_assessment": {
                        "type": "string",
                        "description": (
                            "Deterministic summary stating whether the wallet qualifies for copy trading "
                            "based on the checks above."
                        ),
                    },
                },
                "required": [
                    "statistical_significance",
                    "profitability",
                    "risk_profile",
                    "temporal_robustness",
                    "statistical_robustness",
                    "activity_and_longevity",
                    "final_assessment",
                ],
            },
            "result": {
                "type": "boolean",
                "description": (
                    "Final binary classification of copy-trading potential derived from the evaluation checks."
                ),
            },
        },
        "required": ["reasoning", "result"],
    },
    "strict": True,
}

JSON_SCHEMA_COIN_AGENT = {
    "name": "coin_investment_evaluation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "manipulation_risk": {
                        "type": "string",
                        "description": (
                            "Verifier-style check for launch manipulation risk (e.g., launch bundle bot), "
                            "with explicit observed value and pass/fail."
                        ),
                    },
                    "sniper_risk": {
                        "type": "string",
                        "description": (
                            "Verifier-style check for sniper bot presence, with explicit observed value and pass/fail."
                        ),
                    },
                    "price_action_quality": {
                        "type": "string",
                        "description": (
                            "Verifier-style check of candlestick pattern quality (gradual trend vs pump-like spike), "
                            "with explicit observed pattern and pass/fail."
                        ),
                    },
                    "liquidity_visibility_support": {
                        "type": "string",
                        "description": (
                            "Auxiliary check for bump-bot/visibility/liquidity support signal, with explicit pass/fail."
                        ),
                    },
                    "engagement_signal_quality": {
                        "type": "string",
                        "description": (
                            "Auxiliary check for engagement/comment signals, discounting bot noise, with explicit conclusion."
                        ),
                    },
                    "final_assessment": {
                        "type": "string",
                        "description": (
                            "Deterministic conclusion summarizing required and auxiliary checks and the final decision."
                        ),
                    },
                },
                "required": [
                    "manipulation_risk",
                    "sniper_risk",
                    "price_action_quality",
                    "liquidity_visibility_support",
                    "engagement_signal_quality",
                    "final_assessment",
                ],
            },
            "result": {
                "type": "boolean",
                "description": "Final binary classification of whether the meme coin is a good investment opportunity.",
            },
        },
        "required": ["reasoning", "result"],
    },
    "strict": True,
}


# Wallet Agent

SYSTEM_INSTRUCTION_WALLET_AGENT = """You are a professional on-chain analyst specializing \
in meme coin wallets trading behavior. You will be given features of a wallet's historical \
meme coin trading activity. Your task is to assess whether the wallet's next trade will be \
profitable so that a user can copy trade from it. Your output must strictly follow the \
specified JSON format."""


PROMPT_WALLET_AGENT = """T-statistic of Returns: {t_stat}
Average Return {average_ret}
Return Standard Deviation {std_ret}
Number of Trades: {num_trades}
Five-to-One Return: {five_to_one_ret}
Ten-to-Six Return: {ten_to_six_ret}
Fifteen-to-Eleven Return: {fifteen_to_eleven_ret}
Time Since Last Trade: {time_since_last_trade}
Time Since First Trade: {time_since_first_trade}
Based on these features, determine whether the wallet's next trade will be profitable so \
that a user can copy trade from it."""

COT_WALLET_AGENT = [
    # Perfect copy trading potential
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_WALLET_AGENT.format(
                    t_stat=24.39,
                    average_ret=1.25,
                    std_ret=0.84,
                    five_to_one_ret=1.14,
                    ten_to_six_ret=1.79,
                    fifteen_to_eleven_ret=1.48,
                    num_trades=4114,
                    time_since_last_trade=371,
                    time_since_first_trade=19118974,
                ),
            }
        ],
    },
    # # 20260114
    # {
    #     "role": "assistant",
    #     "content": """{reasoning":"First, the return t-statistic is substantially \
    # greater than the one-tailed 5% critical value (1.645), indicating strong \
    # and statistically significant historical profitability. Second, the average \
    # return exceeds 1, confirming that the wallet has generated positive net \
    # returns on average. Third, the return standard deviation is less than 1, \
    # suggesting that these profits are achieved with relatively stable \
    # performance rather than excessive volatility. Next, we examine \
    # returns across multiple historical horizons (5-to-1, 10-to-6, and 15-to-11), \
    # all of which are greater than 1. Forth, The wallet has executed a number \
    # of trades over 1000, providing sufficient sample size to rule out noise-driven \
    # outcomes and supporting the reliability of the observed statistics. Fifth, The short time \
    # since the last trade indicates ongoing activity and ensures that the trading \
    # strategy remains current. Finally, the long time span since the first recorded \
    # trade reflects sustained market participation and resilience across different \
    # market regimes. Taken together, the combination of statistical significance, \
    # consistent profitability, low volatility, high trading experience, \
    # and recent activity strongly suggests that this wallet exhibits excellent \
    # copy-trading potential.","result":true"}""",
    # },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "statistical_significance": (
                        "Check whether the return t-statistic exceeds the one-tailed 5% critical value "
                        "(threshold = 1.645). Observed t-statistic = 24.39. Since 24.39 > 1.645, this check "
                        "passes and unlocks subsequent evaluation."
                    ),
                    "profitability": (
                        "Check whether the average return exceeds the profitability threshold (threshold = 1.0). "
                        "Observed average return = 1.25. Since 1.25 > 1.0, this check passes."
                    ),
                    "risk_profile": (
                        "Check whether the return standard deviation is below the volatility threshold "
                        "(threshold = 1.0). Observed standard deviation = 0.84. Since 0.84 < 1.0, this check passes."
                    ),
                    "temporal_robustness": (
                        "Check whether returns across multiple horizons exceed the consistency threshold "
                        "(threshold = 1.0). Observed: 5-to-1 = 1.14, 10-to-6 = 1.79, 15-to-11 = 1.48. "
                        "Since all values are > 1.0, this check passes."
                    ),
                    "statistical_robustness": (
                        "Check whether the number of trades exceeds the minimum sample size threshold "
                        "(threshold = 1000). Observed number of trades = 4114. Since 4114 > 1000, this check passes."
                    ),
                    "activity_and_longevity": (
                        "Check activity using two conditions: (i) recency—time since last trade must be less than "
                        "1 day (threshold = 86400 seconds), and (ii) longevity—time since first trade must exceed "
                        "1 year (threshold = 31536000 seconds). Observed time since last trade = 371 seconds "
                        "(371 < 86400), and time since first trade = 19118974 seconds "
                        "(19118974 < 31536000). Recency condition passes, longevity condition fails."
                    ),
                    "final_assessment": (
                        "All checks except the longevity condition pass. While the wallet demonstrates strong "
                        "statistical significance, profitability, risk control, robustness, and recent activity, "
                        "its trading history does not yet exceed the one-year longevity threshold. The wallet is "
                        "therefore classified as having strong but not maximal copy-trading potential."
                    ),
                },
                "result": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
    # Poor copy trading potential
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_WALLET_AGENT.format(
                    t_stat=0.45,
                    average_ret=1.00,
                    std_ret=0.31,
                    last_ret=0.93,
                    five_to_one_ret=0.98,
                    ten_to_six_ret=1.11,
                    fifteen_to_eleven_ret=0.88,
                    num_trades=461,
                    time_since_last_trade=10,
                    time_since_first_trade=15449,
                ),
            }
        ],
    },
    # # 20260114
    #     {
    #         "role": "assistant",
    #         "content": """{"result":false,"reasoning":"First, the return t-statistic of 0.45 is well below \
    # the one-tailed 5% critical value (1.645), indicating that the wallet's historical \
    # profitability is not statistically significant. This suggests that any observed \
    # profits could be due to random chance rather than a reliable trading strategy. \
    # Second, the average return of 1.00 indicates that the wallet has essentially \
    # broken even on its trades, showing no real profit over time. Third, the return \
    # standard deviation of 0.31 suggests moderate volatility, which, combined with \
    # the lack of significant profits, points to inconsistent performance. Next, \
    # the returns across multiple historical horizons (5-to-1, 10-to-6, and 15-to-11) \
    # are mixed, with some below 1, indicating losses in certain periods. Forth, \
    # the wallet has executed a relatively low number of trades (461), which may not provide \
    # a sufficient sample size to draw reliable conclusions about its trading strategy. \
    # Fifth, while the wallet has traded recently (10 minutes ago), the short time \
    # since the last trade does not compensate for the overall lack of profitability. \
    # Finally, the relatively short time span since the first recorded trade suggests \
    # that the wallet may lack extensive market experience. Taken together, the \
    # combination of statistical insignificance, break-even returns, inconsistent \
    # performance, limited trading experience, and recent activity suggests that \
    # this wallet exhibits poor copy-trading potential."}""",
    #     },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "statistical_significance": (
                        "Check whether the return t-statistic exceeds the one-tailed 5% critical value "
                        "(threshold = 1.645). Observed t-statistic = 0.45. Since 0.45 < 1.645, this check "
                        "fails. The statistical significance gate is not satisfied."
                    ),
                    "profitability": (
                        "Check whether the average return exceeds the profitability threshold "
                        "(threshold = 1.0). Observed average return = 1.00. Since 1.00 is not greater "
                        "than 1.0, this check fails."
                    ),
                    "risk_profile": (
                        "Check whether the return standard deviation is below the volatility threshold "
                        "(threshold = 1.0). Observed standard deviation = 0.31. Since 0.31 < 1.0, this "
                        "check passes, although low volatility alone does not imply profitability."
                    ),
                    "temporal_robustness": (
                        "Check whether returns across multiple horizons exceed the consistency threshold "
                        "(threshold = 1.0). Observed: 5-to-1 = 0.98, 10-to-6 = 1.11, 15-to-11 = 0.88. "
                        "Since multiple horizons are below 1.0, this check fails."
                    ),
                    "statistical_robustness": (
                        "Check whether the number of trades exceeds the minimum sample size threshold "
                        "(threshold = 1000). Observed number of trades = 461. Since 461 < 1000, this "
                        "check fails."
                    ),
                    "activity_and_longevity": (
                        "Check activity using two conditions: (i) recency—time since last trade must be "
                        "less than 1 day (threshold = 86400 seconds), and (ii) longevity—time since first "
                        "trade must exceed 1 year (threshold = 31536000 seconds). Observed time since last "
                        "trade = 10 seconds (10 < 86400), so recency passes. Observed time since first "
                        "trade = 15449 seconds (15449 < 31536000), so longevity fails."
                    ),
                    "final_assessment": (
                        "The primary statistical significance gate fails, and additional checks reveal "
                        "break-even profitability, inconsistent performance across horizons, insufficient "
                        "sample size, and inadequate trading longevity. Although the wallet is recently "
                        "active and exhibits low volatility, these factors do not compensate for the lack "
                        "of statistically significant and robust profitability. The wallet is therefore "
                        "classified as having poor copy-trading potential."
                    ),
                },
                "result": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
]

# Coin Agent
SYSTEM_INSTRUCTION_COIN_AGENT = """You are a professional on-chain analyst specializing \
in meme coin investment potential. You will be given various transaction features, \
candlestick chart, and comment history related to a meme coin. Your task is to \
assess whether the meme coin is a good investment opportunity. Your output must \
strictly follow the specified JSON format."""

PROMPT_COIN_AGENT = """Transaction Features:
Bundle Bot: {launch_bundle}
Sniper Bot: {sniper_bot}
Bump Bot: {bump_bot}
Comment Bot: {comment_bot}
Comment History: {comment_history}
Based on these features, determine whether the meme coin is a good investment opportunity."""


COT_COIN_AGENT = [
    # Perfect coin to invest
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_COIN_AGENT.format(
                    launch_bundle="No",
                    sniper_bot="No",
                    bump_bot="Yes",
                    comment_bot="Yes",
                    comment_history=load_comment(
                        token_add="2F84uaBysP4sD7Qby33fAM9RrSzhcR9KWrUT1ixwpump",
                        wallet_add="6RWPsWL82zshgpxJCQ2Z7rr57njQxz3LVPVBG5qbM1Yi",
                    ),
                ),
            },
            {
                # Gradually increasing candlestick chart
                "type": "image_url",
                "image_url": {
                    "url": IMAGE_URL_TEMP.format(
                        ca="2F84uaBysP4sD7Qby33fAM9RrSzhcR9KWrUT1ixwpump"
                        "_"
                        "6RWPsWL82zshgpxJCQ2Z7rr57njQxz3LVPVBG5qbM1Yi"
                    ),
                    "detail": "high",
                },
            },
        ],
    },
    #     # 20260114
    #     {
    #         "role": "assistant",
    #         "content": """{"result":true,"reasoning":"First, the absence of a launch \
    # bundle bot suggests that the coin was not artificially inflated at inception, \
    # pointing to a more organic launch process. Second, the lack of sniper bot activity \
    # indicates that early trading was not dominated by predatory behavior, which often \
    # contributes to excessive volatility and market instability. The presence of a bump \
    # bot is interpreted as a supportive—but non-essential—signal, as it may help sustain \
    # short-term visibility and liquidity without being a necessary condition for positive \
    # performance. Similarly, the detection of comment bots is treated as an auxiliary \
    # indicator of engagement rather than a decisive factor, as automated activity alone \
    # does not guarantee long-term value. Importantly, the candlestick chart exhibits \
    # a gradual and consistent upward trend, indicating sustained investor interest rather \
    # than abrupt speculative price movements. Finally, the comment history reflects \
    # ongoing activity with a mix of automated and seemingly genuine interactions, \
    # which supports the interpretation of continued market attention. Taken together, \
    # the coin demonstrates low manipulation risk and favorable price dynamics, with \
    # auxiliary engagement signals providing incremental confidence. These factors \
    # collectively suggest a strong investment opportunity."}""",
    #     },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "manipulation_risk": (
                        "Check whether a launch bundle bot is absent (required condition: Bundle Bot == 'No'). "
                        "Observed Bundle Bot = 'No'. This check passes, reducing risk of artificial launch inflation."
                    ),
                    "sniper_risk": (
                        "Check whether sniper bot activity is absent (required condition: Sniper Bot == 'No'). "
                        "Observed Sniper Bot = 'No'. This check passes, reducing risk of predatory early trading."
                    ),
                    "price_action_quality": (
                        "Check whether the candlestick chart indicates a gradual, sustained upward trend rather than "
                        "abrupt spikes and dumps (required condition: trend is gradual and consistent). "
                        "Observed chart pattern = gradual increase. This check passes, supporting healthier price discovery."
                    ),
                    "liquidity_visibility_support": (
                        "Check whether a bump bot is present (auxiliary condition: Bump Bot == 'Yes'). "
                        "Observed Bump Bot = 'Yes'. This check passes and is treated as supportive but not required."
                    ),
                    "engagement_signal_quality": (
                        "Check whether comment activity indicates ongoing attention while discounting purely automated spam "
                        "(auxiliary condition: evidence of sustained discussion beyond repeated boilerplate). "
                        "Observed Comment Bot = 'Yes', but comment history shows ongoing activity; treat this as weakly "
                        "supportive and non-decisive."
                    ),
                    "final_assessment": (
                        "All required checks pass (no launch bundle bot, no sniper bot, and favorable gradual price action). "
                        "Auxiliary signals (bump bot presence and ongoing comment activity) provide incremental confidence "
                        "but are not necessary for a positive decision. Classify the coin as a good investment opportunity."
                    ),
                },
                "result": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
    # Poor coin to invest
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_COIN_AGENT.format(
                    launch_bundle="Yes",
                    sniper_bot="Yes",
                    bump_bot="No",
                    comment_bot="No",
                    comment_history=load_comment(
                        token_add="2f6Qh5c2UjiBKEe7oaifJLpdU52JSoQmhfCDGu3Epump",
                        wallet_add="DQjaat2dDjZvyADdBKmuFzBWU9ey9M7yCVhK4MdSPTXE",
                    ),
                ),
            },
            {
                # One large green candle
                "type": "image_url",
                "image_url": {
                    "url": IMAGE_URL_TEMP.format(
                        ca="2f6Qh5c2UjiBKEe7oaifJLpdU52JSoQmhfCDGu3Epump"
                        "_"
                        "DQjaat2dDjZvyADdBKmuFzBWU9ey9M7yCVhK4MdSPTXE"
                    ),
                    "detail": "high",
                },
            },
        ],
    },
    # # 20260114
    #     {
    #         "role": "assistant",
    #         "content": """{"result":false,"reasoning":"First, the presence of a launch bundle bot \
    # indicates that the coin was artificially hyped at launch, which often leads to \
    # unsustainable price levels and subsequent crashes. Next, the candlestick chart \
    # shows a single large green candle, which may indicate a pump-and-dump scheme \
    # rather than genuine investor interest. Taken together, these factors suggest that \
    # the coin exhibits poor investment potential."}""",
    #     },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "manipulation_risk": (
                        "Check whether a launch bundle bot is absent (required condition: Bundle Bot == 'No'). "
                        "Observed Bundle Bot = 'Yes'. Since 'Yes' != 'No', this check fails, indicating elevated "
                        "risk of artificial launch inflation and coordinated hype."
                    ),
                    "sniper_risk": (
                        "Check whether sniper bot activity is absent (required condition: Sniper Bot == 'No'). "
                        "Observed Sniper Bot = 'Yes'. Since 'Yes' != 'No', this check fails, indicating predatory "
                        "early trading pressure and increased manipulation risk."
                    ),
                    "price_action_quality": (
                        "Check whether the candlestick chart indicates gradual, sustained price discovery (required "
                        "condition: gradual and consistent trend; no single-candle surge). Observed chart pattern = "
                        "one large green candle. This check fails, consistent with a pump-like move rather than "
                        "organic growth."
                    ),
                    "liquidity_visibility_support": (
                        "Check whether a bump bot is present (auxiliary condition: Bump Bot == 'Yes'). "
                        "Observed Bump Bot = 'No'. This auxiliary check fails and provides no supportive signal."
                    ),
                    "engagement_signal_quality": (
                        "Check whether there is evidence of sustained engagement (auxiliary condition: non-trivial "
                        "comment activity beyond absence of bots). Observed Comment Bot = 'No' and comment history "
                        "does not provide a strong engagement signal. Treat this as non-supportive."
                    ),
                    "final_assessment": (
                        "Multiple required checks fail (launch bundle bot present, sniper bot present, and pump-like "
                        "price action). Auxiliary signals do not provide offsetting support. Classify the coin as a "
                        "poor investment opportunity."
                    ),
                },
                "result": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
]
