"""System Instruction, Prompt, and JSON Schema"""

import json

from environ.constants import FIGURE_PATH


# ------------ Utils ------------ #
def load_comment(token_add: str, wallet_add: str) -> list[dict]:
    """Load comment history for a given token and wallet address"""
    with open(
        FIGURE_PATH / "comment" / f"{token_add}_{wallet_add}.txt",
        "r",
        encoding="utf-8",
    ) as f:
        comments = f.readlines()
    return "".join(comments)


IMAGE_URL_TEMP = (
    "https://raw.githubusercontent.com/lyc0603/meme/"
    + "refs/heads/main/figures/candle/{ca}.png"
)

# ------------ Wallet Agent ------------ #
SYSTEM_INSTRUCTION_WALLET_AGENT = """You are a professional on-chain analyst specializing in meme coin wallets trading behavior. You will be given features of a wallet's historical meme coin trading activity. Your task is to assess whether the wallet's next trade will be profitable so that a user can copy trade from it."""

JSON_SCHEMA_WALLET_AGENT = {
    "name": "wallet_agent",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "object",
                "properties": {
                    "Statistical Significance": {"type": "string"},
                    "Profitability": {"type": "string"},
                    "Risk Profile": {"type": "string"},
                    "Temporal Robustness": {"type": "string"},
                    "Statistical Robustness": {"type": "string"},
                    "Activity and Longevity": {"type": "string"},
                    "Summary": {"type": "string"},
                },
                "required": [
                    "Statistical Significance",
                    "Profitability",
                    "Risk Profile",
                    "Temporal Robustness",
                    "Statistical Robustness",
                    "Activity and Longevity",
                    "Summary",
                ],
                "additionalProperties": False,
            },
            "result": {"type": "boolean"},
        },
        "required": ["reasoning", "result"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT_WALLET_AGENT = """T-statistic of Returns: {t_stat}
Average Return: {average_ret}
Return Standard Deviation: {std_ret}
Number of Trades: {num_trades}
Five-to-One Return: {five_to_one_ret}
Ten-to-Six Return: {ten_to_six_ret}
Fifteen-to-Eleven Return: {fifteen_to_eleven_ret}
Time Since Last Trade: {time_since_last_trade}
Time Since First Trade: {time_since_first_trade}
Based on these features, determine whether the wallet's next trade will be profitable so \
that a user can copy trade from it."""

COT_WALLET_AGENT = [
    # CoT 1: Strong copy trading potential
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
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "Statistical Significance": (
                        "Gate check: require t-statistic > 1.645 (one-tailed 5%). "
                        "Observed t-statistic = 24.39. Since 24.39 > 1.645, the gate passes."
                    ),
                    "Profitability": (
                        "Check average return > 1.0. Observed average return = 1.25. "
                        "Since 1.25 > 1.0, this check passes."
                    ),
                    "Risk Profile": (
                        "Check standard deviation < 1.0. Observed std = 0.84. "
                        "Since 0.84 < 1.0, this check passes."
                    ),
                    "Temporal Robustness": (
                        "Check multi-horizon consistency: require 5-to-1, 10-to-6, and 15-to-11 returns all > 1.0. "
                        "Observed (1.14, 1.79, 1.48). All exceed 1.0, so this check passes."
                    ),
                    "Statistical Robustness": (
                        "Check sample size: require num_trades > 1000. Observed num_trades = 4114. "
                        "Since 4114 > 1000, this check passes."
                    ),
                    "Activity and Longevity": (
                        "Auxiliary checks: (i) recency require time_since_last_trade < 86400 seconds; "
                        "observed 371 < 86400 so recency passes. (ii) longevity prefer time_since_first_trade > "
                        "31536000 seconds; observed 19118974 < 31536000 so longevity fails. "
                        "Longevity is treated as a preference, not a veto."
                    ),
                    "Summary": (
                        "All core checks pass (statistical significance, profitability, risk control, "
                        "multi-horizon robustness, and sufficient sample size). The wallet is attractive "
                        "for copy trading despite not meeting the one-year longevity preference."
                    ),
                },
                "result": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
    # CoT 2: Poor copy trading potential
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_WALLET_AGENT.format(
                    t_stat=0.45,
                    average_ret=1.00,
                    std_ret=0.31,
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
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "Statistical Significance": (
                        "Gate check: require t-statistic > 1.645 (one-tailed 5%). "
                        "Observed t-statistic = 0.45. Since 0.45 < 1.645, the gate fails."
                    ),
                    "Profitability": (
                        "Check average return > 1.0. Observed average return = 1.00. "
                        "Since 1.00 is not greater than 1.0, this check fails."
                    ),
                    "Risk Profile": (
                        "Check standard deviation < 1.0. Observed std = 0.31. "
                        "This passes, but low volatility without profitability is not sufficient."
                    ),
                    "Temporal Robustness": (
                        "Check multi-horizon consistency: require all horizons > 1.0. "
                        "Observed (0.98, 1.11, 0.88). Since multiple horizons are below 1.0, this check fails."
                    ),
                    "Statistical Robustness": (
                        "Check sample size: require num_trades > 1000. Observed num_trades = 461. "
                        "Since 461 < 1000, this check fails."
                    ),
                    "Activity and Longevity": (
                        "Auxiliary checks: recency 10 < 86400 passes; longevity 15449 < 31536000 fails. "
                        "Auxiliary signals cannot compensate for failed core checks."
                    ),
                    "Summary": (
                        "The statistical-significance gate fails and robustness checks do not support "
                        "reliable profitability. The wallet is unsuitable for next-trade copy trading."
                    ),
                },
                "result": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
]


# ------------ Coin Agent ------------ #
SYSTEM_INSTRUCTION_COIN_AGENT = """You are a professional on-chain analyst specializing in meme coin investment potential. You will be given various transaction features, candlestick chart, and comment history related to a meme coin. Your task is to assess whether the meme coin is a good investment opportunity."""

JSON_SCHEMA_COIN_AGENT = {
    "name": "coin_agent",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "object",
                "properties": {
                    "Bundle Bot": {"type": "string"},
                    "Candlestick Pattern": {"type": "string"},
                    "Sniper Bot": {"type": "string"},
                    "Bump Bot": {"type": "string"},
                    "Comments": {"type": "string"},
                    "Summary": {"type": "string"},
                },
                "required": [
                    "Bundle Bot",
                    "Candlestick Pattern",
                    "Sniper Bot",
                    "Bump Bot",
                    "Comments",
                    "Summary",
                ],
                "additionalProperties": False,
            },
            "result": {"type": "boolean"},
        },
        "required": ["reasoning", "result"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT_COIN_AGENT = """Transaction Features:
Bundle Bot: {launch_bundle}
Sniper Bot: {sniper_bot}
Bump Bot: {bump_bot}
Comment Bot: {comment_bot}
Comment History: {comment_history}"""

COT_COIN_AGENT = [
    # CoT 1: Good coin (positive case)
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
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "Bundle Bot": (
                        "Required check: Bundle Bot must be 'No'. "
                        "Observed Bundle Bot = 'No'. Since 'No' == 'No', this check passed."
                    ),
                    "Candlestick Pattern": (
                        "Required check: Candlestick chart should show gradual and sustained price discovery "
                        "(no single-candle spike-and-dump signature). "
                        "Observed pattern = gradual increase. This check passed."
                    ),
                    "Sniper Bot": (
                        "Auxiliary check: Sniper Bot ideally 'No'. "
                        "Observed Sniper Bot = 'No'. This auxiliary check passed, reducing early predatory trading risk."
                    ),
                    "Bump Bot": (
                        "Auxiliary check: Bump Bot may be 'Yes' as a supportive visibility signal but is not required. "
                        "Observed Bump Bot = 'Yes'. Treat as weakly supportive."
                    ),
                    "Comments": (
                        "Auxiliary check: Engagement should show sustained activity, discounting repetitive spam. "
                        "Observed Comment Bot = 'Yes'. Treat bot presence as a caution, but if the comment history contains "
                        "non-boilerplate discussion, it can still provide weak support. Overall: weakly supportive / non-decisive."
                    ),
                    "Summary": (
                        "All required checks passed (Bundle Bot absent; chart shows gradual upward trend). "
                        "Auxiliary signals do not contradict the decision. Classify as a good investment opportunity."
                    ),
                },
                "result": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
    # CoT 2: Bad coin (negative case)
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
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "Bundle Bot": (
                        "Required check: Bundle Bot must be 'No'. "
                        "Observed Bundle Bot = 'Yes'. Since 'Yes' != 'No', this check failed."
                    ),
                    "Candlestick Pattern": (
                        "Required check: Candlestick chart should show gradual and sustained price discovery. "
                        "Observed pattern = one large green candle (pump-like move). This check failed."
                    ),
                    "Sniper Bot": (
                        "Auxiliary check: Sniper Bot ideally 'No'. "
                        "Observed Sniper Bot = 'Yes'. This is a negative auxiliary signal (predatory early trading risk)."
                    ),
                    "Bump Bot": (
                        "Auxiliary check: Bump Bot may be 'Yes' as supportive but is not required. "
                        "Observed Bump Bot = 'No'. Not supportive."
                    ),
                    "Comments": (
                        "Auxiliary check: Engagement should show sustained activity. "
                        "Observed Comment Bot = 'No'. Without strong engagement evidence in comment history, treat as non-supportive."
                    ),
                    "Summary": (
                        "One or more required checks failed (Bundle Bot present and pump-like price action). "
                        "Auxiliary signals do not offset these failures. Classify as a poor investment opportunity."
                    ),
                },
                "result": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
]


# ------------ Timing Agent ------------ #
SYSTEM_INSTRUCTION_TIMING_AGENT = """You are a professional on-chain analyst specializing in memecoin copy trading timing. You will be given timing features of a targeted wallet in a memecoin. Your task is to assess whether the wallet is suitable for copy trading in this memecoin."""

JSON_SCHEMA_TIMING_AGENT = {
    "name": "timing_agent",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "object",
                "properties": {
                    "Trader Purchase Price": {"type": "string"},
                    "Trader Purchase Amount": {"type": "string"},
                    "Trader Purchase Quantity": {"type": "string"},
                    "Summary": {"type": "string"},
                },
                "required": [
                    "Trader Purchase Price",
                    "Trader Purchase Amount",
                    "Trader Purchase Quantity",
                    "Summary",
                ],
                "additionalProperties": False,
            },
            "result": {"type": "boolean"},
        },
        "required": ["reasoning", "result"],
        "additionalProperties": False,
    },
    "strict": True,
}

PROMPT_TIMING_AGENT = """Trader Purchase Price: {first_txn_price}
Trader Purchase Amount: {first_txn_amount}
Trader Purchase Quantity: {first_txn_quantity}"""

COT_TIMING_AGENT = [
    # CoT 1: Successful case
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_TIMING_AGENT.format(
                    first_txn_price="4.95e-06",
                    first_txn_amount="99.65",
                    first_txn_quantity="6026170.61",
                ),
            }
        ],
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "Trader Purchase Price": (
                        "Check whether the purchase price is lower than 1.13e-05. "
                        "Observed value is 4.95e-06. Since 4.95e-06 < 1.13e-05, this check passed."
                    ),
                    "Trader Purchase Amount": (
                        "Check whether the purchase amount is less than 167.12. Observed value is 99.65. "
                        "Since 99.65 < 167.12, this check passed."
                    ),
                    "Trader Purchase Quantity": (
                        "Check whether the purchase quantity is less than 20922965.16. Observed value is 6026170.61. "
                        "Since 6026170.61 < 20922965.16, this check passed."
                    ),
                    "Summary": "All checks passed, indicating a suitable timing for copy trading.",
                },
                "result": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
    # CoT 2: Unsuccessful case
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT_TIMING_AGENT.format(
                    first_txn_price="1.19e-05",
                    first_txn_amount="661.21",
                    first_txn_quantity="55153573.25",
                ),
            }
        ],
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "reasoning": {
                    "Trader Purchase Price": (
                        "Check whether the purchase price is lower than 1.13e-05. Observed value is 1.19e-05. "
                        "Since 1.19e-05 > 1.13e-05, this check failed."
                    ),
                    "Trader Purchase Amount": (
                        "Check whether the purchase amount is less than 167.12. Observed value is 661.21. "
                        "Since 661.21 > 167.12, this check failed."
                    ),
                    "Trader Purchase Quantity": (
                        "Check whether the purchase quantity is less than 20922965.16. Observed value is 55153573.25. "
                        "Since 55153573.25 > 20922965.16, this check failed."
                    ),
                    "Summary": "One or more checks failed, indicating an unsuitable timing for copy trading.",
                },
                "result": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    },
]
