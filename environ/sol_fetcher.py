"""Solana Fetcher"""

from typing import Iterable, Literal, Optional

from borsh_construct import U64, CStruct
from construct import Bytes, Int8ul, Int32ul
from solana.rpc.api import Client
from solders.pubkey import Pubkey
from solders.rpc.responses import GetSignaturesForAddressResp
from solders.signature import Signature
from solders.transaction_status import (
    EncodedConfirmedTransactionWithStatusMeta,
    ParsedInstruction,
)

PUBLIC_KEY_LAYOUT = Bytes(32)
SPL_ACCOUNT_LAYOUT = CStruct(
    "mint" / PUBLIC_KEY_LAYOUT,
    "owner" / PUBLIC_KEY_LAYOUT,
    "amount" / U64,
    "delegateOption" / Int32ul,
    "delegate" / PUBLIC_KEY_LAYOUT,
    "state" / Int8ul,
    "isNativeOption" / Int32ul,
    "isNative" / U64,
    "delegatedAmount" / U64,
    "closeAuthorityOption" / Int32ul,
    "closeAuthority" / PUBLIC_KEY_LAYOUT,
)
SPL_MINT_LAYOUT = CStruct(
    "mintAuthorityOption" / Int32ul,
    "mintAuthority" / PUBLIC_KEY_LAYOUT,
    "supply" / U64,
    "decimals" / Int8ul,
    "isInitialized" / Int8ul,
    "freezeAuthorityOption" / Int32ul,
    "freezeAuthority" / PUBLIC_KEY_LAYOUT,
)
solana_client = Client(
    # "https://solana-mainnet.core.chainstack.com/a0db22a6450d2ad8bfabb1b8254b7abb"
    "https://api.mainnet-beta.solana.com"
)

RAYDIUM_ADDRESSES_STR = [
    # Standard AMM (CP-Swap, New)
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",
    # Legacy AMM v4 (OpenBook)
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    # Stable Swap AMM
    "5quBtoiQqxF9Jv6KYKctB59NT3gtJD2Y65kdnB1Uev3h",
    # Concentrated Liquidity (CLMM)
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",
]
RAYDIUM_ADDRESSES_PUBKEY = [Pubkey.from_string(_) for _ in RAYDIUM_ADDRESSES_STR]
WRAPPED_SOL_MINT = "So11111111111111111111111111111111111111112"
token_decimals = {}


def get_txns_for_address(
    address: str,
    limit: int = 1000,
    before: Optional[Signature] = None,
    until: Optional[Signature] = None,
) -> GetSignaturesForAddressResp:
    """Function to get the transactions for the address"""

    return solana_client.get_signatures_for_address(
        Pubkey.from_string(address), limit=limit, before=before, until=until
    )


def get_token_acct(acct: str) -> Pubkey:
    """Function to parse the data of token account
    and find the token program, owner of the account"""

    data = solana_client.get_account_info(Pubkey.from_string(acct)).value.data
    parsed_data = SPL_ACCOUNT_LAYOUT.parse(data)
    mint = Pubkey.from_bytes(parsed_data.mint)

    return mint


def get_token_decimals(mint: Pubkey) -> int:
    """Function to get the token details"""

    if mint in token_decimals:
        return token_decimals[mint]

    data = solana_client.get_account_info(mint).value.data
    parsed_data = SPL_MINT_LAYOUT.parse(data)
    token_decimals[mint] = parsed_data.decimals

    return parsed_data.decimals


def get_txn(
    sig: Signature,
) -> EncodedConfirmedTransactionWithStatusMeta | None:
    """Function to get the transaction details"""

    return solana_client.get_transaction(
        sig, encoding="jsonParsed", max_supported_transaction_version=0
    ).value


def parse_signer(
    txn: EncodedConfirmedTransactionWithStatusMeta | None,
) -> Pubkey:
    """Function to parse the signer of the transaction"""

    signer = [
        _ for _ in txn.transaction.transaction.message.account_keys if _.signer == True
    ]
    return signer[0].pubkey


def parse_instruction(
    txn: EncodedConfirmedTransactionWithStatusMeta | None,
) -> list:
    """Function to parse the instructions of the transaction"""

    return txn.transaction.transaction.message.instructions


def parse_inner_instruction(
    txn: EncodedConfirmedTransactionWithStatusMeta | None,
) -> list:
    """Function to parse the inner instructions of the transaction"""

    return txn.transaction.meta.inner_instructions


def parse_log_messages(
    txn: EncodedConfirmedTransactionWithStatusMeta | None,
) -> list:
    """Function to parse the log messages of the transaction"""

    return txn.transaction.meta.log_messages


def parse_token_info(
    token_info: ParsedInstruction,
    direction: Literal["in", "out", "mint"],
) -> tuple:
    """Function to parse the token info of the transaction"""

    if "tokenAmount" in token_info.parsed["info"]:
        token = token_info.parsed["info"]["mint"]
        amount = token_info.parsed["info"]["tokenAmount"]["uiAmount"]
    else:

        match direction:
            case "out":
                token = str(get_token_acct(token_info.parsed["info"]["destination"]))
            case "in":
                token = str(get_token_acct(token_info.parsed["info"]["source"]))
            case "mint":
                token = str(token_info.parsed["info"]["mint"])

        amount = float(token_info.parsed["info"]["amount"]) / 10 ** get_token_decimals(
            Pubkey.from_string(token)
        )

    return token, amount


def parse_simple_raydium_txn(instructions: list, inner_instructions: list) -> None:
    """Function to parse the simple raydium transaction"""

    raydium_instructions = [
        i
        for i, j in enumerate(instructions)
        if j.program_id in RAYDIUM_ADDRESSES_PUBKEY
    ]

    for instruction_index in raydium_instructions:
        raydium_inner_instructions = check_inner_instruction_index(
            inner_instructions, instruction_index
        )

        for inner_instructions in raydium_inner_instructions:
            # two inner instructions are the token info
            out_token_info, in_token_info = inner_instructions.instructions

            out_token, out_amount = parse_token_info(out_token_info, "out")
            in_token, in_amount = parse_token_info(in_token_info, "in")

            print(f"Swap {out_amount} {out_token} for {in_amount} {in_token}")


def parse_complex_raydium_txn(inner_instructions: list) -> None:
    """Function to parse the complex raydium transaction"""

    for inner_instruction in inner_instructions:
        raydium_instruction_indexes = check_instruction_program_id(
            inner_instruction.instructions, RAYDIUM_ADDRESSES_PUBKEY
        )

        if raydium_instruction_indexes:
            # the following two inner instructions are the token info
            for raydium_instruction_index in raydium_instruction_indexes:
                out_otken_info = inner_instruction.instructions[
                    raydium_instruction_index + 1
                ]
                in_token_info = inner_instruction.instructions[
                    raydium_instruction_index + 2
                ]

                out_token, out_amount = parse_token_info(out_otken_info, "out")
                in_token, in_amount = parse_token_info(in_token_info, "in")

                print(f"Swap {out_amount} {out_token} for {in_amount} {in_token}")


def parse_mint_burn(
    inner_instructions: list, type_name: Literal["mint", "burn"]
) -> None:
    """Function to parse the mint and burn transaction"""

    token0, token0_amount = parse_token_info(
        inner_instructions[-3], "out" if type_name == "mint" else "in"
    )
    token1, token1_amount = parse_token_info(
        inner_instructions[-2], "out" if type_name == "mint" else "in"
    )
    lp_token, lp_token_amount = parse_token_info(inner_instructions[-1], "mint")

    match type_name:
        case "mint":
            print(
                f"add liquidity: {token0_amount} {token0}, {token1_amount} {token1} for {lp_token_amount} {lp_token}"
            )
        case "burn":
            print(
                f"remove liquidity: {token0_amount} {token0}, {token1_amount} {token1} for {lp_token_amount} {lp_token}"
            )


def parse_raydium_txn(signature: Signature) -> None:
    """Function to parse the raydium transaction"""

    transaction = get_txn(signature)
    instructions = parse_instruction(transaction)
    inner_instructions = parse_inner_instruction(transaction)

    raydium_instructions_indexes = check_instruction_program_id(
        instructions, RAYDIUM_ADDRESSES_PUBKEY
    )

    # if there is raydium instruction
    if raydium_instructions_indexes:
        for radium_instruction_index in raydium_instructions_indexes:
            # if there is mintTo sub-instruction
            if check_inner_instruction_type(inner_instructions, "mintTo"):

                # all mints rely on the instruction index
                raydium_inner_instructions = check_inner_instruction_index(
                    inner_instructions, radium_instruction_index
                )

                # if there is new pool created
                if check_inner_instruction_type(inner_instructions, "initializeMint"):
                    for raydium_inner_instruction in raydium_inner_instructions:
                        parse_mint_burn(raydium_inner_instruction.instructions, "mint")
                # if there is regular liquidity added
                else:
                    for raydium_inner_instruction in raydium_inner_instructions:
                        parse_mint_burn(raydium_inner_instruction.instructions, "mint")
            elif check_inner_instruction_type(inner_instructions, "burn"):
                for inner_instruction in inner_instructions:
                    parse_mint_burn(inner_instruction.instructions, "burn")
            else:
                parse_simple_raydium_txn(instructions, inner_instructions)
    else:
        # if there is no raydium instruction
        for inner_instruction in inner_instructions:
            raydium_inner_instructions_index = check_instruction_program_id(
                inner_instruction.instructions, RAYDIUM_ADDRESSES_PUBKEY
            )

            if raydium_inner_instructions_index:
                parse_complex_raydium_txn(inner_instructions)


def check_inner_instruction_type(inner_instructions: list, type_name: str) -> list:
    """Function to return the list of index of inner instruction of a specific type"""
    return [
        i
        for inner_instruction in inner_instructions
        for i, j in enumerate(
            [
                k
                for k in inner_instruction.instructions
                if isinstance(k, ParsedInstruction)
            ]
        )
        if j.parsed["type"] == type_name
    ]


def check_instruction_program_id(
    instructions: list, program_id: Iterable[Pubkey] | Pubkey
) -> list:
    """Function to return the list of index of instruction of a specific program id"""

    if isinstance(program_id, Pubkey):
        program_id = [program_id]

    return [i for i, j in enumerate(instructions) if j.program_id in program_id]


def check_inner_instruction_index(
    inner_instructions: list, instruction_index: int
) -> list:
    """Function to check the inner instruction index
    and directly return the list of inner instructions"""
    return [
        inner_instruction
        for inner_instruction in inner_instructions
        if inner_instruction.index == instruction_index
    ]


if __name__ == "__main__":
    pass
    # parse_raydium_txn(
    #     Signature.from_string(
    #         "2NgtxMCmqbXuDkaPLguZdWYkyEhVQgnAPWXD5JoT661H1K3NsaHmJDgphhTbyrC7FyK58kK22My2zaSKsFKcyWw2"
    #     )
    # )

    transaction = get_txn(
        Signature.from_string(
            "2eDiS9j4Pav9jLaA8uKRgYLWphJXMNCZfwD9RfUvqGziURo2mtTYMDntXNfKEoFuKN3Hp9NR5AEob2M8iNCPyUoB"
        )
    )
    instructions = parse_instruction(transaction)
    inner_instructions = parse_inner_instruction(transaction)
