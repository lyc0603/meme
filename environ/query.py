"""SQL Query"""

SWAPPER_QUERY = """SELECT *
FROM defi.ez_dex_swaps
WHERE swapper = '{swapper}'
ORDER BY block_timestamp ASC;"""

UNCON_SWAP_QUERY = """SELECT *
FROM defi.ez_dex_swaps
WHERE (
         swap_from_mint = '{token_address}'
      OR swap_to_mint = '{token_address}'
      )
ORDER BY block_timestamp ASC;"""

UNCON_TRANSFER_QUERY = """SELECT *
FROM core.fact_transfers
WHERE (
         mint = '{token_address}'
      )
ORDER BY block_timestamp ASC;"""

SWAP_QUERY = """SELECT *
FROM defi.ez_dex_swaps
WHERE (
         swap_from_mint = '{token_address}'
      OR swap_to_mint = '{token_address}'
      )
  AND block_timestamp < TIMESTAMP '{migration_timestamp}' + INTERVAL '12 hours'
ORDER BY block_timestamp ASC;"""

TRANSFER_QUERY = """SELECT *
FROM core.fact_transfers
WHERE (
         mint = '{token_address}'
      )
  AND block_timestamp < TIMESTAMP '{migration_timestamp}' + INTERVAL '12 hours'
ORDER BY block_timestamp ASC;"""

LAUNCH_QUERY = """SELECT
  block_id AS launch_block_id,
  block_timestamp AS launch_time,
  tx_id AS launch_tx_id,
  decoded_instruction:accounts[0]:pubkey::string AS token_address,
  signers[0] AS token_creator,
  decoded_instruction:accounts[2]:pubkey::string AS pumpfun_pool_address
FROM
  core.ez_events_decoded
WHERE
  program_id = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
  AND event_type = 'create'
  AND block_timestamp > TIMESTAMP '{timestamp}'
ORDER BY block_timestamp
LIMIT {num};"""

MIGRATION_QUERY = """SELECT
  mig.block_id AS block_id, 
  mig.block_timestamp AS block_timestamp,
  mig.tx_id AS tx_id,
  mig.token_address,
  mig.sol_lamports,
  mig.meme_amount,
  lau.block_id AS launch_block_id,
  lau.block_timestamp AS launch_time,
  lau.tx_id AS launch_tx_id,
  lau.token_creator,
  lau.pumpfun_pool_address
FROM (
  SELECT
    block_id,
    block_timestamp,
    tx_id,
    decoded_instruction:accounts[9]:pubkey::string AS token_address,
    decoded_instruction:args:initCoinAmount::number AS sol_lamports,
    decoded_instruction:args:initPcAmount::number AS meme_amount
  FROM
    core.ez_events_decoded
  WHERE
    signers[0] = '39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg'
    AND program_id = '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8'
    AND event_type = 'initialize2'
    AND succeeded
    AND decoded_instruction:accounts[8]:pubkey::string = 'So11111111111111111111111111111111111111112'
    AND block_timestamp > TIMESTAMP '{timestamp}'
  ORDER BY block_timestamp
  LIMIT {num}
) mig
LEFT JOIN (
  SELECT
    block_id,
    block_timestamp,
    tx_id,
    decoded_instruction:accounts[0]:pubkey::string AS token_address,
    signers[0] AS token_creator,
    decoded_instruction:accounts[2]:pubkey::string AS pumpfun_pool_address
  FROM
    core.ez_events_decoded
  WHERE
    program_id = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
    AND event_type = 'create'
) lau
ON mig.token_address = lau.token_address
ORDER BY block_timestamp;"""

LAUNCH_QUERY_TEMPLATE = """
SELECT
  block_id AS launch_block_id,
  block_timestamp AS launch_time,
  tx_id AS launch_tx_id,
  decoded_instruction:accounts[0]:pubkey::string AS token_address,
  signers[0] AS token_creator,
  decoded_instruction:accounts[2]:pubkey::string AS pumpfun_pool_address
FROM
  core.ez_events_decoded
WHERE
  program_id = '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'
  AND event_type = 'create'
  AND decoded_instruction:accounts[0]:pubkey::string IN ({address_list})
"""
