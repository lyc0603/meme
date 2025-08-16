# Meme

## 1. Configuration

- Clone the repository

```
git clone https://github.com/lyc0603/meme.git
cd meme
```

- Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

## Set up the environmental variables

- Rename the `.env.example` file into `.env`.
- Put your Snowflake and OpenAI APIs in `.env`:

```
SNOWFLAKE_USER = "..."
OPENAI_API = "..."
```

## 2. Fetch Pumpfun Data

- Fetch Pre-Trump data

```
python scripts_fin/fetch_pumpfun.py \
    --category pre_trump_raydium \
    --num 1000 \
    --timestamp "2024-10-17 14:01:48"
```
```
python scripts_fin/fetch_pumpfun.py \
    --category pre_trump_pumpfun \
    --num 3000 \
    --timestamp "2024-10-17 14:01:48"
```

- Fetch Post-Trump data

```
python scripts_fin/fetch_pumpfun.py \
    --category raydium \
    --num 1000 \
    --timestamp "2025-01-17 14:01:48"
```


```
python scripts_fin/fetch_pumpfun.py \
    --category pumpfun \
    --num 3000 \
    --timestamp "2025-01-17 14:01:48"
```

## 3. Process Raw Pumpfun Data

```
python scripts_fin/process_pumpfun.py
```


