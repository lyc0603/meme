# Meme

## Setup

```
git clone https://github.com/lyc0603/meme.git
cd meme
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

## Set up the environmental variables

put your APIs in `.env`:

```
INFURA_API_KEYS = "XXX,XXX,XXX"
```

```
export $(cat .env | xargs)
```

# fetch historical meme data

```
python scripts/fetch_meme_data.py --chain <Chain Name>
```

# fetch timestamp

```
python scripts/fetch_timestamp.py --chain <Chain Name>
```

# plot meme data

```
python scripts/process_meme_data.py
python scripts/plot_historical_meme.py
```

