# [Project Name]

## Setup

```
git clone [HTTPS/SSH URL]
cd [project-name]
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
COINGLASS_SECRET="abc123"
KAIKO_API_KEY="abc123"
TALLY_API_KEY="xxx"
SNAPSHOT_API_KEY="aaa"
OPENAI_API_KEY="sk-xxx"
```

```
export $(cat .env | xargs)
```

# fetch snapshot spaces data

```
python scripts/fetch_spaces.py
```
