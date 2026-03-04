# SurvivorRL

Minimal Survivor pool prototype.

Current files:

- agent model definition in `survivor_agent.py`
- CSV schedule loading in `survivor_schedule.py`
- basic orchestration in `survivor_engine.py`
- source data in `cleaned_grid2.csv`

There is currently no CLI file (`survivor_cli.py` does not exist).

## Setup

```bash
cd /Users/morteza/Desktop/SurvivorRL
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 survivor_engine.py
```

The current engine module defines:

- `weekly_probs`: list of weeks, each containing `(team_name, win_probability)` tuples
- `survivor_agents`: list of `BareBonesPickerNet` agents

## CSV Format

Expected columns:

- `Team`
- week columns named `1`, `2`, `3`, ... up to `num_weeks`

Probabilities can be either:

- decimals in `[0, 1]`
- percentages like `64.3` (automatically converted to `0.643`)
