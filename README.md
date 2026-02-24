# SurvivorRL

This repo currently includes a bare-bones PyTorch model (`nfl_survivor_agent.py`) for generating team-pick distributions in an NFL survivor pool setup.

## Quick start (virtual environment)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

Dependencies are managed in `requirements.txt`:

- `torch`

## Bare-bones survivor environment

`bare_bones_survivor_env.py` runs a simple survivor game:

- builds `1,000` copies of `BareBonesPickerNet` (one model per agent)
- derives each model input size from `featurize(...)` and uses featurized state each week
- receives externally generated matchups and externally sampled game winners (see `survivor_schedule.py`)
- sample schedule uses round-robin opponents (no repeats within cycle) and pairwise odds summing to `1.0`
- each game has one winner, so each week has multiple winning teams
- eliminates agents who picked the wrong team
- stops when one or more winners remain at season end (`max_weeks`) or everyone is eliminated

Run with defaults:

```bash
python3 bare_bones_survivor_env.py
```
