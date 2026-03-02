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
- `tqdm`

## Bare-bones survivor environment

`bare_bones_survivor_env.py` runs a simple survivor game:

- builds `1,000` copies of `BareBonesPickerNet` (one model per agent)
- derives each model input size from `featurize(...)` and uses featurized state each week
- loads matchup odds from `cleaned_grid2.csv` (team x week table in the repo)
- samples weekly team winners from those per-team odds
- eliminates agents who picked the wrong team
- stops when one or more winners remain at season end (`max_weeks`) or everyone is eliminated

Run with defaults:

```bash
python3 bare_bones_survivor_env.py
```

Use a different CSV schedule:

```bash
python3 bare_bones_survivor_env.py --schedule-csv-path path/to/schedule.csv
```

## Evolution loop (1,000,000 agents)

`bare_bones_survivor_env.py` also supports a large-scale evolutionary loop:

- randomize `1,000,000` agents into `1,000` games of `1,000`
- run each game
- clone winners with Gaussian mutation so each game always outputs `1,000` offspring
- repeat for the configured number of generations

By default, evolution mode now auto-loads and auto-saves population weights at:

- `checkpoints/evolution_population.pt`

If that file does not exist yet, the run starts from random weights and creates it at the end.

Example:

```bash
python3 bare_bones_survivor_env.py \
  --mode evolution-loop \
  --total-agents 1000000 \
  --agents-per-game 1000 \
  --num-generations 2 \
  --num-teams 32 \
  --max-weeks 18
```

Save final evolution weights:

```bash
python3 bare_bones_survivor_env.py \
  --mode evolution-loop \
  --num-generations 50 \
  --save-weights-path checkpoints/evo_weights.pt
```

Resume from saved weights:

```bash
python3 bare_bones_survivor_env.py \
  --mode evolution-loop \
  --load-weights-path checkpoints/evo_weights.pt \
  --num-generations 50
```

Write a checkpoint at every generation (`..._genN.pt`) plus the final file:

```bash
python3 bare_bones_survivor_env.py \
  --mode evolution-loop \
  --save-weights-path checkpoints/evo_weights.pt \
  --checkpoint-every-generation
```

## Sample an agent from saved weights

You can load a saved evolution checkpoint, sample one agent from the population, and print that agent's team-pick probability distribution at each week.

Top-5 per week:

```bash
python3 bare_bones_survivor_env.py \
  --mode sample-agent \
  --load-weights-path checkpoints/evolution_population.pt \
  --schedule-csv-path cleaned_grid2.csv \
  --sample-max-weeks 18 \
  --sample-top-k 5 \
  --seed 42
```

Use a specific agent id and print the full per-team distribution:

```bash
python3 bare_bones_survivor_env.py \
  --mode sample-agent \
  --load-weights-path checkpoints/evolution_population.pt \
  --sample-agent-id 1234 \
  --print-full-distribution
```
