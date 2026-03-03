# SurvivorRL

This repo currently includes a bare-bones PyTorch model (`survivor_agent.py`) for generating team-pick distributions in an NFL survivor pool setup.

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

## Evolution loop (1,000,000 agents)

`survivor_cli.py` supports a large-scale evolutionary loop:

- randomize `1,000,000` agents into `1,000` games of `1,000`
- run each game
- clone winners directly so each game always outputs `1,000` offspring
- repeat for the configured number of generations

By default, evolution mode now auto-loads and auto-saves population weights at:

- `checkpoints/evolution_population.pt`

If that file does not exist yet, the run starts from random weights and creates it at the end.

Note: checkpoints now store per-agent parameter vectors under `population_genomes` and are not compatible with older checkpoint formats.

Example:

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --total-agents 1000000 \
  --agents-per-game 1000 \
  --num-generations 2 \
  --num-teams 32 \
  --max-weeks 18
```

Save final evolution weights:

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --num-generations 50 \
  --save-weights-path checkpoints/evo_weights.pt
```

Resume from saved weights:

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --load-weights-path checkpoints/evo_weights.pt \
  --num-generations 50
```

Write a checkpoint at every generation (`..._genN.pt`) plus the final file:

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --save-weights-path checkpoints/evo_weights.pt \
  --checkpoint-every-generation
```

## Sample an agent from saved weights

You can load a saved evolution checkpoint, sample one agent from the population, and print that agent's team-pick probability distribution at each week.

Top-5 per week:

```bash
python3 survivor_cli.py \
  --mode sample-agent \
  --load-weights-path checkpoints/evolution_population.pt \
  --schedule-csv-path cleaned_grid2.csv \
  --sample-max-weeks 18 \
  --sample-top-k 5
```

Use a specific agent id and print the full per-team distribution:

```bash
python3 survivor_cli.py \
  --mode sample-agent \
  --load-weights-path checkpoints/evolution_population.pt \
  --sample-agent-id 1234 \
  --print-full-distribution
```
