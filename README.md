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
- `numpy`

## Evolution loop

`survivor_cli.py` supports an evolutionary loop:

- randomize agents into games
- run each game
- clone winners directly so each game always outputs `agents_per_game` offspring
- repeat for the configured number of generations

The CLI defaults are intentionally small (`200` total agents, `20` per game, `20` generations) so a no-arg run is a practical smoke test. Scale up with explicit flags as needed.

By default, evolution mode now auto-loads and auto-saves population weights at:

- `checkpoints/evolution_population.pt`

If that file does not exist yet, the run starts from random weights and creates it at the end.

Note: checkpoints now store per-agent parameter vectors under `population_genomes` and are not compatible with older checkpoint formats.

Example:

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --total-agents 2000 \
  --agents-per-game 20 \
  --num-generations 2 \
  --device auto \
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
  --device auto \
  --num-generations 50
```

Write a checkpoint at every generation (`..._genN.pt`) plus the final file:

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --save-weights-path checkpoints/evo_weights.pt \
  --checkpoint-every-generation
```

Profile where compute time and memory are spent (CPU + CUDA ops table and Chrome trace):

```bash
python3 survivor_cli.py \
  --mode evolution-loop \
  --device cuda \
  --total-agents 2000 \
  --agents-per-game 20 \
  --num-generations 2 \
  --profile \
  --profile-output-dir profiles \
  --profile-warmup-games 1 \
  --profile-active-games 10
```

When profiling is enabled, the CLI prints a run-specific output folder containing:

- `trace.json` for Chrome/Perfetto timeline view
- `cpu_time_top_ops.txt`
- `cuda_time_top_ops.txt` (when CUDA activity is captured)
- `cuda_memory_top_ops.txt` (when CUDA activity is captured)

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
