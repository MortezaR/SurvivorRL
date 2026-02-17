# SurvivorRL

`NFLSurvivorEnv` is a Gymnasium environment for simulating an NFL survivor pool with a variable number of contestants.

## Features

- Variable number of contestants (`num_contestants`).
- Observation includes:
  - weekly game list,
  - matchup win probabilities,
  - all contestants,
  - each contestant's prior picks.
- Action is each alive contestant's team selection.
- Weekly winners are simulated from matchup probabilities.
- Contestants cannot reuse teams selected in prior weeks.
- Terminal reward pool equals `num_contestants - 1`, split evenly among remaining winners.

## Install

```bash
pip install gymnasium numpy
```

## Example

```python
from nfl_survivor_env import NFLSurvivorEnv

teams = ["KC", "BUF", "PHI", "DAL"]
weekly_schedule = [
    [("KC", "BUF"), ("PHI", "DAL")],
    [("KC", "PHI"), ("BUF", "DAL")],
]
probability_chart = [
    {("KC", "BUF"): 0.55, ("PHI", "DAL"): 0.60},
    {("KC", "PHI"): 0.50, ("BUF", "DAL"): 0.58},
]

env = NFLSurvivorEnv(
    teams=teams,
    weekly_schedule=weekly_schedule,
    probability_chart=probability_chart,
    num_contestants=3,
)

obs, info = env.reset(seed=7)

# Pick by team names (string) or by team index (int).
action = {
    "agent_0": "KC",
    "agent_1": "BUF",
    "agent_2": "PHI",
}
obs, rewards, terminated, truncated, info = env.step(action)
```
