from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from survivor_agent import Config as AgentFeatureConfig
from survivor_checkpoints import load_population_checkpoint
from survivor_engine import (
    _feature_input_dim,
    _load_genome_into_model,
    _matchup_rows_to_weekly_odds,
    _new_picker_model,
    _rows_by_week,
)
from survivor_schedule import (
    FeatureMatchupRow,
    load_schedule_from_csv,
)


@dataclass
class SampledWeekDistribution:
    week_id: int
    sampled_team_id: int
    sampled_team_name: str
    sampled_pick_probability: float
    sampled_team_win_odds: float
    probabilities: List[float]


@dataclass
class SampleAgentDistributionResult:
    checkpoint_generation: Optional[int]
    weights_path: str
    total_agents: int
    num_teams: int
    selected_agent_id: int
    schedule_csv_path: str
    max_weeks: int
    team_names: List[str]
    weeks: List[SampledWeekDistribution]


def sample_agent_distribution(
    weights_path: str,
    schedule_csv_path: str,
    max_weeks: int,
    sample_agent_id: Optional[int],
) -> SampleAgentDistributionResult:
    if max_weeks <= 0:
        raise ValueError("--sample-max-weeks must be > 0")
    if not Path(weights_path).exists():
        raise ValueError(
            f"Weights file was not found: {weights_path}. "
            "Provide --load-weights-path to a saved evolution checkpoint."
        )

    population_genomes, metadata = load_population_checkpoint(weights_path)
    total_agents = len(population_genomes)
    if total_agents == 0:
        raise ValueError(f"Checkpoint {weights_path} has an empty population_genomes list.")

    checkpoint_num_teams = int(metadata.get("num_teams", 32))

    selected_agent_id: int
    if sample_agent_id is None:
        selected_agent_id = int(torch.randint(0, total_agents, (1,)).item())
    else:
        if not 0 <= sample_agent_id < total_agents:
            raise ValueError(
                f"--sample-agent-id must be in [0, {total_agents - 1}], "
                f"found {sample_agent_id}."
            )
        selected_agent_id = sample_agent_id

    checkpoint_generation_raw = metadata.get("generation_id")
    checkpoint_generation: Optional[int]
    if isinstance(checkpoint_generation_raw, int):
        checkpoint_generation = checkpoint_generation_raw
    else:
        checkpoint_generation = None

    feature_max_contestants_raw = metadata.get(
        "feature_max_contestants",
        metadata.get("agents_per_game", 1),
    )
    if not isinstance(feature_max_contestants_raw, int) or feature_max_contestants_raw <= 0:
        raise ValueError(
            f"Checkpoint {weights_path} has invalid feature_max_contestants={feature_max_contestants_raw}."
        )
    feature_max_contestants = feature_max_contestants_raw

    checkpoint_max_weeks_raw = metadata.get("max_weeks", max_weeks)
    if not isinstance(checkpoint_max_weeks_raw, int) or checkpoint_max_weeks_raw <= 0:
        raise ValueError(
            f"Checkpoint {weights_path} has invalid max_weeks={checkpoint_max_weeks_raw}."
        )
    checkpoint_max_weeks = checkpoint_max_weeks_raw

    if max_weeks > checkpoint_max_weeks:
        raise ValueError(
            f"--sample-max-weeks={max_weeks} exceeds checkpoint max weeks ({checkpoint_max_weeks})."
        )

    feature_cfg = AgentFeatureConfig(
        max_contestants=feature_max_contestants,
        max_teams=checkpoint_num_teams,
        max_weeks=checkpoint_max_weeks,
    )
    input_dim = _feature_input_dim(feature_cfg)
    model = _new_picker_model(
        input_dim=input_dim,
        num_teams=checkpoint_num_teams,
        feature_cfg=feature_cfg,
        agent_id=0,
        device=torch.device("cpu"),
    )
    _load_genome_into_model(model, population_genomes[selected_agent_id])
    model.eval()

    schedule = load_schedule_from_csv(
        csv_path=schedule_csv_path,
        num_weeks=checkpoint_max_weeks,
        num_teams=checkpoint_num_teams,
    )
    rows_by_week = _rows_by_week(
        matchup_table=schedule.feature_rows,
        max_weeks=checkpoint_max_weeks,
    )
    weekly_odds = _matchup_rows_to_weekly_odds(
        matchup_table=schedule.feature_rows,
        num_weeks=checkpoint_max_weeks,
        num_teams=checkpoint_num_teams,
        device=torch.device("cpu"),
    )

    picked_teams: set[int] = set()
    matchup_rows_seen: List[FeatureMatchupRow] = []
    week_distributions: List[SampledWeekDistribution] = []

    with torch.no_grad():
        for week_id in range(max_weeks):
            matchup_rows_seen.extend(rows_by_week[week_id])
            probs = model(
                contestant_picks={},
                matchup_table=matchup_rows_seen,
                current_week=week_id,
                unavailable_team_ids=list(picked_teams),
            )
            picked_team = int(torch.multinomial(probs, num_samples=1).item())
            picked_teams.add(picked_team)

            week_distributions.append(
                SampledWeekDistribution(
                    week_id=week_id + 1,
                    sampled_team_id=picked_team,
                    sampled_team_name=schedule.team_names[picked_team],
                    sampled_pick_probability=float(probs[picked_team]),
                    sampled_team_win_odds=float(weekly_odds[week_id, picked_team]),
                    probabilities=probs.tolist(),
                )
            )

    return SampleAgentDistributionResult(
        checkpoint_generation=checkpoint_generation,
        weights_path=weights_path,
        total_agents=total_agents,
        num_teams=checkpoint_num_teams,
        selected_agent_id=selected_agent_id,
        schedule_csv_path=schedule_csv_path,
        max_weeks=max_weeks,
        team_names=schedule.team_names,
        weeks=week_distributions,
    )
