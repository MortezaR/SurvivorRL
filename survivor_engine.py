from __future__ import annotations

import random
from typing import List, Tuple, Union

import torch

from survivor_agent import PickerNet, Config
from survivor_schedule import load_schedule_from_csv

num_agents = 1_000
num_contestants = 1_000
num_weeks = 18
num_teams = 32
config = Config(num_contestants, num_teams, num_weeks)

all_agents = [PickerNet( agent_id, config) for agent_id in range(num_agents)]

schedule = load_schedule_from_csv("cleaned_grid2.csv",num_weeks,num_teams)

weekly_probs = [[] for _ in range(num_weeks)]
for week_id, team_id, win_prob in schedule.feature_rows:
    team_name = schedule.team_names[team_id]
    weekly_probs[week_id].append((team_name, win_prob))


def sample_weekly_winners(
    weekly_probs: List[List[Tuple[str, float]]],
) -> List[Tuple[int, List[int]]]:
    """
    For each week, sample winners using each team's win probability.

    Returns:
      [(week_id, [winning_team_ids]), ...]
    """
    sampled: List[Tuple[int, List[int]]] = []
    for week_id, week_probs in enumerate(weekly_probs):
        week_winners: List[int] = []
        for team_id, (_, win_prob) in enumerate(week_probs):
            if random.random() < win_prob:
                week_winners.append(team_id)
        sampled.append((week_id, week_winners))
    return sampled


def survivor_game(
    agents: List[PickerNet],
    sampled_winning_teams: List[Tuple[int, List[int]]],
) -> Union[PickerNet, List[PickerNet]]:

    active_agents = list(agents)
    contestant_picks = {}
    for week_idx in range(num_weeks):
        last_week_agents = list(active_agents)
        contestant_picks_upto_last_week = contestant_picks.copy()
        winning_team_ids = set(sampled_winning_teams[week_idx][1])

        survivors: List[PickerNet] = []
        for agent in active_agents:
            pick_dist = agent(
                contestant_picks=contestant_picks_upto_last_week,
                matchup_table=schedule.feature_rows,
                current_week=week_idx,
            )
            picked_team_id = int(torch.multinomial(pick_dist, num_samples=1).item())

            if picked_team_id in winning_team_ids:
                survivors.append(agent)
                contestant_picks[agent.agent_id] = picked_team_id
            else:
                contestant_picks.pop(agent.agent_id, None)

        if len(survivors) == 0:
            return last_week_agents
        if len(survivors) == 1:
            return survivors[0]
        active_agents = survivors

    return active_agents


def replicate_winners(
    winners: List[PickerNet],
    num_contestants: int,
    noise_std: float = 0.01,
) -> List[PickerNet]:
    """
    Replicate winners proportionally to fill `num_contestants`.

    Each replica is a mutated copy produced by PickerNet.mutated_copy.
    Returns exactly `num_contestants` mutated agents.
    """
    if num_contestants <= 0:
        return []
    if len(winners) == 0:
        raise ValueError("replicate_winners requires at least one winner.")

    base_copies = num_contestants // len(winners)
    remainder = num_contestants % len(winners)

    replicated: List[PickerNet] = []
    next_agent_id = 0
    for winner_idx, winner in enumerate(winners):
        num_copies = base_copies + (1 if winner_idx < remainder else 0)
        for _ in range(num_copies):
            child = PickerNet.mutated_copy(winner, std=noise_std)
            child.agent_id = next_agent_id
            replicated.append(child)
            next_agent_id += 1

    return replicated
# def
# 
# def evo_loop():
#     winners = []
#     for week in range(num_weeks):
        



if __name__ == "__main__":
    pass
