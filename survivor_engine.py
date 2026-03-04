from __future__ import annotations

import random
from typing import List, Tuple

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
) -> List[PickerNet]:

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
            return survivors
        active_agents = survivors

    return active_agents


def replicate_winners(
    winners: List[PickerNet],
    num_contestants: int,
    noise_std: float = 0.01,
) -> List[PickerNet]:

    base_copies = num_contestants // len(winners)
    remainder = num_contestants % len(winners)

    replicated: List[PickerNet] = []
    for winner_idx, winner in enumerate(winners):
        num_copies = base_copies + (1 if winner_idx < remainder else 0)
        for _ in range(num_copies):
            child = PickerNet.mutated_copy(winner, std=noise_std)
            replicated.append(child)

    return replicated


def evo_loop(
    all_agents: List[PickerNet],
    num_loops: int,
    num_contestants: int,
    noise_std: float = 0.01,
) -> List[PickerNet]:

    for _ in range(num_loops):

        shuffled = list(all_agents)
        random.shuffle(shuffled)
        new_all_agents: List[PickerNet] = []

        for start in range(0, len(shuffled), num_contestants):
            game_agents = shuffled[start:start + num_contestants]
            sampled_winning_teams = sample_weekly_winners(weekly_probs)
            winners = survivor_game(game_agents, sampled_winning_teams)

            replicated = replicate_winners(
                winners=winners,
                num_contestants=num_contestants,
                noise_std=noise_std,
            )
            new_all_agents.extend(replicated)

        for new_id, agent in enumerate(new_all_agents):
            agent.agent_id = new_id

        all_agents = new_all_agents

    return all_agents
        



if __name__ == "__main__":
    pass
