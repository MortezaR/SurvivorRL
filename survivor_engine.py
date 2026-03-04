from __future__ import annotations

from pathlib import Path
import random
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from survivor_agent import PickerNet, Config
from survivor_schedule import load_schedule_from_csv

num_agents = 1000
num_contestants = 100
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
) -> Tuple[List[PickerNet], int]:

    with torch.inference_mode():
        active_agents = list(agents)
        contestant_picks = {}
        weeks_played = 0
        for week_idx in range(num_weeks):
            last_week_agents = list(active_agents)
            contestant_picks_upto_last_week = contestant_picks.copy()
            winning_team_ids = set(sampled_winning_teams[week_idx][1])

            survivors: List[PickerNet] = []
            mu_table = schedule.feature_rows
            for agent in active_agents:
                pick_dist = agent(
                    contestant_picks=contestant_picks_upto_last_week,
                    matchup_table=mu_table,
                    current_week=week_idx,
                )
                picked_team_id = int(torch.multinomial(pick_dist, num_samples=1).item())

                if picked_team_id in winning_team_ids:
                    survivors.append(agent)
                    contestant_picks[agent.agent_id] = picked_team_id
                else:
                    contestant_picks.pop(agent.agent_id, None)

            weeks_played = week_idx + 1
            if len(survivors) == 0:
                return last_week_agents, weeks_played
            if len(survivors) == 1:
                return survivors, weeks_played
            active_agents = survivors

        return active_agents, weeks_played


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
def save_population_weights(population: List[PickerNet], output_path: str) -> None:

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dicts": [agent.state_dict() for agent in population],
        "agent_ids": [agent.agent_id for agent in population],
    }
    torch.save(payload, path)


def load_population_weights(weights_path: str, cfg: Config) -> List[PickerNet]:

    payload = torch.load(weights_path, map_location="cpu")
    state_dicts = payload["state_dicts"]
    agent_ids = payload.get("agent_ids", list(range(len(state_dicts))))

    population: List[PickerNet] = []
    for idx, state_dict in enumerate(state_dicts):
        agent_id = agent_ids[idx] if idx < len(agent_ids) else idx
        agent = PickerNet(agent_id=agent_id, cfg=cfg)
        agent.load_state_dict(state_dict)
        population.append(agent)

    return population


def move_population_to_device(population: List[PickerNet], device: torch.device) -> List[PickerNet]:

    for agent in population:
        agent.to(device)
    return population


def evo_loop(
    all_agents: List[PickerNet],
    num_loops: int,
    num_contestants: int,
    noise_std: float = 0.01,
    profile: bool = False,
    profile_output_path: Optional[str] = None,
) -> List[PickerNet]:

    if num_contestants <= 0:
        raise ValueError("num_contestants must be > 0")

    def _run_loop(
        population: List[PickerNet],
        profiler: Optional[torch.profiler.profile] = None,
    ) -> List[PickerNet]:
        with torch.inference_mode():
            loop_iter = tqdm(range(num_loops), desc="Evo loops")

            for loop_idx in loop_iter:
                shuffled = list(population)
                random.shuffle(shuffled)
                new_all_agents: List[PickerNet] = []
                game_week_lengths: List[int] = []

                generation = tqdm(
                    range(0, len(shuffled), num_contestants),
                    desc=f"Loop {loop_idx + 1}/{num_loops} generation",
                    leave=False,
                )

                for start in generation:
                    game_agents = shuffled[start:start + num_contestants]
                    sampled_winning_teams = sample_weekly_winners(weekly_probs)
                    winners, weeks_played = survivor_game(game_agents, sampled_winning_teams)
                    game_week_lengths.append(weeks_played)

                    replicated = replicate_winners(
                        winners=winners,
                        num_contestants=num_contestants,
                        noise_std=noise_std,
                    )
                    new_all_agents.extend(replicated)

                    if profiler is not None:
                        profiler.step()

                for new_id, agent in enumerate(new_all_agents):
                    agent.agent_id = new_id

                avg_week_length = sum(game_week_lengths) / len(game_week_lengths)
                print(
                    f"Loop {loop_idx + 1}/{num_loops} average game length: "
                    f"{avg_week_length:.2f} weeks"
                )

                population = new_all_agents

            return population

    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(activities=activities) as profiler:
            final_population = _run_loop(all_agents, profiler=profiler)

        if profile_output_path:
            trace_path = Path(profile_output_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
    else:
        final_population = _run_loop(all_agents, profiler=None)

    return final_population
        



if __name__ == "__main__":
    pass
