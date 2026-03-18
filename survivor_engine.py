from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from survivor_agent import PickerNet, Config
from survivor_schedule import load_schedule_from_csv

num_agents = 1000000
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


@dataclass
class GameWorkItem:
    game_idx: int
    agents: List[PickerNet]
    sampled_winning_teams: List[Tuple[int, List[int]]]


@dataclass
class GameWorkResult:
    game_idx: int
    replicated_agents: List[PickerNet]
    weeks_played: int


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
        contestant_picks = {}  # {agent_id: [picked_team_id, ...]} for active agents only
        weeks_played = 0
        for week_idx in range(num_weeks):
            last_week_agents = list(active_agents)
            contestant_picks_upto_last_week = {
                cid: picks.copy() for cid, picks in contestant_picks.items()
            }
            winning_team_ids = set(sampled_winning_teams[week_idx][1])

            survivors: List[PickerNet] = []
            next_contestant_picks = {}
            mu_table = schedule.feature_rows
            for agent in active_agents:
                prior_picks = contestant_picks_upto_last_week.get(agent.agent_id, [])
                pick_dist = agent(
                    contestant_picks=contestant_picks_upto_last_week,
                    matchup_table=mu_table,
                    current_week=week_idx,
                    num_players=len(active_agents),
                    unavailable_team_ids=prior_picks,
                )
                picked_team_id = int(
                    torch.multinomial(pick_dist, num_samples=1).item()
                )

                if picked_team_id in winning_team_ids:
                    survivors.append(agent)
                    next_contestant_picks[agent.agent_id] = prior_picks + [picked_team_id]

            weeks_played = week_idx + 1
            if len(survivors) == 0:
                return last_week_agents, weeks_played
            if len(survivors) == 1:
                return survivors, weeks_played
            contestant_picks = next_contestant_picks
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


def _population_device(population: List[PickerNet]) -> torch.device:

    if not population:
        return torch.device("cpu")
    return next(population[0].parameters()).device


def _run_game_work_item(
    work_item: GameWorkItem,
    num_contestants: int,
    noise_std: float,
    device: torch.device,
) -> GameWorkResult:
    game_agents = move_population_to_device(work_item.agents, device)

    if device.type == "cuda":
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream), torch.inference_mode():
                winners, weeks_played = survivor_game(
                    game_agents,
                    work_item.sampled_winning_teams,
                )
                replicated_agents = replicate_winners(
                    winners=winners,
                    num_contestants=num_contestants,
                    noise_std=noise_std,
                )
            stream.synchronize()
    else:
        with torch.inference_mode():
            winners, weeks_played = survivor_game(
                game_agents,
                work_item.sampled_winning_teams,
            )
            replicated_agents = replicate_winners(
                winners=winners,
                num_contestants=num_contestants,
                noise_std=noise_std,
            )

    return GameWorkResult(
        game_idx=work_item.game_idx,
        replicated_agents=replicated_agents,
        weeks_played=weeks_played,
    )


class GameDispatcher:
    def __init__(
        self,
        devices: List[torch.device],
        max_workers: Optional[int] = None,
    ) -> None:

        if not devices:
            raise ValueError("GameDispatcher requires at least one dispatch device.")
        self.devices = [torch.device(device) for device in devices]
        has_cuda = any(device.type == "cuda" for device in self.devices)
        default_workers = 2 * len(self.devices) if has_cuda else 1
        self.max_workers = default_workers if max_workers is None else max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="game-dispatcher",
        )

    def run_generation(
        self,
        work_items: List[GameWorkItem],
        num_contestants: int,
        noise_std: float,
        desc: str,
    ) -> List[GameWorkResult]:

        if not work_items:
            return []

        futures = [
            self._executor.submit(
                _run_game_work_item,
                work_item,
                num_contestants,
                noise_std,
                self.devices[work_item.game_idx % len(self.devices)],
            )
            for work_item in work_items
        ]
        ordered_results: List[Optional[GameWorkResult]] = [None] * len(work_items)
        generation = tqdm(total=len(futures), desc=desc, leave=False)

        try:
            for future in as_completed(futures):
                result = future.result()
                ordered_results[result.game_idx] = result
                generation.update(1)
        except Exception:
            for future in futures:
                future.cancel()
            raise
        finally:
            generation.close()

        return [result for result in ordered_results if result is not None]

    def close(self) -> None:

        self._executor.shutdown(wait=True)


def evo_loop(
    all_agents: List[PickerNet],
    num_loops: int,
    num_contestants: int,
    noise_std: float = 0.01,
    game_workers: Optional[int] = None,
    dispatch_devices: Optional[List[torch.device]] = None,
) -> List[PickerNet]:

    if num_contestants <= 0:
        raise ValueError("num_contestants must be > 0")
    if game_workers is not None and game_workers < 1:
        raise ValueError("game_workers must be >= 1 when provided.")
    if dispatch_devices is not None and not dispatch_devices:
        raise ValueError("dispatch_devices must not be empty when provided.")

    def _run_loop(
        population: List[PickerNet],
    ) -> List[PickerNet]:
        devices = (
            [torch.device(device) for device in dispatch_devices]
            if dispatch_devices is not None
            else [_population_device(population)]
        )
        dispatcher = GameDispatcher(
            devices=devices,
            max_workers=game_workers,
        )
        try:
            loop_iter = tqdm(range(num_loops), desc="Evo loops")

            for loop_idx in loop_iter:
                shuffled = list(population)
                random.shuffle(shuffled)
                work_items: List[GameWorkItem] = []

                for game_idx, start in enumerate(range(0, len(shuffled), num_contestants)):
                    game_agents = shuffled[start:start + num_contestants]
                    work_items.append(
                        GameWorkItem(
                            game_idx=game_idx,
                            agents=game_agents,
                            sampled_winning_teams=sample_weekly_winners(weekly_probs),
                        )
                    )

                results = dispatcher.run_generation(
                    work_items=work_items,
                    num_contestants=num_contestants,
                    noise_std=noise_std,
                    desc=f"Loop {loop_idx + 1}/{num_loops} generation",
                )

                new_all_agents: List[PickerNet] = []
                game_week_lengths: List[int] = []
                for result in results:
                    game_week_lengths.append(result.weeks_played)
                    new_all_agents.extend(result.replicated_agents)

                for new_id, agent in enumerate(new_all_agents):
                    agent.agent_id = new_id

                avg_week_length = sum(game_week_lengths) / len(game_week_lengths)
                print(
                    f"Loop {loop_idx + 1}/{num_loops} average game length: "
                    f"{avg_week_length:.2f} weeks"
                )

                population = new_all_agents

            return population
        finally:
            dispatcher.close()

    return _run_loop(all_agents)
        



if __name__ == "__main__":
    pass
