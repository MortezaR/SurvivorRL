from __future__ import annotations

from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from survivor_agent import PickerNet, Config
from survivor_schedule import load_schedule_from_csv

num_agents = 100
num_contestants = 10
num_weeks = 18
num_teams = 32
config = Config(num_contestants, num_teams, num_weeks)

schedule = load_schedule_from_csv("cleaned_grid2.csv",num_weeks,num_teams)

weekly_probs = [[] for _ in range(num_weeks)]
for week_id, team_id, win_prob in schedule.feature_rows:
    team_name = schedule.team_names[team_id]
    weekly_probs[week_id].append((team_name, win_prob))

POPULATION_SIZE_FILENAME = "population_size.txt"


def _agent_checkpoint_path(population_dir: Path, agent_index: int) -> Path:
    return population_dir / f"agent_{agent_index:06d}.pt"


def _remove_existing_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return

    if path.exists():
        path.unlink()


def _prepare_population_dir(population_dir: Path) -> None:
    if population_dir.exists():
        _remove_existing_path(population_dir)
    population_dir.mkdir(parents=True, exist_ok=True)


def _state_dict_to_cpu(state_dict: Dict[str, object]) -> Dict[str, object]:
    cpu_state_dict: Dict[str, object] = {}
    for name, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state_dict[name] = value.detach().cpu()
        else:
            cpu_state_dict[name] = value
    return cpu_state_dict


def _write_population_size(population_dir: Path, population_size: int) -> None:
    (population_dir / POPULATION_SIZE_FILENAME).write_text(f"{population_size}\n")


def get_population_store_size(population_dir: str | Path) -> int:
    return int((Path(population_dir) / POPULATION_SIZE_FILENAME).read_text().strip())


def save_agent_checkpoint(
    agent: PickerNet,
    population_dir: str | Path,
    agent_index: int,
) -> None:
    population_path = Path(population_dir)
    payload = {
        "agent_id": agent.agent_id,
        "state_dict": _state_dict_to_cpu(agent.state_dict()),
    }
    torch.save(payload, _agent_checkpoint_path(population_path, agent_index))


def load_agent_checkpoint(
    population_dir: str | Path,
    agent_index: int,
    cfg: Config,
    device: torch.device,
) -> PickerNet:
    population_path = Path(population_dir)
    payload = torch.load(_agent_checkpoint_path(population_path, agent_index), map_location="cpu")
    agent_id = payload.get("agent_id", agent_index)
    agent = PickerNet(agent_id=agent_id, cfg=cfg)
    agent.load_state_dict(payload["state_dict"])
    agent.to(device)
    return agent


def initialize_random_population_store(
    population_dir: str | Path,
    cfg: Config,
    population_size: int,
) -> None:
    population_path = Path(population_dir)
    _prepare_population_dir(population_path)

    for agent_idx in range(population_size):
        agent = PickerNet(agent_id=agent_idx, cfg=cfg)
        save_agent_checkpoint(agent, population_path, agent_idx)

    _write_population_size(population_path, population_size)


def load_population_checkpoint_into_store(
    checkpoint_path: str,
    population_dir: str | Path,
) -> None:
    checkpoint = Path(checkpoint_path)
    population_path = Path(population_dir)
    _prepare_population_dir(population_path)

    if checkpoint.is_dir():
        shutil.copytree(checkpoint, population_path, dirs_exist_ok=True)
        return

    payload = torch.load(checkpoint, map_location="cpu")
    state_dicts = payload["state_dicts"]
    agent_ids = payload.get("agent_ids", list(range(len(state_dicts))))

    for idx, state_dict in enumerate(state_dicts):
        agent_id = agent_ids[idx] if idx < len(agent_ids) else idx
        torch.save(
            {"agent_id": agent_id, "state_dict": _state_dict_to_cpu(state_dict)},
            _agent_checkpoint_path(population_path, idx),
        )

    _write_population_size(population_path, len(state_dicts))


def save_population_store(
    population_dir: str | Path,
    output_path: str,
) -> None:
    population_path = Path(population_dir)
    path = Path(output_path)

    if path.suffix != ".pt":
        if path.exists():
            _remove_existing_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(population_path, path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    population_size = get_population_store_size(population_path)
    state_dicts = []
    agent_ids = []

    for agent_idx in range(population_size):
        payload = torch.load(_agent_checkpoint_path(population_path, agent_idx), map_location="cpu")
        state_dicts.append(payload["state_dict"])
        agent_ids.append(payload.get("agent_id", agent_idx))

    torch.save({"state_dicts": state_dicts, "agent_ids": agent_ids}, path)


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
                    unavailable_team_ids=prior_picks,
                )
                picked_team_id = int(torch.multinomial(pick_dist, num_samples=1).item())

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

    if Path(output_path).suffix != ".pt":
        population_path = Path(output_path)
        _prepare_population_dir(population_path)

        for idx, agent in enumerate(population):
            agent.agent_id = idx
            save_agent_checkpoint(agent, population_path, idx)

        _write_population_size(population_path, len(population))
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dicts": [_state_dict_to_cpu(agent.state_dict()) for agent in population],
        "agent_ids": [agent.agent_id for agent in population],
    }
    torch.save(payload, path)


def load_population_weights(weights_path: str, cfg: Config) -> List[PickerNet]:

    weights = Path(weights_path)
    if weights.is_dir():
        population_size = get_population_store_size(weights)
        return [
            load_agent_checkpoint(weights, agent_idx, cfg, torch.device("cpu"))
            for agent_idx in range(population_size)
        ]

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
    population_dir: str | Path,
    cfg: Config,
    num_loops: int,
    num_contestants: int,
    device: torch.device,
    noise_std: float = 0.01,
    work_dir: str | Path | None = None,
) -> Path:

    if num_contestants <= 0:
        raise ValueError("num_contestants must be > 0")

    current_population_dir = Path(population_dir)
    generation_root = Path(work_dir) if work_dir is not None else current_population_dir.parent

    with torch.inference_mode():
        loop_iter = tqdm(range(num_loops), desc="Evo loops")

        for loop_idx in loop_iter:
            population_size = get_population_store_size(current_population_dir)
            shuffled_indices = list(range(population_size))
            random.shuffle(shuffled_indices)
            next_population_dir = generation_root / f"generation_{loop_idx + 1:04d}"
            _prepare_population_dir(next_population_dir)
            next_agent_id = 0
            game_week_lengths: List[int] = []

            generation = tqdm(
                range(0, population_size, num_contestants),
                desc=f"Loop {loop_idx + 1}/{num_loops} generation",
                leave=False,
            )

            for start in generation:
                game_agent_indices = shuffled_indices[start:start + num_contestants]
                game_agents = [
                    load_agent_checkpoint(current_population_dir, agent_idx, cfg, device)
                    for agent_idx in game_agent_indices
                ]
                sampled_winning_teams = sample_weekly_winners(weekly_probs)
                winners, weeks_played = survivor_game(game_agents, sampled_winning_teams)
                game_week_lengths.append(weeks_played)

                replicated = replicate_winners(
                    winners=winners,
                    num_contestants=len(game_agent_indices),
                    noise_std=noise_std,
                )

                for agent in replicated:
                    agent.agent_id = next_agent_id
                    save_agent_checkpoint(agent, next_population_dir, next_agent_id)
                    next_agent_id += 1

                del game_agents, winners, replicated

            _write_population_size(next_population_dir, next_agent_id)
            avg_week_length = sum(game_week_lengths) / len(game_week_lengths)
            print(
                f"Loop {loop_idx + 1}/{num_loops} average game length: "
                f"{avg_week_length:.2f} weeks"
            )

            if current_population_dir != Path(population_dir):
                shutil.rmtree(current_population_dir)

            current_population_dir = next_population_dir

    return current_population_dir
        



if __name__ == "__main__":
    pass
