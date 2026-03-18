from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from survivor_agent import PickerNet, Config
from survivor_schedule import load_schedule_from_csv

num_agents = 10000
num_contestants = 10
num_weeks = 18
num_teams = 32
config = Config(num_contestants, num_teams, num_weeks)

schedule = load_schedule_from_csv("cleaned_grid2.csv", num_weeks, num_teams)

weekly_probs = [[] for _ in range(num_weeks)]
for week_id, team_id, win_prob in schedule.feature_rows:
    team_name = schedule.team_names[team_id]
    weekly_probs[week_id].append((team_name, win_prob))


def _empty_parameter_tensors(cfg: Config) -> Dict[str, torch.Tensor]:

    template_state = PickerNet(agent_id=0, cfg=cfg).state_dict()
    return {
        name: tensor.new_empty((0, *tensor.shape))
        for name, tensor in template_state.items()
    }


@dataclass
class PopulationStore:
    cfg: Config
    parameter_tensors: Dict[str, torch.Tensor]
    agent_ids: torch.Tensor

    def __post_init__(self) -> None:

        self.agent_ids = self.agent_ids.to(dtype=torch.long, device="cpu")
        for name, tensor in self.parameter_tensors.items():
            self.parameter_tensors[name] = tensor.detach().to(device="cpu")

    def __len__(self) -> int:

        return int(self.agent_ids.numel())

    @classmethod
    def empty(cls, cfg: Config) -> PopulationStore:

        return cls(
            cfg=cfg,
            parameter_tensors=_empty_parameter_tensors(cfg),
            agent_ids=torch.empty(0, dtype=torch.long),
        )

    @classmethod
    def initialize(cls, num_agents: int, cfg: Config) -> PopulationStore:

        if num_agents < 0:
            raise ValueError("num_agents must be >= 0")
        if num_agents == 0:
            return cls.empty(cfg)

        template_state = PickerNet(agent_id=0, cfg=cfg).state_dict()
        parameter_tensors = {
            name: torch.empty((num_agents, *tensor.shape), dtype=tensor.dtype)
            for name, tensor in template_state.items()
        }
        agent_ids = torch.arange(num_agents, dtype=torch.long)

        init_iter = tqdm(
            range(num_agents),
            desc="Initializing population",
            leave=False,
        )
        for agent_id in init_iter:
            agent = PickerNet(agent_id=agent_id, cfg=cfg)
            state_dict = agent.state_dict()
            for name, tensor in state_dict.items():
                parameter_tensors[name][agent_id].copy_(tensor.detach())

        return cls(
            cfg=cfg,
            parameter_tensors=parameter_tensors,
            agent_ids=agent_ids,
        )

    @classmethod
    def from_agents(
        cls,
        agents: List[PickerNet],
        cfg: Optional[Config] = None,
    ) -> PopulationStore:

        if not agents:
            if cfg is None:
                raise ValueError("cfg is required to build an empty population store.")
            return cls.empty(cfg)

        resolved_cfg = cfg or agents[0].cfg
        first_state = agents[0].state_dict()
        parameter_tensors = {
            name: torch.empty((len(agents), *tensor.shape), dtype=tensor.dtype)
            for name, tensor in first_state.items()
        }
        agent_ids = torch.empty(len(agents), dtype=torch.long)

        for idx, agent in enumerate(agents):
            agent_ids[idx] = agent.agent_id
            state_dict = agent.state_dict()
            for name, tensor in state_dict.items():
                parameter_tensors[name][idx].copy_(tensor.detach().to(device="cpu"))

        return cls(
            cfg=resolved_cfg,
            parameter_tensors=parameter_tensors,
            agent_ids=agent_ids,
        )

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Dict[str, object],
        cfg: Config,
    ) -> PopulationStore:

        if "parameter_tensors" in payload:
            raw_parameter_tensors = payload["parameter_tensors"]
            if not isinstance(raw_parameter_tensors, dict):
                raise ValueError("Invalid checkpoint: parameter_tensors must be a dict.")

            parameter_tensors = {
                name: tensor.detach().to(device="cpu")
                for name, tensor in raw_parameter_tensors.items()
            }
            first_tensor = next(iter(parameter_tensors.values()), None)
            population_size = first_tensor.shape[0] if first_tensor is not None else 0
            raw_agent_ids = payload.get("agent_ids", torch.arange(population_size))
            agent_ids = torch.as_tensor(raw_agent_ids, dtype=torch.long).clone()
            return cls(
                cfg=cfg,
                parameter_tensors=parameter_tensors,
                agent_ids=agent_ids,
            )

        raw_state_dicts = payload.get("state_dicts")
        if raw_state_dicts is None:
            raise ValueError("Checkpoint did not contain a supported population format.")

        state_dicts = list(raw_state_dicts)
        if not state_dicts:
            return cls.empty(cfg)

        first_state = state_dicts[0]
        parameter_tensors = {
            name: torch.empty((len(state_dicts), *tensor.shape), dtype=tensor.dtype)
            for name, tensor in first_state.items()
        }
        for idx, state_dict in enumerate(state_dicts):
            for name, tensor in state_dict.items():
                parameter_tensors[name][idx].copy_(tensor.detach().to(device="cpu"))

        raw_agent_ids = payload.get("agent_ids", list(range(len(state_dicts))))
        agent_ids = torch.as_tensor(raw_agent_ids, dtype=torch.long).clone()
        return cls(
            cfg=cfg,
            parameter_tensors=parameter_tensors,
            agent_ids=agent_ids,
        )

    @staticmethod
    def concatenate(
        populations: List[PopulationStore],
        cfg: Config,
    ) -> PopulationStore:

        non_empty = [population for population in populations if len(population) > 0]
        if not non_empty:
            return PopulationStore.empty(cfg)

        parameter_names = list(non_empty[0].parameter_tensors.keys())
        parameter_tensors = {
            name: torch.cat(
                [population.parameter_tensors[name] for population in non_empty],
                dim=0,
            )
            for name in parameter_names
        }
        agent_ids = torch.cat(
            [population.agent_ids for population in non_empty],
            dim=0,
        )

        return PopulationStore(
            cfg=cfg,
            parameter_tensors=parameter_tensors,
            agent_ids=agent_ids,
        )

    def checkpoint_payload(self) -> Dict[str, object]:

        return {
            "format": "population_store_v1",
            "parameter_tensors": self.parameter_tensors,
            "agent_ids": self.agent_ids,
        }

    def materialize_agents(
        self,
        indices: List[int],
        device: torch.device,
    ) -> List[PickerNet]:

        agents: List[PickerNet] = []
        for population_idx in indices:
            agent_id = int(self.agent_ids[population_idx].item())
            agent = PickerNet(agent_id=agent_id, cfg=self.cfg)
            agent.to(device)
            state_dict = {
                name: tensor[population_idx].to(device=device)
                for name, tensor in self.parameter_tensors.items()
            }
            agent.load_state_dict(state_dict)
            agents.append(agent)
        return agents

    def reset_agent_ids(self) -> None:

        self.agent_ids = torch.arange(len(self), dtype=torch.long)


@dataclass
class GameWorkItem:
    game_idx: int
    agent_indices: List[int]
    sampled_winning_teams: List[Tuple[int, List[int]]]


@dataclass
class GameWorkResult:
    game_idx: int
    population_shard: PopulationStore
    weeks_played: int


def create_initial_population(
    num_agents: int,
    cfg: Config,
) -> PopulationStore:

    return PopulationStore.initialize(num_agents=num_agents, cfg=cfg)


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


def save_population_weights(population: PopulationStore, output_path: str) -> None:

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(population.checkpoint_payload(), path)


def load_population_weights(weights_path: str, cfg: Config) -> PopulationStore:

    payload = torch.load(weights_path, map_location="cpu")
    return PopulationStore.from_checkpoint_payload(payload, cfg)


def _run_game_work_item(
    work_item: GameWorkItem,
    population: PopulationStore,
    num_contestants: int,
    noise_std: float,
    device: torch.device,
) -> GameWorkResult:

    game_agents = population.materialize_agents(work_item.agent_indices, device)

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
        population_shard=PopulationStore.from_agents(
            replicated_agents,
            cfg=population.cfg,
        ),
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
        population: PopulationStore,
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
                population,
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
    population: PopulationStore,
    num_loops: int,
    num_contestants: int,
    noise_std: float = 0.01,
    game_workers: Optional[int] = None,
    dispatch_devices: Optional[List[torch.device]] = None,
) -> PopulationStore:

    if num_contestants <= 0:
        raise ValueError("num_contestants must be > 0")
    if game_workers is not None and game_workers < 1:
        raise ValueError("game_workers must be >= 1 when provided.")
    if dispatch_devices is not None and not dispatch_devices:
        raise ValueError("dispatch_devices must not be empty when provided.")

    def _run_loop(current_population: PopulationStore) -> PopulationStore:
        devices = (
            [torch.device(device) for device in dispatch_devices]
            if dispatch_devices is not None
            else [torch.device("cpu")]
        )
        dispatcher = GameDispatcher(
            devices=devices,
            max_workers=game_workers,
        )
        try:
            loop_iter = tqdm(range(num_loops), desc="Evo loops")

            for loop_idx in loop_iter:
                shuffled_indices = list(range(len(current_population)))
                random.shuffle(shuffled_indices)
                work_items: List[GameWorkItem] = []

                for game_idx, start in enumerate(range(0, len(shuffled_indices), num_contestants)):
                    game_indices = shuffled_indices[start:start + num_contestants]
                    work_items.append(
                        GameWorkItem(
                            game_idx=game_idx,
                            agent_indices=game_indices,
                            sampled_winning_teams=sample_weekly_winners(weekly_probs),
                        )
                    )

                results = dispatcher.run_generation(
                    population=current_population,
                    work_items=work_items,
                    num_contestants=num_contestants,
                    noise_std=noise_std,
                    desc=f"Loop {loop_idx + 1}/{num_loops} generation",
                )

                game_week_lengths = [result.weeks_played for result in results]
                current_population = PopulationStore.concatenate(
                    [result.population_shard for result in results],
                    cfg=current_population.cfg,
                )
                current_population.reset_agent_ids()

                avg_week_length = sum(game_week_lengths) / len(game_week_lengths)
                print(
                    f"Loop {loop_idx + 1}/{num_loops} average game length: "
                    f"{avg_week_length:.2f} weeks"
                )

            return current_population
        finally:
            dispatcher.close()

    return _run_loop(population)


if __name__ == "__main__":
    pass
