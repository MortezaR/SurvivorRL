from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm.auto import tqdm

from nfl_survivor_agent import (
    BareBonesPickerNet,
    Config as AgentFeatureConfig,
    featurize,
)
from survivor_schedule import (
    FeatureMatchupRow,
    load_schedule_from_csv,
    sample_weekly_winners,
)

DEFAULT_EVOLUTION_WEIGHTS_PATH = "checkpoints/evolution_population.pt"
DEFAULT_SCHEDULE_CSV_PATH = "cleaned_grid2.csv"


def _runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SurvivorConfig:
    num_agents: int = 1_000
    num_teams: int = 32
    max_weeks: int = 18


@dataclass
class EvolutionLoopConfig:
    total_agents: int = 1_000_000
    agents_per_game: int = 1_000
    num_teams: int = 32
    max_weeks: int = 18
    num_generations: int = 10
    mutation_std: float = 0.05
    odds_weight: float = 1.0
    seed: int = 42
    load_weights_path: Optional[str] = DEFAULT_EVOLUTION_WEIGHTS_PATH
    save_weights_path: Optional[str] = DEFAULT_EVOLUTION_WEIGHTS_PATH
    checkpoint_every_generation: bool = False
    schedule_csv_path: str = DEFAULT_SCHEDULE_CSV_PATH


@dataclass
class SurvivorGameResult:
    weeks_played: int
    winner_ids: List[int]
    survivors: List[int]
    survivors_by_week: List[List[int]]
    winning_teams_by_week: List[List[int]]
    eliminated_by_week: List[int]


class SurvivorEnvironment:
    def __init__(self, cfg: SurvivorConfig) -> None:
        self.cfg = cfg
        self.device = _runtime_device()

        self.feature_cfg = AgentFeatureConfig(
            max_contestants=cfg.num_agents,
            max_teams=cfg.num_teams,
            max_weeks=cfg.max_weeks,
        )
        sample_x = featurize(
            cfg=self.feature_cfg,
            contestant_picks={},
            matchup_table=[],
            agent_id=0,
            current_week=0,
        )
        input_dim = int(sample_x.numel())

        self.agents = [
            BareBonesPickerNet(input_dim=input_dim, num_teams=cfg.num_teams, agent_id=agent_id)
            for agent_id in range(cfg.num_agents)
        ]
        for model in self.agents:
            model.to(self.device)
            model.eval()

    def play_game(
        self,
        matchup_table: List[FeatureMatchupRow],
        winning_teams_by_week: List[List[int]],
    ) -> SurvivorGameResult:
        rows_by_week: List[List[FeatureMatchupRow]] = [[] for _ in range(self.cfg.max_weeks)]
        for week_id, team_id, win_prob in matchup_table:
            if 0 <= week_id < self.cfg.max_weeks:
                rows_by_week[week_id].append((week_id, team_id, win_prob))

        matchup_rows_seen: List[FeatureMatchupRow] = []
        active_agents = list(range(self.cfg.num_agents))
        picked_teams_by_agent = [set() for _ in range(self.cfg.num_agents)]
        survivors_by_week: List[List[int]] = []
        eliminated_by_week: List[int] = []

        with torch.no_grad():
            for week_id in range(self.cfg.max_weeks):
                if len(active_agents) <= 1:
                    break
                round_start_agents = active_agents.copy()

                matchup_rows_seen.extend(rows_by_week[week_id])
                week_winners = set()
                if week_id < len(winning_teams_by_week):
                    week_winners = set(winning_teams_by_week[week_id])

                next_active_agents: List[int] = []
                contestant_picks = {}
                for agent_id in active_agents:
                    prior_picks = picked_teams_by_agent[agent_id]
                    if len(prior_picks) >= self.cfg.num_teams:
                        continue

                    x = featurize(
                        cfg=self.feature_cfg,
                        contestant_picks=contestant_picks,
                        matchup_table=matchup_rows_seen,
                        agent_id=agent_id,
                        current_week=week_id,
                    ).to(self.device)
                    pick_probs = self.agents[agent_id](
                        x,
                        unavailable_team_ids=list(prior_picks),
                    )
                    picked_team = int(torch.multinomial(pick_probs, num_samples=1).item())
                    picked_teams_by_agent[agent_id].add(picked_team)
                    contestant_picks[agent_id] = picked_team
                    if picked_team in week_winners:
                        next_active_agents.append(agent_id)

                if len(next_active_agents) == 0 and len(round_start_agents) > 0:
                    # If everyone is eliminated in the final played round,
                    # treat all round entrants as co-winners.
                    active_agents = round_start_agents
                    eliminated_by_week.append(0)
                    survivors_by_week.append(active_agents.copy())
                    break

                eliminated_by_week.append(len(active_agents) - len(next_active_agents))
                active_agents = next_active_agents
                survivors_by_week.append(active_agents.copy())

        winner_ids = active_agents.copy()
        return SurvivorGameResult(
            weeks_played=len(survivors_by_week),
            winner_ids=winner_ids,
            survivors=active_agents,
            survivors_by_week=survivors_by_week,
            winning_teams_by_week=winning_teams_by_week,
            eliminated_by_week=eliminated_by_week,
        )


def _matchup_rows_to_weekly_odds(
    matchup_table: List[FeatureMatchupRow],
    num_weeks: int,
    num_teams: int,
    device: torch.device,
) -> torch.Tensor:
    weekly_odds = torch.zeros((num_weeks, num_teams), dtype=torch.float32, device=device)
    for week_id, team_id, win_prob in matchup_table:
        if 0 <= week_id < num_weeks and 0 <= team_id < num_teams:
            weekly_odds[week_id, team_id] = float(win_prob)
    return weekly_odds


def _sample_winner_masks(
    weekly_odds: torch.Tensor,
) -> torch.Tensor:
    draws = torch.rand(weekly_odds.shape, device=weekly_odds.device)
    return draws < weekly_odds.clamp(min=0.0, max=1.0)


def _play_survivor_game_with_population(
    game_population: torch.Tensor,
    weekly_log_odds: torch.Tensor,
    winner_masks_by_week: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """
    Lightweight survivor simulation for one game population.
    Returns winner row indices into `game_population` and number of played weeks.
    """
    game_size, num_teams = game_population.shape
    picks_used = torch.zeros((game_size, num_teams), dtype=torch.bool, device=game_population.device)
    alive = torch.ones((game_size,), dtype=torch.bool, device=game_population.device)
    weeks_played = 0

    for week_id in range(weekly_log_odds.shape[0]):
        alive_ids = alive.nonzero(as_tuple=False).flatten()
        if alive_ids.numel() <= 1:
            break
        round_start_alive = alive_ids

        logits = game_population[alive_ids] + weekly_log_odds[week_id].unsqueeze(0)
        logits = logits.masked_fill(picks_used[alive_ids], -1e9)
        pick_probs = torch.softmax(logits, dim=1)
        picks = torch.multinomial(pick_probs, num_samples=1).squeeze(1)
        picks_used[alive_ids, picks] = True
        weeks_played = week_id + 1

        survived = winner_masks_by_week[week_id, picks]
        if not bool(survived.any()):
            # If everyone dies in the final played round, co-winners are round entrants.
            return round_start_alive, weeks_played

        alive[:] = False
        alive[alive_ids[survived]] = True

    return alive.nonzero(as_tuple=False).flatten(), weeks_played


def _clone_winners_to_game_size(
    winners: torch.Tensor,
    game_size: int,
    mutation_std: float,
) -> torch.Tensor:
    num_winners = int(winners.shape[0])
    if num_winners == 0:
        raise ValueError("Cannot clone from an empty winner set.")

    base_children = game_size // num_winners
    counts = torch.full(
        (num_winners,),
        base_children,
        dtype=torch.long,
        device=winners.device,
    )
    remainder = game_size - int(counts.sum().item())
    if remainder > 0:
        bonus_ids = torch.randperm(num_winners, device=winners.device)[:remainder]
        counts[bonus_ids] += 1

    offspring = winners.repeat_interleave(counts, dim=0)
    mutation_noise = torch.randn(
        offspring.shape,
        dtype=offspring.dtype,
        device=offspring.device,
    ) * mutation_std
    return offspring + mutation_noise


def _checkpoint_path_for_generation(base_path: str, generation_id: int) -> Path:
    path = Path(base_path)
    suffix = path.suffix if path.suffix else ".pt"
    return path.with_name(f"{path.stem}_gen{generation_id}{suffix}")


def _save_evolution_weights(
    path: str,
    population: torch.Tensor,
    cfg: EvolutionLoopConfig,
    generation_id: int,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "population": population.detach().cpu(),
        "generation_id": generation_id,
        "total_agents": cfg.total_agents,
        "agents_per_game": cfg.agents_per_game,
        "num_teams": cfg.num_teams,
        "max_weeks": cfg.max_weeks,
        "mutation_std": cfg.mutation_std,
        "odds_weight": cfg.odds_weight,
        "seed": cfg.seed,
        "schedule_csv_path": cfg.schedule_csv_path,
    }
    torch.save(payload, checkpoint_path)


def _load_evolution_weights(path: str, cfg: EvolutionLoopConfig) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "population" in payload:
        population = payload["population"]
    else:
        population = payload

    if not isinstance(population, torch.Tensor):
        raise ValueError(
            f"Checkpoint {path} did not contain a valid tensor under 'population'."
        )

    expected_shape = (cfg.total_agents, cfg.num_teams)
    if tuple(population.shape) != expected_shape:
        raise ValueError(
            f"Checkpoint shape mismatch for {path}: expected {expected_shape}, "
            f"found {tuple(population.shape)}"
        )
    return population.detach().clone().to(dtype=torch.float32)


def run_evolution_loop(cfg: EvolutionLoopConfig) -> None:
    if cfg.total_agents <= 0:
        raise ValueError("--total-agents must be > 0")
    if cfg.agents_per_game <= 1:
        raise ValueError("--agents-per-game must be > 1")
    if cfg.total_agents % cfg.agents_per_game != 0:
        raise ValueError("--total-agents must be divisible by --agents-per-game")
    if cfg.num_generations <= 0:
        raise ValueError("--num-generations must be > 0")
    if cfg.num_teams <= 1:
        raise ValueError("--num-teams must be > 1")
    if cfg.max_weeks <= 0:
        raise ValueError("--max-weeks must be > 0")

    num_games = cfg.total_agents // cfg.agents_per_game
    device = _runtime_device()
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    schedule = load_schedule_from_csv(
        csv_path=cfg.schedule_csv_path,
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
    )
    weekly_odds = _matchup_rows_to_weekly_odds(
        matchup_table=schedule.feature_rows,
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
        device=device,
    )
    weekly_log_odds = torch.log(weekly_odds.clamp_min(1e-6)) * cfg.odds_weight

    # Each agent is a compact genome: one learnable preference score per team.
    if cfg.load_weights_path and Path(cfg.load_weights_path).exists():
        population = _load_evolution_weights(cfg.load_weights_path, cfg=cfg).to(device=device)
        print(f"Loaded evolution weights from: {cfg.load_weights_path}")
    else:
        if cfg.load_weights_path:
            print(
                f"No saved weights found at {cfg.load_weights_path}. "
                "Starting from random initialization."
            )
        population = torch.randn((cfg.total_agents, cfg.num_teams), device=device)

    print(
        "Starting loop: "
        f"{cfg.total_agents} agents -> {num_games} games x {cfg.agents_per_game} agents/game "
        f"on {device.type}"
    )
    generation_iter = tqdm(
        range(cfg.num_generations),
        desc="Generations",
        unit="gen",
    )
    for generation_id in generation_iter:
        shuffled = population[torch.randperm(cfg.total_agents, device=device)]
        next_population = torch.empty_like(shuffled)
        winners_per_game: List[int] = []
        weeks_ended_per_game: List[int] = []

        game_iter = tqdm(
            range(num_games),
            desc=f"Generation {generation_id + 1}/{cfg.num_generations} games",
            unit="game",
            leave=False,
        )
        for game_id in game_iter:
            start = game_id * cfg.agents_per_game
            end = start + cfg.agents_per_game
            game_population = shuffled[start:end]

            winner_masks = _sample_winner_masks(
                weekly_odds=weekly_odds,
            )
            winner_rows, weeks_played = _play_survivor_game_with_population(
                game_population=game_population,
                weekly_log_odds=weekly_log_odds,
                winner_masks_by_week=winner_masks,
            )
            winners_per_game.append(int(winner_rows.numel()))
            weeks_ended_per_game.append(weeks_played)

            offspring = _clone_winners_to_game_size(
                winners=game_population[winner_rows],
                game_size=cfg.agents_per_game,
                mutation_std=cfg.mutation_std,
            )
            next_population[start:end] = offspring

        population = next_population
        avg_winners = sum(winners_per_game) / len(winners_per_game)
        avg_week_game_ended = sum(weeks_ended_per_game) / len(weeks_ended_per_game)
        generation_summary = (
            f"Generation {generation_id + 1}/{cfg.num_generations}: "
            f"avg winners/game={avg_winners:.2f}, "
            f"avg week game ended={avg_week_game_ended:.2f}, "
            f"max={max(winners_per_game)}, "
            f"new_agents={population.shape[0]}"
        )
        tqdm.write(generation_summary)
        generation_iter.set_postfix(
            avg_winners=f"{avg_winners:.2f}",
            avg_week_ended=f"{avg_week_game_ended:.2f}",
            max_winners=max(winners_per_game),
        )
        if cfg.checkpoint_every_generation and cfg.save_weights_path:
            checkpoint_path = _checkpoint_path_for_generation(
                cfg.save_weights_path,
                generation_id=generation_id + 1,
            )
            _save_evolution_weights(
                str(checkpoint_path),
                population=population,
                cfg=cfg,
                generation_id=generation_id + 1,
            )
            tqdm.write(f"Saved generation checkpoint: {checkpoint_path}")

    if cfg.save_weights_path:
        _save_evolution_weights(
            cfg.save_weights_path,
            population=population,
            cfg=cfg,
            generation_id=cfg.num_generations,
        )
        print(f"Saved final evolution weights to: {cfg.save_weights_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-game and evolutionary survivor simulations.")
    parser.add_argument(
        "--mode",
        choices=["single-game", "evolution-loop"],
        default="evolution-loop",
    )

    # Single game mode args.
    parser.add_argument("--num-agents", type=int, default=1_000)
    parser.add_argument("--num-teams", type=int, default=32)
    parser.add_argument("--max-weeks", type=int, default=18)

    # Evolution loop mode args.
    parser.add_argument("--total-agents", type=int, default=1_000_000)
    parser.add_argument("--agents-per-game", type=int, default=1_000)
    parser.add_argument("--num-generations", type=int, default=10000)
    parser.add_argument("--mutation-std", type=float, default=0.05)
    parser.add_argument("--odds-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-weights-path", type=str, default=DEFAULT_EVOLUTION_WEIGHTS_PATH)
    parser.add_argument("--save-weights-path", type=str, default=DEFAULT_EVOLUTION_WEIGHTS_PATH)
    parser.add_argument("--checkpoint-every-generation", action="store_true")
    parser.add_argument("--schedule-csv-path", type=str, default=DEFAULT_SCHEDULE_CSV_PATH)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "evolution-loop":
        loop_cfg = EvolutionLoopConfig(
            total_agents=args.total_agents,
            agents_per_game=args.agents_per_game,
            num_teams=args.num_teams,
            max_weeks=args.max_weeks,
            num_generations=args.num_generations,
            mutation_std=args.mutation_std,
            odds_weight=args.odds_weight,
            seed=args.seed,
            load_weights_path=args.load_weights_path,
            save_weights_path=args.save_weights_path,
            checkpoint_every_generation=args.checkpoint_every_generation,
            schedule_csv_path=args.schedule_csv_path,
        )
        run_evolution_loop(loop_cfg)
        return

    cfg = SurvivorConfig(
        num_agents=args.num_agents,
        num_teams=args.num_teams,
        max_weeks=args.max_weeks,
    )

    # Matchups and game winners are loaded and sampled from CSV outside the environment.
    schedule = load_schedule_from_csv(
        csv_path=args.schedule_csv_path,
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
    )
    winning_teams_by_week = sample_weekly_winners(
        matchup_table=schedule.feature_rows,
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
    )

    env = SurvivorEnvironment(cfg)
    print(f"Running single game on {env.device.type}")
    result = env.play_game(
        matchup_table=schedule.feature_rows,
        winning_teams_by_week=winning_teams_by_week,
    )

    print(f"Weeks played: {result.weeks_played}")
    print(f"Num winners: {len(result.winner_ids)}")
    for week_index, survivors in enumerate(result.survivors_by_week):
        print(f"Week {week_index} survivors: {survivors}")


if __name__ == "__main__":
    main()
