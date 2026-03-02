from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

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
    num_games: int,
) -> torch.Tensor:
    draws = torch.rand(
        (num_games, weekly_odds.shape[0], weekly_odds.shape[1]),
        device=weekly_odds.device,
    )
    return draws < weekly_odds.unsqueeze(0).clamp(min=0.0, max=1.0)


def _play_survivor_games_with_population(
    game_populations: torch.Tensor,
    weekly_log_odds: torch.Tensor,
    winner_masks_by_game_week: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched survivor simulation for all games.
    Returns winner masks [num_games, agents_per_game] and weeks played [num_games].
    """
    num_games, game_size, num_teams = game_populations.shape
    picks_used = torch.zeros(
        (num_games, game_size, num_teams),
        dtype=torch.bool,
        device=game_populations.device,
    )
    alive = torch.ones(
        (num_games, game_size),
        dtype=torch.bool,
        device=game_populations.device,
    )
    weeks_played = torch.zeros(
        (num_games,),
        dtype=torch.int64,
        device=game_populations.device,
    )

    for week_id in range(weekly_log_odds.shape[0]):
        alive_counts = alive.sum(dim=1)
        active_games = alive_counts > 1
        if not bool(active_games.any()):
            break
        round_start_alive = alive.clone()

        active_alive = alive & active_games.unsqueeze(1)
        active_alive_ids = active_alive.nonzero(as_tuple=False)
        logits = (
            game_populations[active_alive_ids[:, 0], active_alive_ids[:, 1]]
            + weekly_log_odds[week_id].unsqueeze(0)
        )
        logits = logits.masked_fill(
            picks_used[active_alive_ids[:, 0], active_alive_ids[:, 1]],
            -1e9,
        )
        pick_probs = torch.softmax(logits, dim=1)
        picks = torch.multinomial(pick_probs, num_samples=1).squeeze(1)

        picks_used[active_alive_ids[:, 0], active_alive_ids[:, 1], picks] = True
        survived = winner_masks_by_game_week[active_alive_ids[:, 0], week_id, picks]

        next_alive = alive.clone()
        next_alive[active_games] = False
        next_alive[active_alive_ids[:, 0], active_alive_ids[:, 1]] = survived

        surviving_counts = next_alive.sum(dim=1)
        zero_survivors = active_games & (surviving_counts == 0)
        if bool(zero_survivors.any()):
            # If everyone dies in the final played round, co-winners are round entrants.
            next_alive[zero_survivors] = round_start_alive[zero_survivors]

        alive = next_alive
        weeks_played[active_games] = week_id + 1

    return alive, weeks_played


def _clone_winners_to_game_size(
    game_populations: torch.Tensor,
    winner_masks: torch.Tensor,
    mutation_std: float,
) -> torch.Tensor:
    """
    Sample winners with replacement to refill each game to full size.
    """
    winner_counts = winner_masks.sum(dim=1)
    if bool((winner_counts == 0).any()):
        raise ValueError("Cannot clone from an empty winner set.")

    sampling_weights = winner_masks.to(dtype=torch.float32)
    game_size = game_populations.shape[1]
    parent_rows = torch.multinomial(
        sampling_weights,
        num_samples=game_size,
        replacement=True,
    )
    gather_ids = parent_rows.unsqueeze(-1).expand(-1, -1, game_populations.shape[2])
    offspring = torch.gather(game_populations, dim=1, index=gather_ids)
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


def _load_population_checkpoint(path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        if "population" not in payload:
            raise ValueError(
                f"Checkpoint {path} did not contain a 'population' tensor."
            )
        population = payload["population"]
        metadata = {k: v for k, v in payload.items() if k != "population"}
    else:
        population = payload

    if not isinstance(population, torch.Tensor):
        raise ValueError(
            f"Checkpoint {path} did not contain a valid tensor under 'population'."
        )
    if population.ndim != 2:
        raise ValueError(
            f"Checkpoint {path} population must be rank-2 [total_agents, num_teams], "
            f"found shape {tuple(population.shape)}."
        )

    return population.detach().clone().to(dtype=torch.float32), metadata


def _load_evolution_weights(path: str, cfg: EvolutionLoopConfig) -> torch.Tensor:
    population, _ = _load_population_checkpoint(path)

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
        game_populations = shuffled.view(num_games, cfg.agents_per_game, cfg.num_teams)
        winner_masks = _sample_winner_masks(
            weekly_odds=weekly_odds,
            num_games=num_games,
        )
        winner_rows_by_game, weeks_ended_by_game = _play_survivor_games_with_population(
            game_populations=game_populations,
            weekly_log_odds=weekly_log_odds,
            winner_masks_by_game_week=winner_masks,
        )

        offspring_populations = _clone_winners_to_game_size(
            game_populations=game_populations,
            winner_masks=winner_rows_by_game,
            mutation_std=cfg.mutation_std,
        )
        population = offspring_populations.reshape(cfg.total_agents, cfg.num_teams)

        winners_per_game = winner_rows_by_game.sum(dim=1)
        avg_winners = float(winners_per_game.float().mean().item())
        avg_week_game_ended = float(weeks_ended_by_game.float().mean().item())
        max_winners = int(winners_per_game.max().item())
        generation_summary = (
            f"Generation {generation_id + 1}/{cfg.num_generations}: "
            f"avg winners/game={avg_winners:.2f}, "
            f"avg week game ended={avg_week_game_ended:.2f}, "
            f"max={max_winners}, "
            f"new_agents={population.shape[0]}"
        )
        tqdm.write(generation_summary)
        generation_iter.set_postfix(
            avg_winners=f"{avg_winners:.2f}",
            avg_week_ended=f"{avg_week_game_ended:.2f}",
            max_winners=max_winners,
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


def _format_probs(
    probs: torch.Tensor,
    team_names: List[str],
    team_ids: List[int],
) -> str:
    return ", ".join(
        f"{team_names[team_id]}={float(probs[team_id]):.4f}" for team_id in team_ids
    )


def run_sample_agent_distribution(
    weights_path: str,
    schedule_csv_path: str,
    max_weeks: int,
    sample_agent_id: Optional[int],
    top_k: int,
    print_full_distribution: bool,
    odds_weight: float,
    seed: int,
) -> None:
    if top_k <= 0:
        raise ValueError("--sample-top-k must be > 0")
    if max_weeks <= 0:
        raise ValueError("--sample-max-weeks must be > 0")
    if not Path(weights_path).exists():
        raise ValueError(
            f"Weights file was not found: {weights_path}. "
            "Provide --load-weights-path to a saved evolution checkpoint."
        )

    torch.manual_seed(seed)
    population, metadata = _load_population_checkpoint(weights_path)
    total_agents, num_teams = int(population.shape[0]), int(population.shape[1])

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

    checkpoint_generation = metadata.get("generation_id")
    if checkpoint_generation is not None:
        print(f"Checkpoint generation: {checkpoint_generation}")
    print(f"Loaded weights from: {weights_path}")
    print(f"Population shape: agents={total_agents}, teams={num_teams}")
    print(f"Sampled agent id: {selected_agent_id}")
    print(f"Sampling seed: {seed}")

    schedule = load_schedule_from_csv(
        csv_path=schedule_csv_path,
        num_weeks=max_weeks,
        num_teams=num_teams,
    )
    weekly_odds = _matchup_rows_to_weekly_odds(
        matchup_table=schedule.feature_rows,
        num_weeks=max_weeks,
        num_teams=num_teams,
        device=torch.device("cpu"),
    )
    weekly_log_odds = torch.log(weekly_odds.clamp_min(1e-6)) * odds_weight
    agent_logits = population[selected_agent_id]
    picked_teams: set[int] = set()
    week_pick_generator = torch.Generator(device="cpu")
    week_pick_generator.manual_seed(seed + 1)

    print(
        f"Schedule: {schedule_csv_path} | weeks={max_weeks} | odds_weight={odds_weight:.3f}"
    )
    for week_id in range(max_weeks):
        logits = agent_logits + weekly_log_odds[week_id]
        if picked_teams and len(picked_teams) < num_teams:
            masked_logits = logits.clone()
            masked_logits[list(picked_teams)] = -1e9
            logits = masked_logits
        probs = torch.softmax(logits, dim=0)

        picked_team = int(
            torch.multinomial(probs, num_samples=1, generator=week_pick_generator).item()
        )
        picked_teams.add(picked_team)

        pick_name = schedule.team_names[picked_team]
        pick_prob = float(probs[picked_team])
        pick_win_odds = float(weekly_odds[week_id, picked_team])
        print(
            f"Week {week_id + 1:02d}: sampled_pick={pick_name} "
            f"(pick_prob={pick_prob:.4f}, win_odds={pick_win_odds:.4f})"
        )

        if print_full_distribution:
            team_ids = list(range(num_teams))
            print(f"  distribution: {_format_probs(probs, schedule.team_names, team_ids)}")
        else:
            k = min(top_k, num_teams)
            top_team_ids = torch.topk(probs, k=k).indices.tolist()
            print(f"  top_{k}: {_format_probs(probs, schedule.team_names, top_team_ids)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-game and evolutionary survivor simulations.")
    parser.add_argument(
        "--mode",
        choices=["single-game", "evolution-loop", "sample-agent"],
        default="evolution-loop",
    )

    # Single game mode args.
    parser.add_argument("--num-agents", type=int, default=1_000)
    parser.add_argument("--num-teams", type=int, default=32)
    parser.add_argument("--max-weeks", type=int, default=18)

    # Evolution loop mode args.
    parser.add_argument("--total-agents", type=int, default=10_000_000)
    parser.add_argument("--agents-per-game", type=int, default=1_000)
    parser.add_argument("--num-generations", type=int, default=100)
    parser.add_argument("--mutation-std", type=float, default=0.6)
    parser.add_argument("--odds-weight", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-weights-path", type=str, default=DEFAULT_EVOLUTION_WEIGHTS_PATH)
    parser.add_argument("--save-weights-path", type=str, default=DEFAULT_EVOLUTION_WEIGHTS_PATH)
    parser.add_argument("--checkpoint-every-generation", action="store_true")
    parser.add_argument("--schedule-csv-path", type=str, default=DEFAULT_SCHEDULE_CSV_PATH)

    # Sample-agent mode args.
    parser.add_argument("--sample-agent-id", type=int, default=None)
    parser.add_argument("--sample-top-k", type=int, default=5)
    parser.add_argument("--sample-max-weeks", type=int, default=18)
    parser.add_argument("--print-full-distribution", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "sample-agent":
        run_sample_agent_distribution(
            weights_path=args.load_weights_path,
            schedule_csv_path=args.schedule_csv_path,
            max_weeks=args.sample_max_weeks,
            sample_agent_id=args.sample_agent_id,
            top_k=args.sample_top_k,
            print_full_distribution=args.print_full_distribution,
            odds_weight=args.odds_weight,
            seed=args.seed,
        )
        return

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
