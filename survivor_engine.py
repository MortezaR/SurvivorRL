from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from survivor_agent import (
    BareBonesPickerNet,
    Config as AgentFeatureConfig,
)
from survivor_checkpoints import (
    Genome,
    checkpoint_path_for_generation,
    load_population_for_evolution,
    save_population_checkpoint,
)
from survivor_schedule import (
    FeatureMatchupRow,
    load_schedule_from_csv,
)

DEFAULT_EVOLUTION_WEIGHTS_PATH = "checkpoints/evolution_population.pt"
DEFAULT_SCHEDULE_CSV_PATH = "cleaned_grid2.csv"


def resolve_runtime_device(requested_device: str = "auto") -> torch.device:
    requested = requested_device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda requested, but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_built():
            raise ValueError("--device=mps requested, but this PyTorch build has no MPS support.")
        if not mps_backend.is_available():
            raise ValueError("--device=mps requested, but MPS is not available on this machine.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{requested_device}'. Use auto/cuda/mps/cpu.")


@dataclass
class EvolutionLoopConfig:
    total_agents: int = 200
    agents_per_game: int = 20
    num_teams: int = 32
    max_weeks: int = 18
    num_generations: int = 20
    device: str = "auto"
    load_weights_path: Optional[str] = DEFAULT_EVOLUTION_WEIGHTS_PATH
    save_weights_path: Optional[str] = DEFAULT_EVOLUTION_WEIGHTS_PATH
    checkpoint_every_generation: bool = False
    schedule_csv_path: str = DEFAULT_SCHEDULE_CSV_PATH
    profiler: Optional["ProfilerConfig"] = None


@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_dir: str = "profiles"
    warmup_games: int = 1
    active_games: int = 10
    record_shapes: bool = True
    with_stack: bool = False


@dataclass
class EvolutionProfilerArtifacts:
    output_dir: str
    chrome_trace_path: Optional[str]
    cpu_time_table_path: Optional[str]
    cuda_time_table_path: Optional[str]
    cuda_memory_table_path: Optional[str]
    profiled_steps: int
    peak_cuda_memory_allocated_mb: Optional[float]
    peak_cuda_memory_reserved_mb: Optional[float]


@dataclass
class EvolutionGenerationSummary:
    generation_id: int
    total_generations: int
    avg_winners: float
    avg_week_game_ended: float
    max_winners: int
    new_agents: int
    checkpoint_path: Optional[str]


@dataclass
class EvolutionLoopResult:
    final_save_path: Optional[str]
    runtime_device: str
    profiler_artifacts: Optional[EvolutionProfilerArtifacts] = None


def _feature_input_dim(feature_cfg: AgentFeatureConfig) -> int:
    C = feature_cfg.max_contestants
    T = feature_cfg.max_teams
    W = feature_cfg.max_weeks
    return (C * T) + (W * T) + C + W


def _new_picker_model(
    input_dim: int,
    num_teams: int,
    feature_cfg: AgentFeatureConfig,
    agent_id: int,
    device: torch.device,
) -> BareBonesPickerNet:
    model = BareBonesPickerNet(
        input_dim=input_dim,
        num_teams=num_teams,
        agent_id=agent_id,
        feature_cfg=feature_cfg,
    )
    model.to(device)
    model.eval()
    return model


def _model_num_params(model: BareBonesPickerNet) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _extract_genome(model: BareBonesPickerNet) -> Genome:
    flat_params: List[torch.Tensor] = []
    for param in model.parameters():
        flat_params.append(param.detach().reshape(-1).to(device="cpu", dtype=torch.float32))
    if not flat_params:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(flat_params, dim=0)


def _load_genome_into_model(model: BareBonesPickerNet, genome: Genome) -> None:
    expected_params = _model_num_params(model)
    if genome.numel() != expected_params:
        raise ValueError(
            f"Genome size mismatch: expected {expected_params}, found {int(genome.numel())}"
        )

    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            chunk = genome[offset:offset + numel].to(device=param.device, dtype=param.dtype)
            param.copy_(chunk.view_as(param))
            offset += numel


def _rows_by_week(
    matchup_table: List[FeatureMatchupRow],
    max_weeks: int,
) -> List[List[FeatureMatchupRow]]:
    rows: List[List[FeatureMatchupRow]] = [[] for _ in range(max_weeks)]
    for week_id, team_id, win_prob in matchup_table:
        if 0 <= week_id < max_weeks:
            rows[week_id].append((week_id, team_id, win_prob))
    return rows


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


def _play_single_game_with_models(
    game_models: List[BareBonesPickerNet],
    feature_cfg: AgentFeatureConfig,
    rows_by_week: List[List[FeatureMatchupRow]],
    winner_mask_by_week: torch.Tensor,
) -> tuple[List[int], int]:
    game_size = len(game_models)
    active_slots = list(range(game_size))
    picked_teams_by_slot = [set() for _ in range(game_size)]
    matchup_rows_seen: List[FeatureMatchupRow] = []
    weeks_played = 0

    for week_id in range(feature_cfg.max_weeks):
        if len(active_slots) <= 1:
            break
        round_start_slots = active_slots.copy()

        matchup_rows_seen.extend(rows_by_week[week_id])
        week_winners = set(
            torch.nonzero(winner_mask_by_week[week_id], as_tuple=False).flatten().cpu().tolist()
        )

        next_active_slots: List[int] = []
        contestant_picks: Dict[int, int] = {}
        week_picks_by_slot: Dict[int, int] = {}

        for slot_id in active_slots:
            prior_picks = picked_teams_by_slot[slot_id]
            if len(prior_picks) >= feature_cfg.max_teams:
                continue

            pick_probs = game_models[slot_id](
                contestant_picks=contestant_picks,
                matchup_table=matchup_rows_seen,
                current_week=week_id,
                unavailable_team_ids=list(prior_picks),
            )
            picked_team = int(torch.multinomial(pick_probs, num_samples=1).item())
            week_picks_by_slot[slot_id] = picked_team
            contestant_picks[slot_id] = picked_team
            if picked_team in week_winners:
                next_active_slots.append(slot_id)

        for slot_id, picked_team in week_picks_by_slot.items():
            picked_teams_by_slot[slot_id].add(picked_team)

        weeks_played = week_id + 1
        if len(next_active_slots) == 0 and len(round_start_slots) > 0:
            # If everyone is eliminated in the final played round,
            # treat all round entrants as co-winners.
            active_slots = round_start_slots
            break

        active_slots = next_active_slots

    return active_slots, weeks_played


def _clone_game_winners_to_offspring_models(
    game_models: List[BareBonesPickerNet],
    winner_slots: List[int],
    game_size: int,
) -> List[BareBonesPickerNet]:
    if not winner_slots:
        raise ValueError("Cannot clone from an empty winner set.")

    offspring_models: List[BareBonesPickerNet] = []
    for _ in range(game_size):
        parent_slot = winner_slots[int(torch.randint(0, len(winner_slots), (1,)).item())]
        child = BareBonesPickerNet.mutated_copy(game_models[parent_slot], std=0.05)
        child.to(torch.device("cpu"))
        child.eval()
        offspring_models.append(child)

    return offspring_models


def _write_profiler_table(
    output_dir: Path,
    filename: str,
    table: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(table, encoding="utf-8")
    return str(path)


def _build_profiler(
    profiler_cfg: ProfilerConfig,
    runtime_device: torch.device,
) -> tuple[Optional[torch.profiler.profile], List[torch.profiler.ProfilerActivity], int]:
    if not profiler_cfg.enabled:
        return None, [], 0

    activities: List[torch.profiler.ProfilerActivity] = [torch.profiler.ProfilerActivity.CPU]
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_profile_steps = profiler_cfg.warmup_games + profiler_cfg.active_games
    if total_profile_steps <= 0:
        raise ValueError("--profile-warmup-games + --profile-active-games must be > 0")

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=max(0, profiler_cfg.warmup_games),
            active=max(1, profiler_cfg.active_games),
            repeat=1,
        ),
        record_shapes=profiler_cfg.record_shapes,
        profile_memory=True,
        with_stack=profiler_cfg.with_stack,
    )
    return profiler, activities, total_profile_steps


def _finalize_profiler(
    profiler: torch.profiler.profile,
    profiler_cfg: ProfilerConfig,
    activities: List[torch.profiler.ProfilerActivity],
    profiled_steps: int,
    runtime_device: torch.device,
) -> EvolutionProfilerArtifacts:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(profiler_cfg.output_dir) / f"evolution_profile_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_time_table_path: Optional[str] = None
    cuda_time_table_path: Optional[str] = None
    cuda_memory_table_path: Optional[str] = None
    chrome_trace_path: Optional[str] = None

    cpu_table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=100)
    cpu_time_table_path = _write_profiler_table(output_dir, "cpu_time_top_ops.txt", cpu_table)

    if torch.profiler.ProfilerActivity.CUDA in activities:
        cuda_time_table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)
        cuda_time_table_path = _write_profiler_table(
            output_dir,
            "cuda_time_top_ops.txt",
            cuda_time_table,
        )
        cuda_memory_table = profiler.key_averages().table(
            sort_by="self_cuda_memory_usage",
            row_limit=100,
        )
        cuda_memory_table_path = _write_profiler_table(
            output_dir,
            "cuda_memory_top_ops.txt",
            cuda_memory_table,
        )

    try:
        trace_path = output_dir / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        chrome_trace_path = str(trace_path)
    except RuntimeError:
        # If no active step was captured, export_chrome_trace can fail.
        chrome_trace_path = None

    peak_allocated_mb: Optional[float] = None
    peak_reserved_mb: Optional[float] = None
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        peak_allocated_mb = float(torch.cuda.max_memory_allocated(runtime_device)) / (1024 ** 2)
        peak_reserved_mb = float(torch.cuda.max_memory_reserved(runtime_device)) / (1024 ** 2)

    return EvolutionProfilerArtifacts(
        output_dir=str(output_dir),
        chrome_trace_path=chrome_trace_path,
        cpu_time_table_path=cpu_time_table_path,
        cuda_time_table_path=cuda_time_table_path,
        cuda_memory_table_path=cuda_memory_table_path,
        profiled_steps=profiled_steps,
        peak_cuda_memory_allocated_mb=peak_allocated_mb,
        peak_cuda_memory_reserved_mb=peak_reserved_mb,
    )


def run_evolution_loop(
    cfg: EvolutionLoopConfig,
    on_generation: Optional[Callable[[EvolutionGenerationSummary], None]] = None,
) -> EvolutionLoopResult:
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
    if cfg.profiler is not None:
        if cfg.profiler.warmup_games < 0:
            raise ValueError("--profile-warmup-games must be >= 0")
        if cfg.profiler.active_games <= 0:
            raise ValueError("--profile-active-games must be > 0")

    num_games = cfg.total_agents // cfg.agents_per_game
    device = resolve_runtime_device(cfg.device)

    profiler_cfg = cfg.profiler if cfg.profiler is not None else ProfilerConfig(enabled=False)
    profiler, profiler_activities, total_profile_steps = _build_profiler(
        profiler_cfg=profiler_cfg,
        runtime_device=device,
    )
    profiled_steps = 0

    if profiler is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    feature_cfg = AgentFeatureConfig(
        max_contestants=cfg.agents_per_game,
        max_teams=cfg.num_teams,
        max_weeks=cfg.max_weeks,
    )
    input_dim = _feature_input_dim(feature_cfg)

    schedule = load_schedule_from_csv(
        csv_path=cfg.schedule_csv_path,
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
    )
    rows_by_week = _rows_by_week(
        matchup_table=schedule.feature_rows,
        max_weeks=cfg.max_weeks,
    )
    weekly_odds = _matchup_rows_to_weekly_odds(
        matchup_table=schedule.feature_rows,
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
        device=device,
    )

    template_model = _new_picker_model(
        input_dim=input_dim,
        num_teams=cfg.num_teams,
        feature_cfg=feature_cfg,
        agent_id=0,
        device=torch.device("cpu"),
    )
    num_params = _model_num_params(template_model)

    if cfg.load_weights_path and Path(cfg.load_weights_path).exists():
        population_genomes = load_population_for_evolution(
            path=cfg.load_weights_path,
            expected_total_agents=cfg.total_agents,
            expected_num_params=num_params,
        )
        population_models: List[BareBonesPickerNet] = []
        for genome in population_genomes:
            model = _new_picker_model(
                input_dim=input_dim,
                num_teams=cfg.num_teams,
                feature_cfg=feature_cfg,
                agent_id=0,
                device=torch.device("cpu"),
            )
            _load_genome_into_model(model, genome)
            population_models.append(model)
    else:
        population_models: List[BareBonesPickerNet] = []
        for _ in range(cfg.total_agents):
            model = _new_picker_model(
                input_dim=input_dim,
                num_teams=cfg.num_teams,
                feature_cfg=feature_cfg,
                agent_id=0,
                device=torch.device("cpu"),
            )
            population_models.append(model)

    profiler_context = profiler if profiler is not None else nullcontext()
    with profiler_context as prof:
        for generation_id in range(cfg.num_generations):
            shuffled_indices = torch.randperm(cfg.total_agents).tolist()
            winner_masks = _sample_winner_masks(
                weekly_odds=weekly_odds,
                num_games=num_games,
            )

            next_population_models: List[BareBonesPickerNet] = []
            winners_per_game: List[int] = []
            weeks_ended_by_game: List[int] = []

            with torch.no_grad():
                for game_id in range(num_games):
                    start = game_id * cfg.agents_per_game
                    stop = start + cfg.agents_per_game
                    game_agent_indices = shuffled_indices[start:stop]

                    game_models: List[BareBonesPickerNet] = []
                    for seat_id, population_idx in enumerate(game_agent_indices):
                        model = BareBonesPickerNet.mutated_copy(
                            population_models[population_idx],
                            std=0.0,
                        )
                        model.agent_id = seat_id
                        model.to(device)
                        model.eval()
                        game_models.append(model)

                    winner_slots, weeks_played = _play_single_game_with_models(
                        game_models=game_models,
                        feature_cfg=feature_cfg,
                        rows_by_week=rows_by_week,
                        winner_mask_by_week=winner_masks[game_id],
                    )
                    winners_per_game.append(len(winner_slots))
                    weeks_ended_by_game.append(weeks_played)

                    offspring_models = _clone_game_winners_to_offspring_models(
                        game_models=game_models,
                        winner_slots=winner_slots,
                        game_size=cfg.agents_per_game,
                    )
                    next_population_models.extend(offspring_models)

                    if prof is not None and profiled_steps < total_profile_steps:
                        prof.step()
                        profiled_steps += 1

            population_models = next_population_models

            winners_per_game_tensor = torch.tensor(winners_per_game, dtype=torch.float32)
            weeks_ended_tensor = torch.tensor(weeks_ended_by_game, dtype=torch.float32)
            avg_winners = float(winners_per_game_tensor.mean().item())
            avg_week_game_ended = float(weeks_ended_tensor.mean().item())
            max_winners = int(winners_per_game_tensor.max().item())

            checkpoint_path: Optional[str] = None
            if cfg.checkpoint_every_generation and cfg.save_weights_path:
                path = checkpoint_path_for_generation(
                    cfg.save_weights_path,
                    generation_id=generation_id + 1,
                )
                save_population_checkpoint(
                    path=str(path),
                    population_genomes=[_extract_genome(model) for model in population_models],
                    metadata={
                        "generation_id": generation_id + 1,
                        "total_agents": cfg.total_agents,
                        "agents_per_game": cfg.agents_per_game,
                        "num_teams": cfg.num_teams,
                        "max_weeks": cfg.max_weeks,
                        "schedule_csv_path": cfg.schedule_csv_path,
                        "feature_max_contestants": feature_cfg.max_contestants,
                        "num_params": num_params,
                    },
                )
                checkpoint_path = str(path)

            if on_generation is not None:
                on_generation(
                    EvolutionGenerationSummary(
                        generation_id=generation_id + 1,
                        total_generations=cfg.num_generations,
                        avg_winners=avg_winners,
                        avg_week_game_ended=avg_week_game_ended,
                        max_winners=max_winners,
                        new_agents=len(population_models),
                        checkpoint_path=checkpoint_path,
                    )
                )

        if prof is not None and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    final_save_path: Optional[str] = None
    if cfg.save_weights_path:
        save_population_checkpoint(
            path=cfg.save_weights_path,
            population_genomes=[_extract_genome(model) for model in population_models],
            metadata={
                "generation_id": cfg.num_generations,
                "total_agents": cfg.total_agents,
                "agents_per_game": cfg.agents_per_game,
                "num_teams": cfg.num_teams,
                "max_weeks": cfg.max_weeks,
                "schedule_csv_path": cfg.schedule_csv_path,
                "feature_max_contestants": feature_cfg.max_contestants,
                "num_params": num_params,
            },
        )
        final_save_path = cfg.save_weights_path

    profiler_artifacts: Optional[EvolutionProfilerArtifacts] = None
    if profiler is not None:
        profiler_artifacts = _finalize_profiler(
            profiler=profiler,
            profiler_cfg=profiler_cfg,
            activities=profiler_activities,
            profiled_steps=profiled_steps,
            runtime_device=device,
        )

    return EvolutionLoopResult(
        final_save_path=final_save_path,
        runtime_device=str(device),
        profiler_artifacts=profiler_artifacts,
    )
