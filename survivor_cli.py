from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from tqdm.auto import tqdm

from survivor_engine import (
    DEFAULT_EVOLUTION_WEIGHTS_PATH,
    DEFAULT_SCHEDULE_CSV_PATH,
    EvolutionGenerationSummary,
    EvolutionLoopConfig,
    ProfilerConfig,
    resolve_runtime_device,
    run_evolution_loop,
)
from survivor_sampling import sample_agent_distribution


def _format_probs(
    probabilities: List[float],
    team_names: List[str],
    team_ids: Iterable[int],
) -> str:
    return ", ".join(
        f"{team_names[team_id]}={probabilities[team_id]:.4f}" for team_id in team_ids
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutionary survivor simulations.")
    parser.add_argument(
        "--mode",
        choices=["evolution-loop", "sample-agent"],
        default="evolution-loop",
    )

    # Evolution loop mode args.
    parser.add_argument("--total-agents", type=int, default=1000000)
    parser.add_argument("--agents-per-game", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument("--num-teams", type=int, default=32)
    parser.add_argument("--max-weeks", type=int, default=18)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--load-weights-path", type=str, default=DEFAULT_EVOLUTION_WEIGHTS_PATH)
    parser.add_argument("--save-weights-path", type=str, default=DEFAULT_EVOLUTION_WEIGHTS_PATH)
    parser.add_argument("--checkpoint-every-generation", action="store_true")
    parser.add_argument("--schedule-csv-path", type=str, default=DEFAULT_SCHEDULE_CSV_PATH)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output-dir", type=str, default="profiles")
    parser.add_argument("--profile-warmup-games", type=int, default=1)
    parser.add_argument("--profile-active-games", type=int, default=10)
    parser.add_argument("--profile-with-stack", action="store_true")

    # Sample-agent mode args.
    parser.add_argument("--sample-agent-id", type=int, default=None)
    parser.add_argument("--sample-top-k", type=int, default=5)
    parser.add_argument("--sample-max-weeks", type=int, default=18)
    parser.add_argument("--print-full-distribution", action="store_true")
    return parser


def _print_generation_summary(summary: EvolutionGenerationSummary) -> str:
    return (
        f"Generation {summary.generation_id}/{summary.total_generations}: "
        f"avg winners/game={summary.avg_winners:.2f}, "
        f"avg week game ended={summary.avg_week_game_ended:.2f}, "
        f"max={summary.max_winners}, "
        f"new_agents={summary.new_agents}"
    )


def _run_evolution_mode(args: argparse.Namespace) -> None:
    loop_cfg = EvolutionLoopConfig(
        total_agents=args.total_agents,
        agents_per_game=args.agents_per_game,
        num_teams=args.num_teams,
        max_weeks=args.max_weeks,
        num_generations=args.num_generations,
        device=args.device,
        load_weights_path=args.load_weights_path,
        save_weights_path=args.save_weights_path,
        checkpoint_every_generation=args.checkpoint_every_generation,
        schedule_csv_path=args.schedule_csv_path,
        profiler=ProfilerConfig(
            enabled=args.profile,
            output_dir=args.profile_output_dir,
            warmup_games=args.profile_warmup_games,
            active_games=args.profile_active_games,
            with_stack=args.profile_with_stack,
        ),
    )

    if loop_cfg.load_weights_path and Path(loop_cfg.load_weights_path).exists():
        print(f"Loaded evolution weights from: {loop_cfg.load_weights_path}")
    elif loop_cfg.load_weights_path:
        print(
            f"No saved weights found at {loop_cfg.load_weights_path}. "
            "Starting from random initialization."
        )

    # Validate and normalize the device choice before creating the progress bar.
    loop_cfg.device = str(resolve_runtime_device(loop_cfg.device))

    num_games = loop_cfg.total_agents // loop_cfg.agents_per_game
    progress = tqdm(total=loop_cfg.num_generations, desc="Generations", unit="gen")

    def on_generation(summary: EvolutionGenerationSummary) -> None:
        tqdm.write(_print_generation_summary(summary))
        progress.update(1)
        progress.set_postfix(
            avg_winners=f"{summary.avg_winners:.2f}",
            avg_week_ended=f"{summary.avg_week_game_ended:.2f}",
            max_winners=summary.max_winners,
        )
        if summary.checkpoint_path:
            tqdm.write(f"Saved generation checkpoint: {summary.checkpoint_path}")

    print(
        "Starting loop: "
        f"{loop_cfg.total_agents} agents -> {num_games} games x {loop_cfg.agents_per_game} agents/game"
    )
    try:
        result = run_evolution_loop(loop_cfg, on_generation=on_generation)
    finally:
        progress.close()

    print(f"Runtime device: {result.runtime_device}")
    if result.final_save_path:
        print(f"Saved final evolution weights to: {result.final_save_path}")
    if result.profiler_artifacts is not None:
        print(
            "Profiler output: "
            f"{result.profiler_artifacts.output_dir} "
            f"(steps={result.profiler_artifacts.profiled_steps})"
        )
        if result.profiler_artifacts.chrome_trace_path:
            print(f"  Chrome trace: {result.profiler_artifacts.chrome_trace_path}")
        if result.profiler_artifacts.cpu_time_table_path:
            print(f"  CPU op table: {result.profiler_artifacts.cpu_time_table_path}")
        if result.profiler_artifacts.cuda_time_table_path:
            print(f"  CUDA op table: {result.profiler_artifacts.cuda_time_table_path}")
        if result.profiler_artifacts.cuda_memory_table_path:
            print(f"  CUDA memory table: {result.profiler_artifacts.cuda_memory_table_path}")
        if result.profiler_artifacts.peak_cuda_memory_allocated_mb is not None:
            print(
                "  Peak CUDA memory: "
                f"allocated={result.profiler_artifacts.peak_cuda_memory_allocated_mb:.2f} MB, "
                f"reserved={result.profiler_artifacts.peak_cuda_memory_reserved_mb:.2f} MB"
            )


def _run_sample_agent_mode(args: argparse.Namespace) -> None:
    if args.sample_top_k <= 0:
        raise ValueError("--sample-top-k must be > 0")

    result = sample_agent_distribution(
        weights_path=args.load_weights_path,
        schedule_csv_path=args.schedule_csv_path,
        max_weeks=args.sample_max_weeks,
        sample_agent_id=args.sample_agent_id,
    )

    if result.checkpoint_generation is not None:
        print(f"Checkpoint generation: {result.checkpoint_generation}")
    print(f"Loaded weights from: {result.weights_path}")
    print(f"Population shape: agents={result.total_agents}, teams={result.num_teams}")
    print(f"Sampled agent id: {result.selected_agent_id}")
    print(f"Schedule: {result.schedule_csv_path} | weeks={result.max_weeks}")

    for week in result.weeks:
        print(
            f"Week {week.week_id:02d}: sampled_pick={week.sampled_team_name} "
            f"(pick_prob={week.sampled_pick_probability:.4f}, "
            f"win_odds={week.sampled_team_win_odds:.4f})"
        )
        if args.print_full_distribution:
            print(
                "  distribution: "
                f"{_format_probs(week.probabilities, result.team_names, range(result.num_teams))}"
            )
        else:
            top_k = min(args.sample_top_k, result.num_teams)
            top_team_ids = sorted(
                range(result.num_teams),
                key=lambda team_id: week.probabilities[team_id],
                reverse=True,
            )[:top_k]
            print(
                f"  top_{top_k}: "
                f"{_format_probs(week.probabilities, result.team_names, top_team_ids)}"
            )


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "sample-agent":
        _run_sample_agent_mode(args)
        return

    _run_evolution_mode(args)


if __name__ == "__main__":
    main()
