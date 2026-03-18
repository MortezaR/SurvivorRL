import argparse
from pathlib import Path

import torch

from survivor_engine import (
    all_agents,
    config,
    evo_loop,
    load_population_weights,
    move_population_to_device,
    num_contestants,
    save_population_weights,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SurvivorRL evolution loop.")
    parser.add_argument("--num-loops", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load population weights before running.",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save population weights after running.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="checkpoints/picker_population.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Run model inference/evolution on this device.",
    )
    parser.add_argument(
        "--game-workers",
        type=int,
        default=None,
        help="Number of concurrent game workers. Defaults to 2 on CUDA and 1 on CPU.",
    )
    args = parser.parse_args()

    if args.game_workers is not None and args.game_workers < 1:
        raise ValueError("--game-workers must be >= 1 when provided.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available. Use --device cpu.")
    runtime_device = torch.device(args.device)

    population = list(all_agents)
    if args.load:
        weights_path = Path(args.weights_path)
        if weights_path.exists():
            population = load_population_weights(str(weights_path), config)
            print(f"Loaded population from: {weights_path}")
        else:
            print(f"Load requested but no checkpoint found at: {weights_path}")
            print("Starting from initialized population.")

    population = move_population_to_device(population, runtime_device)
    print(f"Using device: {runtime_device}")

    evolved_population = evo_loop(
        all_agents=population,
        num_loops=args.num_loops,
        num_contestants=num_contestants,
        noise_std=args.noise_std,
        game_workers=args.game_workers,
    )
    print(f"Finished {args.num_loops} evo loop(s).")
    print(f"Population size: {len(evolved_population)}")

    if args.save:
        save_population_weights(evolved_population, args.weights_path)
        print(f"Saved population to: {args.weights_path}")
