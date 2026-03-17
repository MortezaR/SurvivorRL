import argparse
from pathlib import Path
import tempfile

import torch

from survivor_engine import (
    config,
    evo_loop,
    get_population_store_size,
    initialize_random_population_store,
    load_population_checkpoint_into_store,
    num_agents,
    num_contestants,
    save_population_store,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SurvivorRL evolution loop.")
    parser.add_argument("--num-loops", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument(
        "--load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load a saved population checkpoint before running.",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the evolved population checkpoint after running.",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="checkpoints/picker_population.pt",
        help="Path to a legacy .pt checkpoint file or a disk-backed population directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Run model inference/evolution on this device.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available. Use --device cpu.")
    runtime_device = torch.device(args.device)

    with tempfile.TemporaryDirectory(prefix="survivor_evo_", dir=str(Path.cwd())) as temp_dir:
        temp_root = Path(temp_dir)
        initial_population_dir = temp_root / "generation_0000"
        weights_path = Path(args.weights_path)

        if args.load and weights_path.exists():
            load_population_checkpoint_into_store(str(weights_path), initial_population_dir)
            print(f"Loaded population from: {weights_path}")
        else:
            if args.load:
                print(f"Load requested but no checkpoint found at: {weights_path}")
            print("Starting from initialized population.")
            initialize_random_population_store(
                population_dir=initial_population_dir,
                cfg=config,
                population_size=num_agents,
            )

        print(f"Using device: {runtime_device}")

        evolved_population_dir = evo_loop(
            population_dir=initial_population_dir,
            cfg=config,
            num_loops=args.num_loops,
            num_contestants=num_contestants,
            device=runtime_device,
            noise_std=args.noise_std,
            work_dir=temp_root,
        )
        print(f"Finished {args.num_loops} evo loop(s).")
        print(f"Population size: {get_population_store_size(evolved_population_dir)}")

        if args.save:
            save_population_store(evolved_population_dir, args.weights_path)
            print(f"Saved population to: {args.weights_path}")
