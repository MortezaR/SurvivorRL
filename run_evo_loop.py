import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch

from survivor_engine import (
    config,
    create_initial_population,
    evo_loop,
    load_population_weights,
    num_agents,
    num_contestants,
    save_population_weights,
)


def resolve_dispatch_devices(device_arg: str) -> list[torch.device]:
    if device_arg == "cpu":
        return [torch.device("cpu")]

    try:
        requested_device = torch.device(device_arg)
    except RuntimeError as exc:
        raise ValueError(
            f"Unsupported device '{device_arg}'. Use 'cpu', 'cuda', or 'cuda:N'."
        ) from exc

    if requested_device.type != "cuda":
        raise ValueError(
            f"Unsupported device '{device_arg}'. Use 'cpu', 'cuda', or 'cuda:N'."
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available. Use --device cpu.")

    device_count = torch.cuda.device_count()
    if requested_device.index is None:
        return [torch.device(f"cuda:{idx}") for idx in range(device_count)]
    if requested_device.index < 0 or requested_device.index >= device_count:
        raise ValueError(
            f"CUDA device index {requested_device.index} is out of range for {device_count} visible GPU(s)."
        )
    return [requested_device]


if __name__ == "__main__":
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

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
        default="cuda",
        help="Run on 'cpu', one GPU via 'cuda:N', or all visible GPUs via 'cuda'.",
    )
    parser.add_argument(
        "--game-workers",
        type=int,
        default=None,
        help="Number of active device workers. Defaults to one per device; values above device count are capped.",
    )
    args = parser.parse_args()

    if args.game_workers is not None and args.game_workers < 1:
        raise ValueError("--game-workers must be >= 1 when provided.")
    dispatch_devices = resolve_dispatch_devices(args.device)

    if args.load:
        weights_path = Path(args.weights_path)
        if weights_path.exists():
            population = load_population_weights(str(weights_path), config)
            print(f"Loaded population from: {weights_path}")
        else:
            print(f"Load requested but no checkpoint found at: {weights_path}")
            print("Starting from initialized population.")
            population = create_initial_population(num_agents=num_agents, cfg=config)
    else:
        population = create_initial_population(num_agents=num_agents, cfg=config)

    if len(dispatch_devices) == 1:
        print(f"Dispatching games on device: {dispatch_devices[0]}")
    else:
        print("Dispatching games on devices: " + ", ".join(str(device) for device in dispatch_devices))

    evolved_population = evo_loop(
        population=population,
        num_loops=args.num_loops,
        num_contestants=num_contestants,
        noise_std=args.noise_std,
        game_workers=args.game_workers,
        dispatch_devices=dispatch_devices,
    )
    print(f"Finished {args.num_loops} evo loop(s).")
    print(f"Population size: {len(evolved_population)}")

    if args.save:
        save_population_weights(evolved_population, args.weights_path)
        print(f"Saved population to: {args.weights_path}")
