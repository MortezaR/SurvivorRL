from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch

Genome = torch.Tensor


def checkpoint_path_for_generation(base_path: str, generation_id: int) -> Path:
    path = Path(base_path)
    suffix = path.suffix if path.suffix else ".pt"
    return path.with_name(f"{path.stem}_gen{generation_id}{suffix}")


def genome_to_cpu_float32(genome: Genome) -> Genome:
    return genome.detach().to(device="cpu", dtype=torch.float32).clone()


def validate_population_genomes(
    population_genomes: List[Genome],
    expected_total_agents: int,
    expected_num_params: int,
    source_path: str,
) -> None:
    if len(population_genomes) != expected_total_agents:
        raise ValueError(
            f"Checkpoint shape mismatch for {source_path}: expected {expected_total_agents} "
            f"agents, found {len(population_genomes)}"
        )

    for idx, genome in enumerate(population_genomes):
        if not isinstance(genome, torch.Tensor):
            raise ValueError(
                f"Checkpoint {source_path} has non-tensor genome at agent {idx}."
            )
        if genome.ndim != 1:
            raise ValueError(
                f"Checkpoint {source_path} genome mismatch at agent {idx}: expected rank-1 "
                f"tensor of length {expected_num_params}, found shape {tuple(genome.shape)}"
            )
        if genome.numel() != expected_num_params:
            raise ValueError(
                f"Checkpoint {source_path} genome mismatch at agent {idx}: expected "
                f"{expected_num_params} params, found {int(genome.numel())}"
            )


def save_population_checkpoint(
    path: str,
    population_genomes: List[Genome],
    metadata: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "population_genomes": [
            genome_to_cpu_float32(genome) for genome in population_genomes
        ],
    }
    payload.update(metadata)
    torch.save(payload, checkpoint_path)


def load_population_checkpoint(path: str) -> tuple[List[Genome], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint {path} is invalid for genome-based evolution. Expected a dict payload."
        )

    if "population_genomes" not in payload:
        if "population" in payload:
            raise ValueError(
                f"Checkpoint {path} uses legacy compact population tensors. "
                "This engine now expects 'population_genomes'."
            )
        raise ValueError(
            f"Checkpoint {path} did not contain 'population_genomes'."
        )

    raw_population = payload["population_genomes"]
    if not isinstance(raw_population, list):
        raise ValueError(
            f"Checkpoint {path} has invalid 'population_genomes'; expected list."
        )

    population_genomes: List[Genome] = []
    for idx, raw_genome in enumerate(raw_population):
        if not isinstance(raw_genome, torch.Tensor):
            raise ValueError(
                f"Checkpoint {path} has invalid genome at index {idx}; expected tensor."
            )
        population_genomes.append(genome_to_cpu_float32(raw_genome).flatten())

    metadata = {k: v for k, v in payload.items() if k != "population_genomes"}
    return population_genomes, metadata


def load_population_for_evolution(
    path: str,
    expected_total_agents: int,
    expected_num_params: int,
) -> List[Genome]:
    population_genomes, _ = load_population_checkpoint(path)
    validate_population_genomes(
        population_genomes=population_genomes,
        expected_total_agents=expected_total_agents,
        expected_num_params=expected_num_params,
        source_path=path,
    )
    return population_genomes
