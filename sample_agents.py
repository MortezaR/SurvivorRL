from __future__ import annotations

import argparse
import csv
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "survivorrl-matplotlib-cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "survivorrl-xdg-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from survivor_agent import MODEL_DTYPE, build_matchup_odds_tensor, population_policy_forward
from survivor_engine import config, load_population_weights, num_contestants, sample_weekly_winners
from survivor_schedule import load_schedule_from_csv

DEFAULT_WEIGHTS_PATH = "checkpoints/picker_population.pt"
DEFAULT_SCHEDULE_PATH = "cleaned_grid2.csv"
DEFAULT_CONTESTANT_CSV = "sample_contestant_picks.csv"
SELF_ROW_ALIASES = {"self", "__self__", "focal_agent", "agent"}
PAIRWISE_PLOT_LIMIT = 64


@dataclass
class ContestantContext:
    self_pick_ids: List[int]
    other_contestant_ids: List[str]
    other_pick_histories: List[List[int]]


def normalize_team_key(raw_value: str) -> str:
    return " ".join(raw_value.strip().lower().split())


def build_team_lookup(team_names: Sequence[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for team_id, team_name in enumerate(team_names):
        lookup[normalize_team_key(team_name)] = team_id
    return lookup


def parse_team_token(
    raw_value: str,
    team_lookup: Dict[str, int],
    team_names: Sequence[str],
) -> int:
    value = raw_value.strip()
    if not value:
        raise ValueError("Blank team token cannot be parsed.")

    if value.isdigit():
        team_id = int(value)
        if 0 <= team_id < len(team_names):
            return team_id
        raise ValueError(
            f"Team id '{value}' is outside the valid range [0, {len(team_names) - 1}]."
        )

    normalized = normalize_team_key(value)
    if normalized in team_lookup:
        return team_lookup[normalized]

    raise ValueError(
        f"Unknown team value '{raw_value}'. Use a team id or one of the names in {DEFAULT_SCHEDULE_PATH}."
    )


def discover_pick_columns(fieldnames: Sequence[str]) -> List[str]:
    pick_columns = [name for name in fieldnames if name.lower().startswith("pick_")]

    def column_sort_key(column_name: str) -> tuple[int, int | str]:
        suffix = column_name.split("_", maxsplit=1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    return sorted(pick_columns, key=column_sort_key)


def parse_pick_sequence(
    row: Dict[str, str],
    pick_columns: Sequence[str],
    team_lookup: Dict[str, int],
    team_names: Sequence[str],
) -> List[int]:
    values: List[str] = []
    if pick_columns:
        values = [row.get(column_name, "").strip() for column_name in pick_columns]
    else:
        picks_blob = (row.get("picks") or "").strip()
        if picks_blob:
            values = [
                token.strip()
                for token in picks_blob.replace("|", ";").split(";")
            ]

    return [
        parse_team_token(value, team_lookup, team_names)
        for value in values
        if value
    ]


def load_contestant_context(
    csv_path: str | Path,
    team_lookup: Dict[str, int],
    team_names: Sequence[str],
    current_week: int,
) -> ContestantContext:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Contestant CSV has no header: {path}")

        pick_columns = discover_pick_columns(reader.fieldnames)
        self_pick_ids: List[int] = []
        other_contestant_ids: List[str] = []
        other_pick_histories: List[List[int]] = []
        saw_self_row = False

        for row_idx, row in enumerate(reader, start=2):
            contestant_id = (row.get("contestant_id") or "").strip()
            if not contestant_id:
                contestant_id = f"contestant_{row_idx - 1}"

            pick_ids = parse_pick_sequence(row, pick_columns, team_lookup, team_names)
            if len(pick_ids) > current_week:
                raise ValueError(
                    f"Row {row_idx} in {path} contains {len(pick_ids)} prior picks, "
                    f"but week {current_week + 1} only allows at most {current_week}."
                )

            if normalize_team_key(contestant_id) in SELF_ROW_ALIASES:
                if saw_self_row:
                    raise ValueError(f"{path} may only contain one SELF row.")
                self_pick_ids = pick_ids
                saw_self_row = True
                continue

            other_contestant_ids.append(contestant_id)
            other_pick_histories.append(pick_ids)

    return ContestantContext(
        self_pick_ids=self_pick_ids,
        other_contestant_ids=other_contestant_ids,
        other_pick_histories=other_pick_histories,
    )


def one_hot_pick_history(
    pick_ids: Sequence[int],
    num_teams: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    history = torch.zeros((num_teams,), dtype=dtype, device=device)
    for team_id in pick_ids:
        if 0 <= team_id < num_teams:
            history[team_id] = 1.0
    return history


def batch_self_pick_history(
    batch_size: int,
    pick_ids: Sequence[int],
    num_teams: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    self_history = one_hot_pick_history(
        pick_ids,
        num_teams,
        dtype=dtype,
        device=device,
    )
    return self_history.unsqueeze(0).expand(batch_size, -1).clone()


def mean_other_pick_history(
    pick_histories: Sequence[Sequence[int]],
    num_teams: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not pick_histories:
        return torch.zeros((num_teams,), dtype=dtype, device=device)

    stacked = torch.stack(
        [
            one_hot_pick_history(history, num_teams, dtype=dtype, device=device)
            for history in pick_histories
        ],
        dim=0,
    )
    return stacked.mean(dim=0)


def current_week_one_hot(
    num_weeks: int,
    current_week: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    encoded = torch.zeros((num_weeks,), dtype=dtype, device=device)
    encoded[current_week] = 1.0
    return encoded


def stacked_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias


def population_policy_forward_with_context(
    parameter_tensors: Dict[str, torch.Tensor],
    self_pick_history: torch.Tensor,
    other_pick_mean: torch.Tensor,
    matchup_odds: torch.Tensor,
    current_week: int,
    num_players: int,
) -> torch.Tensor:
    batch_size = self_pick_history.shape[0]
    device = self_pick_history.device
    dtype = self_pick_history.dtype

    other_pick_features = other_pick_mean.reshape(1, -1).expand(batch_size, -1)
    matchup_features = matchup_odds.reshape(1, -1).expand(batch_size, -1)
    week_features = current_week_one_hot(
        matchup_odds.shape[0],
        current_week,
        dtype=dtype,
        device=device,
    ).reshape(1, -1).expand(batch_size, -1)
    num_players_feature = torch.full(
        (batch_size, 1),
        float(num_players),
        dtype=dtype,
        device=device,
    )

    x = torch.cat(
        [
            self_pick_history,
            other_pick_features,
            matchup_features,
            week_features,
            num_players_feature,
        ],
        dim=1,
    )

    hidden_1 = torch.relu(
        stacked_linear(
            x,
            parameter_tensors["fc1.weight"],
            parameter_tensors["fc1.bias"],
        )
    )
    hidden_2 = torch.relu(
        stacked_linear(
            hidden_1,
            parameter_tensors["fc2.weight"],
            parameter_tensors["fc2.bias"],
        )
    )
    logits = stacked_linear(
        hidden_2,
        parameter_tensors["fc3.weight"],
        parameter_tensors["fc3.bias"],
    )

    blocked_mask = self_pick_history.to(dtype=torch.bool)
    if blocked_mask.numel() > 0:
        rows_with_available_teams = ~blocked_mask.all(dim=1, keepdim=True)
        effective_blocked_mask = blocked_mask & rows_with_available_teams
        if effective_blocked_mask.any():
            logits = logits.masked_fill(
                effective_blocked_mask,
                torch.finfo(logits.dtype).min,
            )

    return torch.softmax(logits, dim=-1)


def jensen_shannon_distance_to_mean(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[0] == 0:
        return torch.empty(0, dtype=probabilities.dtype, device=probabilities.device)

    eps = torch.finfo(probabilities.dtype).eps
    p = probabilities.clamp_min(eps)
    q = probabilities.mean(dim=0, keepdim=True).expand_as(probabilities).clamp_min(eps)
    m = 0.5 * (p + q)
    divergence = 0.5 * (
        (p * (p.log() - m.log())).sum(dim=1)
        + (q * (q.log() - m.log())).sum(dim=1)
    )
    return torch.sqrt(divergence.clamp_min(0.0))


def jensen_shannon_pairwise(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[0] <= 1:
        return torch.zeros(
            (probabilities.shape[0], probabilities.shape[0]),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )

    eps = torch.finfo(probabilities.dtype).eps
    p = probabilities[:, None, :].clamp_min(eps)
    q = probabilities[None, :, :].clamp_min(eps)
    m = 0.5 * (p + q)
    divergence = 0.5 * (
        (p * (p.log() - m.log())).sum(dim=-1)
        + (q * (q.log() - m.log())).sum(dim=-1)
    )
    return torch.sqrt(divergence.clamp_min(0.0))


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_weekly_probabilities_from_schedule(
    team_names: Sequence[str],
    feature_rows: Sequence[tuple[int, int, float]],
    num_weeks: int,
) -> List[List[tuple[str, float]]]:
    weekly_probabilities: List[List[tuple[str, float]]] = [[] for _ in range(num_weeks)]
    for week_id, team_id, win_prob in feature_rows:
        weekly_probabilities[week_id].append((team_names[team_id], win_prob))
    return weekly_probabilities


def format_distribution_row(
    probabilities: torch.Tensor,
    team_names: Sequence[str],
) -> str:
    order = torch.argsort(probabilities, descending=True)
    return ", ".join(
        f"{team_names[int(team_id)]}={probabilities[int(team_id)].item():.4f}"
        for team_id in order
    )


def save_probability_heatmap(
    probabilities: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    image = ax.imshow(probabilities, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Team")
    ax.set_ylabel("Agent")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    if len(y_labels) <= 50:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
    else:
        ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="Pick probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_distance_bar(
    values: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(np.arange(len(values)), values, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Agent")
    if len(labels) <= 50:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_pairwise_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, label="Jensen-Shannon distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_mode1_probability_csv(
    output_path: Path,
    agent_ids: Sequence[int],
    team_names: Sequence[str],
    probabilities: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent_id", *team_names])
        for agent_id, row in zip(agent_ids, probabilities):
            writer.writerow([agent_id, *[f"{value:.8f}" for value in row]])


def write_distance_csv(
    output_path: Path,
    agent_ids: Sequence[int],
    distances: np.ndarray,
    header_name: str,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent_id", header_name])
        for agent_id, distance in zip(agent_ids, distances):
            writer.writerow([agent_id, f"{distance:.8f}"])


def write_mode2_probability_csv(
    output_path: Path,
    rows: Sequence[Sequence[object]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["week", "agent_id", "team_name", "probability"])
        writer.writerows(rows)


def write_mode2_distance_csv(
    output_path: Path,
    rows: Sequence[Sequence[object]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["week", "agent_id", "js_distance_to_week_mean"])
        writer.writerows(rows)


def resolve_dtype_for_device(device: torch.device) -> torch.dtype:
    return MODEL_DTYPE if device.type == "cuda" else torch.float32


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device_arg)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return resolved


def validate_week(week: int) -> int:
    if week < 1 or week > config.num_weeks:
        raise ValueError(f"--week must be between 1 and {config.num_weeks}.")
    return week - 1


def load_population_parameter_tensors(
    weights_path: str,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    population = load_population_weights(weights_path, config)
    dtype = resolve_dtype_for_device(device)
    parameter_tensors = {
        name: tensor.to(device=device, dtype=dtype)
        for name, tensor in population.parameter_tensors.items()
    }
    return population.agent_ids.clone(), parameter_tensors


def maybe_save_pairwise_subset(
    probabilities: torch.Tensor,
    agent_labels: Sequence[str],
    output_path: Path,
    *,
    seed: int,
) -> None:
    if probabilities.shape[0] <= 1:
        return

    subset_size = min(PAIRWISE_PLOT_LIMIT, probabilities.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if subset_size == probabilities.shape[0]:
        subset_indices = torch.arange(probabilities.shape[0], dtype=torch.long)
    else:
        subset_indices = torch.randperm(
            probabilities.shape[0],
            generator=generator,
        )[:subset_size]

    subset_probs = probabilities.index_select(0, subset_indices.to(probabilities.device))
    subset_labels = [agent_labels[int(index)] for index in subset_indices.tolist()]
    pairwise = jensen_shannon_pairwise(subset_probs).detach().cpu().numpy()
    save_pairwise_heatmap(
        pairwise,
        subset_labels,
        output_path,
        title="Sampled pairwise Jensen-Shannon distance",
    )


def run_mode1(args: argparse.Namespace) -> None:
    current_week = validate_week(args.week)
    output_dir = ensure_output_dir(args.output_dir)
    device = resolve_device(args.device)

    schedule = load_schedule_from_csv(args.schedule_csv, config.num_weeks, config.num_teams)
    team_lookup = build_team_lookup(schedule.team_names)
    contestant_context = load_contestant_context(
        args.contestant_csv,
        team_lookup,
        schedule.team_names,
        current_week,
    )

    if not Path(args.weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights_path}")

    agent_ids, parameter_tensors = load_population_parameter_tensors(args.weights_path, device)
    dtype = next(iter(parameter_tensors.values())).dtype
    matchup_odds = build_matchup_odds_tensor(
        config,
        schedule.feature_rows,
        device=device,
        dtype=dtype,
    )

    self_pick_history = batch_self_pick_history(
        batch_size=int(agent_ids.numel()),
        pick_ids=contestant_context.self_pick_ids,
        num_teams=config.num_teams,
        dtype=dtype,
        device=device,
    )
    other_pick_mean = mean_other_pick_history(
        contestant_context.other_pick_histories,
        config.num_teams,
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        probabilities = population_policy_forward_with_context(
            parameter_tensors=parameter_tensors,
            self_pick_history=self_pick_history,
            other_pick_mean=other_pick_mean,
            matchup_odds=matchup_odds,
            current_week=current_week,
            num_players=max(1, len(contestant_context.other_pick_histories) + 1),
        )
        js_distance = jensen_shannon_distance_to_mean(probabilities)

    agent_id_list = agent_ids.tolist()
    agent_labels = [str(agent_id) for agent_id in agent_id_list]
    probabilities_np = probabilities.detach().cpu().numpy()
    distances_np = js_distance.detach().cpu().numpy()

    probability_csv_path = output_dir / "mode1_agent_probabilities.csv"
    distance_csv_path = output_dir / "mode1_agent_js_distance.csv"
    write_mode1_probability_csv(
        probability_csv_path,
        agent_id_list,
        schedule.team_names,
        probabilities_np,
    )
    write_distance_csv(
        distance_csv_path,
        agent_id_list,
        distances_np,
        "js_distance_to_population_mean",
    )

    save_probability_heatmap(
        probabilities_np,
        schedule.team_names,
        agent_labels,
        output_dir / "mode1_probability_heatmap.png",
        title=f"Mode 1 pick distributions for week {args.week}",
    )
    save_distance_bar(
        distances_np,
        agent_labels,
        output_dir / "mode1_js_distance_bar.png",
        title="Mode 1 Jensen-Shannon distance to population mean",
        y_label="Jensen-Shannon distance",
    )
    maybe_save_pairwise_subset(
        probabilities,
        agent_labels,
        output_dir / "mode1_pairwise_js_subset.png",
        seed=args.seed,
    )

    top_distance_order = np.argsort(-distances_np)[: min(10, len(distances_np))]
    print(f"Mode 1 evaluated {len(agent_id_list)} agents for week {args.week}.")
    print(f"Contestant CSV: {args.contestant_csv}")
    print(f"Optional SELF picks: {contestant_context.self_pick_ids}")
    print(f"Other contestants loaded: {len(contestant_context.other_pick_histories)}")
    print(f"Probability CSV: {probability_csv_path}")
    print(f"Distance CSV: {distance_csv_path}")
    print(f"Heatmap: {output_dir / 'mode1_probability_heatmap.png'}")
    print(f"Distance plot: {output_dir / 'mode1_js_distance_bar.png'}")
    if len(agent_id_list) > 1:
        print(f"Pairwise sample plot: {output_dir / 'mode1_pairwise_js_subset.png'}")
    print("Top agents by Jensen-Shannon distance to the population mean:")
    for rank, idx in enumerate(top_distance_order, start=1):
        print(
            f"  {rank}. agent {agent_id_list[int(idx)]}: "
            f"distance={distances_np[int(idx)]:.6f}, "
            f"distribution={format_distribution_row(probabilities[int(idx)], schedule.team_names)}"
        )


def pick_output_rows(
    week_idx: int,
    active_agent_ids: Sequence[int],
    team_names: Sequence[str],
    probabilities: torch.Tensor,
) -> List[List[object]]:
    rows: List[List[object]] = []
    probabilities_cpu = probabilities.detach().cpu()
    for agent_offset, agent_id in enumerate(active_agent_ids):
        for team_id, team_name in enumerate(team_names):
            rows.append(
                [
                    week_idx + 1,
                    agent_id,
                    team_name,
                    f"{probabilities_cpu[agent_offset, team_id].item():.8f}",
                ]
            )
    return rows


def distance_output_rows(
    week_idx: int,
    active_agent_ids: Sequence[int],
    distances: torch.Tensor,
) -> List[List[object]]:
    distances_cpu = distances.detach().cpu()
    return [
        [week_idx + 1, agent_id, f"{distances_cpu[offset].item():.8f}"]
        for offset, agent_id in enumerate(active_agent_ids)
    ]


def sample_agent_subset(
    all_agent_ids: torch.Tensor,
    parameter_tensors: Dict[str, torch.Tensor],
    sample_size: int,
    *,
    seed: int,
) -> tuple[List[int], torch.Tensor, Dict[str, torch.Tensor]]:
    if sample_size < 1:
        raise ValueError("--sim-group-size must be at least 1.")
    if sample_size > int(all_agent_ids.numel()):
        raise ValueError(
            f"--sim-group-size {sample_size} exceeds checkpoint population {int(all_agent_ids.numel())}."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sample_indices = torch.randperm(int(all_agent_ids.numel()), generator=generator)[:sample_size]
    sampled_agent_ids = all_agent_ids.index_select(0, sample_indices).tolist()
    sampled_parameter_tensors = {
        name: tensor.index_select(0, sample_indices.to(tensor.device))
        for name, tensor in parameter_tensors.items()
    }
    return sampled_agent_ids, sample_indices, sampled_parameter_tensors


def run_mode2(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    device = resolve_device(args.device)

    schedule = load_schedule_from_csv(args.schedule_csv, config.num_weeks, config.num_teams)
    if not Path(args.weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights_path}")

    all_agent_ids, parameter_tensors = load_population_parameter_tensors(args.weights_path, device)
    dtype = next(iter(parameter_tensors.values())).dtype
    matchup_odds = build_matchup_odds_tensor(
        config,
        schedule.feature_rows,
        device=device,
        dtype=dtype,
    )

    sampled_agent_ids, _, sampled_parameter_tensors = sample_agent_subset(
        all_agent_ids,
        parameter_tensors,
        args.sim_group_size,
        seed=args.seed,
    )
    sampled_winning_teams = sample_weekly_winners(
        build_weekly_probabilities_from_schedule(
            schedule.team_names,
            schedule.feature_rows,
            config.num_weeks,
        )
    )

    active_rows = torch.arange(args.sim_group_size, device=device, dtype=torch.long)
    pick_history = torch.zeros(
        (args.sim_group_size, config.num_teams),
        device=device,
        dtype=torch.bool,
    )

    probability_rows: List[List[object]] = []
    distance_rows: List[List[object]] = []

    print(
        f"Mode 2 sampled {args.sim_group_size} agents from the checkpoint and will simulate up to {config.num_weeks} weeks."
    )

    for week_idx in range(config.num_weeks):
        if active_rows.numel() == 0:
            print(f"Week {week_idx + 1}: no active agents remain, stopping simulation.")
            break

        with torch.inference_mode():
            pick_dist = population_policy_forward(
                cfg=config,
                parameter_tensors=sampled_parameter_tensors,
                active_indices=active_rows,
                contestant_pick_history=pick_history,
                matchup_odds=matchup_odds,
                current_week=week_idx,
                num_players=int(active_rows.numel()),
                unavailable_team_mask=pick_history,
            )
            js_distance = jensen_shannon_distance_to_mean(pick_dist)

        active_agent_ids = [sampled_agent_ids[int(index)] for index in active_rows.tolist()]
        probability_rows.extend(
            pick_output_rows(
                week_idx,
                active_agent_ids,
                schedule.team_names,
                pick_dist,
            )
        )
        distance_rows.extend(distance_output_rows(week_idx, active_agent_ids, js_distance))

        weekly_output_dir = ensure_output_dir(output_dir / f"week_{week_idx + 1:02d}")
        probabilities_np = pick_dist.detach().cpu().numpy()
        distances_np = js_distance.detach().cpu().numpy()
        agent_labels = [str(agent_id) for agent_id in active_agent_ids]

        save_probability_heatmap(
            probabilities_np,
            schedule.team_names,
            agent_labels,
            weekly_output_dir / "probability_heatmap.png",
            title=f"Mode 2 week {week_idx + 1} pick distributions",
        )
        save_distance_bar(
            distances_np,
            agent_labels,
            weekly_output_dir / "js_distance_bar.png",
            title=f"Mode 2 week {week_idx + 1} Jensen-Shannon distance",
            y_label="Jensen-Shannon distance",
        )
        maybe_save_pairwise_subset(
            pick_dist,
            agent_labels,
            weekly_output_dir / "pairwise_js_subset.png",
            seed=args.seed + week_idx,
        )

        winning_team_names = [
            schedule.team_names[team_id]
            for team_id in sampled_winning_teams[week_idx][1]
        ]
        print()
        print(f"Week {week_idx + 1}")
        print(f"  Winning teams sampled: {winning_team_names}")
        print(f"  Active agents: {active_agent_ids}")
        print("  Jensen-Shannon distance to the weekly mean distribution:")
        for agent_id, distance in zip(active_agent_ids, distances_np):
            print(f"    agent {agent_id}: {distance:.6f}")
        print("  Full pick distributions:")
        for row_idx, agent_id in enumerate(active_agent_ids):
            print(
                f"    agent {agent_id}: "
                f"{format_distribution_row(pick_dist[row_idx], schedule.team_names)}"
            )

        picked_team_ids = torch.multinomial(pick_dist, num_samples=1).squeeze(1)
        winning_team_mask = torch.zeros(
            (config.num_teams,),
            device=device,
            dtype=torch.bool,
        )
        if sampled_winning_teams[week_idx][1]:
            winning_team_mask[sampled_winning_teams[week_idx][1]] = True
        survivor_mask = winning_team_mask[picked_team_ids]

        print("  Sampled picks and outcomes:")
        for row_idx, agent_id in enumerate(active_agent_ids):
            picked_team_id = int(picked_team_ids[row_idx].item())
            survived = bool(survivor_mask[row_idx].item())
            print(
                f"    agent {agent_id}: picked {schedule.team_names[picked_team_id]} "
                f"({'survived' if survived else 'eliminated'})"
            )

        next_active_rows = active_rows[survivor_mask]
        next_pick_history = pick_history[survivor_mask].clone()
        if next_pick_history.shape[0] > 0:
            next_pick_history[
                torch.arange(next_pick_history.shape[0], device=device),
                picked_team_ids[survivor_mask],
            ] = True

        active_rows = next_active_rows
        pick_history = next_pick_history

    probability_csv_path = output_dir / "mode2_weekly_probabilities.csv"
    distance_csv_path = output_dir / "mode2_weekly_js_distance.csv"
    write_mode2_probability_csv(probability_csv_path, probability_rows)
    write_mode2_distance_csv(distance_csv_path, distance_rows)

    print()
    print(f"Mode 2 probability CSV: {probability_csv_path}")
    print(f"Mode 2 distance CSV: {distance_csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a checkpointed SurvivorRL population in two analysis modes.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    mode1_parser = subparsers.add_parser(
        "mode1",
        help="Evaluate every checkpointed agent against a provided contestant state for a given week.",
    )
    mode1_parser.add_argument("--week", type=int, required=True, help="1-based NFL week to evaluate.")
    mode1_parser.add_argument(
        "--contestant-csv",
        type=str,
        default=DEFAULT_CONTESTANT_CSV,
        help="CSV with other contestants and optional SELF row.",
    )
    mode1_parser.add_argument(
        "--weights-path",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help="Checkpoint file containing the population to evaluate.",
    )
    mode1_parser.add_argument(
        "--schedule-csv",
        type=str,
        default=DEFAULT_SCHEDULE_PATH,
        help="Schedule CSV with team names and per-week probabilities.",
    )
    mode1_parser.add_argument(
        "--output-dir",
        type=str,
        default="agent_samples/mode1",
        help="Directory for plots and CSV exports.",
    )
    mode1_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:N.",
    )
    mode1_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for any sampled diagnostic plots.",
    )
    mode1_parser.set_defaults(func=run_mode1)

    mode2_parser = subparsers.add_parser(
        "mode2",
        help="Simulate a multi-week game with a sampled subset of checkpointed agents.",
    )
    mode2_parser.add_argument(
        "--sim-group-size",
        type=int,
        default=num_contestants,
        help="How many agents to sample from the checkpoint for the simulation.",
    )
    mode2_parser.add_argument(
        "--weights-path",
        type=str,
        default=DEFAULT_WEIGHTS_PATH,
        help="Checkpoint file containing the population to sample from.",
    )
    mode2_parser.add_argument(
        "--schedule-csv",
        type=str,
        default=DEFAULT_SCHEDULE_PATH,
        help="Schedule CSV with team names and per-week probabilities.",
    )
    mode2_parser.add_argument(
        "--output-dir",
        type=str,
        default="agent_samples/mode2",
        help="Directory for plots and CSV exports.",
    )
    mode2_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or cuda:N.",
    )
    mode2_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for agent sampling and winner simulation.",
    )
    mode2_parser.set_defaults(func=run_mode2)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.func(args)


if __name__ == "__main__":
    main()
