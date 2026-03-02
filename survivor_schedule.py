from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

FeatureMatchupRow = Tuple[int, int, float]  # (week_id, team_id, win_prob)


@dataclass
class CsvSchedule:
    feature_rows: List[FeatureMatchupRow]
    team_names: List[str]
    num_weeks: int
    num_teams: int


def _normalize_probability(raw_value: str) -> float:
    """
    CSV values are percentages (e.g. 64.3) and are clamped to [0, 1].
    Already-normalized probabilities in [0, 1] are also accepted.
    """
    value = float(raw_value) if raw_value else 0.0
    if value < 0.0 or value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def load_schedule_from_csv(
    csv_path: str | Path,
    num_weeks: int,
    num_teams: int,
) -> CsvSchedule:
    """
    Load a team-week schedule from CSV into feature rows for featurization.
    """
    if num_weeks <= 0:
        raise ValueError("num_weeks must be > 0")
    if num_teams <= 0:
        raise ValueError("num_teams must be > 0")

    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if "Team" not in reader.fieldnames:
            raise ValueError(f"CSV must include a 'Team' column: {path}")

        week_columns = [str(week_id + 1) for week_id in range(num_weeks)]
        missing_columns = [col for col in week_columns if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"CSV is missing required week columns {missing_columns} in {path}"
            )

        feature_rows: List[FeatureMatchupRow] = []
        team_names: List[str] = []

        for team_id, row in enumerate(reader):
            if team_id >= num_teams:
                break
            team_name = (row.get("Team") or "").strip() or f"Team{team_id}"
            team_names.append(team_name)
            for week_id, column_name in enumerate(week_columns):
                win_prob = _normalize_probability((row.get(column_name) or "").strip())
                feature_rows.append((week_id, team_id, win_prob))

    if len(team_names) < num_teams:
        raise ValueError(
            f"CSV only had {len(team_names)} teams, but num_teams={num_teams} was requested."
        )

    return CsvSchedule(
        feature_rows=feature_rows,
        team_names=team_names,
        num_weeks=num_weeks,
        num_teams=num_teams,
    )


def sample_weekly_winners(
    matchup_table: List[FeatureMatchupRow],
    num_weeks: int,
    num_teams: int,
) -> List[List[int]]:
    """
    Sample winners from per-team weekly probabilities.
    Each team's weekly outcome is sampled independently.
    """
    weekly_probs: List[List[float]] = [[0.0 for _ in range(num_teams)] for _ in range(num_weeks)]
    for week_id, team_id, win_prob in matchup_table:
        if 0 <= week_id < num_weeks and 0 <= team_id < num_teams:
            weekly_probs[week_id][team_id] = max(0.0, min(1.0, float(win_prob)))

    rng = random.Random()
    winners_by_week: List[List[int]] = []
    for week_id in range(num_weeks):
        week_winners: List[int] = []
        for team_id, win_prob in enumerate(weekly_probs[week_id]):
            if win_prob > 0.0 and rng.random() < win_prob:
                week_winners.append(team_id)
        winners_by_week.append(week_winners)

    return winners_by_week
