from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

FeatureMatchupRow = Tuple[int, int, float]  # (week_id, team_id, win_prob)
OpponentMatchupRow = Tuple[int, int, int, float]  # (week_id, team_id, opponent_id, win_prob)
GameRow = Tuple[int, int, int, float, float]  # (week_id, team_a, team_b, p_a, p_b)
GameOutcomeRow = Tuple[int, int, int, int]  # (week_id, team_a, team_b, winner_id)


@dataclass
class GeneratedSchedule:
    feature_rows: List[FeatureMatchupRow]
    opponent_rows: List[OpponentMatchupRow]
    games_by_week: List[List[GameRow]]


def build_round_robin_schedule(
    num_weeks: int,
    num_teams: int,
) -> GeneratedSchedule:
    """
    Build weekly team-vs-team matchups with per-game complementary odds.
    """
    rng = random.Random()
    teams = list(range(num_teams))

    feature_rows: List[FeatureMatchupRow] = []
    opponent_rows: List[OpponentMatchupRow] = []
    games_by_week: List[List[GameRow]] = []

    for week_id in range(num_weeks):
        week_games: List[GameRow] = []
        for i in range(num_teams // 2):
            team_a = teams[i]
            team_b = teams[-(i + 1)]

            p_a = rng.random()
            p_b = 1.0 - p_a

            week_games.append((week_id, team_a, team_b, p_a, p_b))

            feature_rows.append((week_id, team_a, p_a))
            feature_rows.append((week_id, team_b, p_b))

            opponent_rows.append((week_id, team_a, team_b, p_a))
            opponent_rows.append((week_id, team_b, team_a, p_b))

        games_by_week.append(week_games)
        teams = [teams[0], teams[-1], *teams[1:-1]]

    return GeneratedSchedule(
        feature_rows=feature_rows,
        opponent_rows=opponent_rows,
        games_by_week=games_by_week,
    )


def sample_game_outcomes(
    games_by_week: List[List[GameRow]],
) -> List[List[GameOutcomeRow]]:
    """
    Pick one winner per game, outside the environment.
    """
    rng = random.Random()
    outcomes_by_week: List[List[GameOutcomeRow]] = []

    for week_games in games_by_week:
        week_outcomes: List[GameOutcomeRow] = []
        for week_id, team_a, team_b, p_a, p_b in week_games:
            winner_id = rng.choices([team_a, team_b], weights=[p_a, p_b], k=1)[0]
            week_outcomes.append((week_id, team_a, team_b, winner_id))
        outcomes_by_week.append(week_outcomes)

    return outcomes_by_week


def extract_weekly_winners(
    outcomes_by_week: List[List[GameOutcomeRow]],
) -> List[List[int]]:
    winners_by_week: List[List[int]] = []
    for week_outcomes in outcomes_by_week:
        winners_by_week.append([winner_id for _, _, _, winner_id in week_outcomes])
    return winners_by_week
