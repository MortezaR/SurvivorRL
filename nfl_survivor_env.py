from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - clearer import error at runtime
    raise ImportError(
        "gymnasium is required to use NFLSurvivorEnv. Install with `pip install gymnasium`."
    ) from exc


Team = str
Contestant = str
Game = Tuple[Team, Team]
ActionValue = Union[int, Team]


@dataclass(frozen=True)
class WeekData:
    games: List[Game]
    probabilities: Dict[Game, float]


class NFLSurvivorEnv(gym.Env):
    """
    Multi-contestant Gymnasium environment for NFL survivor pools.

    Expected weekly input:
      * games: list of (home_team, away_team)
      * probabilities: dict[(team_a, team_b)] = P(team_a wins)

    Each step receives an action for each alive contestant. Contestants may not
    pick a previously selected team. Winners continue; losers are eliminated.

    Episode ends when:
      * one contestant remains alive, OR
      * the schedule is exhausted.

    Rewards at terminal step:
      * total pool reward is (num_contestants - 1)
      * reward is split evenly across all surviving contestants.
      * eliminated contestants get 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        teams: Sequence[Team],
        weekly_schedule: Sequence[Sequence[Game]],
        probability_chart: Sequence[Mapping[Game, float]],
        num_contestants: int,
        contestant_ids: Optional[Sequence[Contestant]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if num_contestants < 2:
            raise ValueError("num_contestants must be >= 2")
        if len(weekly_schedule) == 0:
            raise ValueError("weekly_schedule must contain at least one week")
        if len(weekly_schedule) != len(probability_chart):
            raise ValueError("weekly_schedule and probability_chart must have the same length")

        self.teams: List[Team] = list(teams)
        self.team_to_idx: Dict[Team, int] = {team: i for i, team in enumerate(self.teams)}
        if len(self.team_to_idx) != len(self.teams):
            raise ValueError("teams must be unique")

        self.weeks: List[WeekData] = []
        for week_i, (games, probs) in enumerate(zip(weekly_schedule, probability_chart)):
            normalized_games: List[Game] = []
            normalized_probs: Dict[Game, float] = {}
            for game in games:
                t1, t2 = game
                if t1 not in self.team_to_idx or t2 not in self.team_to_idx:
                    raise ValueError(f"week {week_i}: unknown team in game {game}")
                if t1 == t2:
                    raise ValueError(f"week {week_i}: game cannot contain same team twice: {game}")
                normalized_games.append((t1, t2))

                if (t1, t2) in probs:
                    p_t1 = probs[(t1, t2)]
                elif (t2, t1) in probs:
                    p_t1 = 1.0 - probs[(t2, t1)]
                else:
                    raise ValueError(
                        f"week {week_i}: missing probability for matchup {(t1, t2)}"
                    )
                if not (0.0 <= p_t1 <= 1.0):
                    raise ValueError(
                        f"week {week_i}: probability for game {(t1, t2)} must be in [0,1]"
                    )
                normalized_probs[(t1, t2)] = float(p_t1)
            self.weeks.append(WeekData(games=normalized_games, probabilities=normalized_probs))

        self.num_weeks = len(self.weeks)
        self.num_teams = len(self.teams)
        self.num_contestants = num_contestants
        if contestant_ids is None:
            self.contestants = [f"agent_{i}" for i in range(num_contestants)]
        else:
            if len(contestant_ids) != num_contestants:
                raise ValueError("contestant_ids length must equal num_contestants")
            if len(set(contestant_ids)) != num_contestants:
                raise ValueError("contestant_ids must be unique")
            self.contestants = list(contestant_ids)

        self.max_games_in_week = max(len(week.games) for week in self.weeks)

        self.action_space = spaces.Dict(
            {contestant: spaces.Discrete(self.num_teams) for contestant in self.contestants}
        )
        self.observation_space = spaces.Dict(
            {
                "week_index": spaces.Discrete(self.num_weeks + 1),
                "alive_mask": spaces.MultiBinary(self.num_contestants),
                "pick_history": spaces.MultiBinary((self.num_contestants, self.num_teams)),
                "games": spaces.Box(
                    low=-1,
                    high=max(0, self.num_teams - 1),
                    shape=(self.max_games_in_week, 2),
                    dtype=np.int32,
                ),
                "matchup_probabilities": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_games_in_week,),
                    dtype=np.float32,
                ),
            }
        )

        self._rng = np.random.default_rng(seed)

        self.week_index: int = 0
        self.alive: np.ndarray = np.ones(self.num_contestants, dtype=np.int8)
        self.pick_history: np.ndarray = np.zeros((self.num_contestants, self.num_teams), dtype=np.int8)
        self.last_winners: Dict[Game, Team] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.week_index = 0
        self.alive = np.ones(self.num_contestants, dtype=np.int8)
        self.pick_history = np.zeros((self.num_contestants, self.num_teams), dtype=np.int8)
        self.last_winners = {}

        observation = self._build_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: Mapping[Contestant, ActionValue]):
        if self.week_index >= self.num_weeks:
            raise RuntimeError("Episode already finished. Call reset().")

        week = self.weeks[self.week_index]
        valid_teams_this_week = {team for game in week.games for team in game}
        picks = self._normalize_action(action)

        for c_idx, contestant in enumerate(self.contestants):
            if not self.alive[c_idx]:
                continue
            if contestant not in picks:
                raise ValueError(f"Missing action for alive contestant '{contestant}'")
            team = picks[contestant]
            if team not in valid_teams_this_week:
                raise ValueError(
                    f"Invalid pick for {contestant}: '{team}' is not playing in week {self.week_index}"
                )
            t_idx = self.team_to_idx[team]
            if self.pick_history[c_idx, t_idx] == 1:
                raise ValueError(
                    f"Invalid pick for {contestant}: team '{team}' was already used in a prior week"
                )

        winners = self._simulate_week(week)
        self.last_winners = winners

        for c_idx, contestant in enumerate(self.contestants):
            if not self.alive[c_idx]:
                continue
            team = picks[contestant]
            t_idx = self.team_to_idx[team]
            self.pick_history[c_idx, t_idx] = 1
            if team != self._winning_team_for_pick(team, winners):
                self.alive[c_idx] = 0

        self.week_index += 1
        terminated = bool(self.alive.sum() <= 1 or self.week_index >= self.num_weeks)
        truncated = False
        rewards = self._terminal_rewards() if terminated else {c: 0.0 for c in self.contestants}

        observation = self._build_observation()
        info = self._build_info()
        return observation, rewards, terminated, truncated, info

    def render(self):
        alive_contestants = [
            contestant
            for i, contestant in enumerate(self.contestants)
            if self.alive[i]
        ]
        print(
            f"Week={self.week_index}/{self.num_weeks} | Alive={alive_contestants} | Last winners={self.last_winners}"
        )

    def _normalize_action(self, action: Mapping[Contestant, ActionValue]) -> Dict[Contestant, Team]:
        normalized: Dict[Contestant, Team] = {}
        for contestant, raw_pick in action.items():
            if contestant not in self.contestants:
                raise ValueError(f"Unknown contestant in action: {contestant}")
            if isinstance(raw_pick, str):
                pick = raw_pick
            elif isinstance(raw_pick, (int, np.integer)):
                idx = int(raw_pick)
                if not (0 <= idx < self.num_teams):
                    raise ValueError(f"Action index for {contestant} out of range: {idx}")
                pick = self.teams[idx]
            else:
                raise ValueError(
                    f"Action for {contestant} must be team name (str) or team index (int), got {type(raw_pick)}"
                )
            if pick not in self.team_to_idx:
                raise ValueError(f"Unknown team picked by {contestant}: {pick}")
            normalized[contestant] = pick
        return normalized

    def _simulate_week(self, week: WeekData) -> Dict[Game, Team]:
        winners: Dict[Game, Team] = {}
        for game in week.games:
            t1, t2 = game
            p_t1 = week.probabilities[game]
            winners[game] = t1 if self._rng.random() < p_t1 else t2
        return winners

    def _winning_team_for_pick(self, picked_team: Team, winners: Mapping[Game, Team]) -> Optional[Team]:
        for (t1, t2), winner in winners.items():
            if picked_team == t1 or picked_team == t2:
                return winner
        return None

    def _terminal_rewards(self) -> Dict[Contestant, float]:
        reward_pool = float(self.num_contestants - 1)
        alive_indices = [i for i in range(self.num_contestants) if self.alive[i] == 1]
        rewards = {contestant: 0.0 for contestant in self.contestants}
        if len(alive_indices) == 0:
            return rewards
        reward_per_winner = reward_pool / len(alive_indices)
        for i in alive_indices:
            rewards[self.contestants[i]] = reward_per_winner
        return rewards

    def _build_observation(self) -> Dict[str, np.ndarray]:
        games = np.full((self.max_games_in_week, 2), -1, dtype=np.int32)
        probs = np.zeros((self.max_games_in_week,), dtype=np.float32)

        if self.week_index < self.num_weeks:
            week = self.weeks[self.week_index]
            for i, (t1, t2) in enumerate(week.games):
                games[i, 0] = self.team_to_idx[t1]
                games[i, 1] = self.team_to_idx[t2]
                probs[i] = week.probabilities[(t1, t2)]

        return {
            "week_index": np.array(self.week_index, dtype=np.int32),
            "alive_mask": self.alive.copy(),
            "pick_history": self.pick_history.copy(),
            "games": games,
            "matchup_probabilities": probs,
        }

    def _build_info(self) -> Dict[str, Any]:
        prior_picks: Dict[Contestant, List[Team]] = {}
        for c_idx, contestant in enumerate(self.contestants):
            used = [
                self.teams[t_idx]
                for t_idx in range(self.num_teams)
                if self.pick_history[c_idx, t_idx] == 1
            ]
            prior_picks[contestant] = used

        games_for_week = self.weeks[self.week_index].games if self.week_index < self.num_weeks else []
        probabilities_for_week = (
            self.weeks[self.week_index].probabilities if self.week_index < self.num_weeks else {}
        )

        return {
            "contestants": self.contestants,
            "alive_contestants": [
                self.contestants[i] for i in range(self.num_contestants) if self.alive[i] == 1
            ],
            "games_for_week": games_for_week,
            "probabilities_for_week": probabilities_for_week,
            "prior_picks": prior_picks,
            "last_winners": self.last_winners,
        }
