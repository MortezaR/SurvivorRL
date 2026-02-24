from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import torch

from nfl_survivor_agent import (
    BareBonesPickerNet,
    Config as AgentFeatureConfig,
    featurize,
)
from survivor_schedule import (
    FeatureMatchupRow,
    build_round_robin_schedule,
    extract_weekly_winners,
    sample_game_outcomes,
)


@dataclass
class SurvivorConfig:
    num_agents: int = 1_000
    num_teams: int = 32
    max_weeks: int = 18


@dataclass
class SurvivorGameResult:
    weeks_played: int
    winner_ids: List[int]
    survivors: List[int]
    survivors_by_week: List[List[int]]
    winning_teams_by_week: List[List[int]]
    eliminated_by_week: List[int]


class SurvivorEnvironment:
    def __init__(self, cfg: SurvivorConfig) -> None:
        self.cfg = cfg

        self.feature_cfg = AgentFeatureConfig(
            max_contestants=cfg.num_agents,
            max_teams=cfg.num_teams,
            max_weeks=cfg.max_weeks,
        )
        sample_x = featurize(
            cfg=self.feature_cfg,
            contestant_picks={},
            matchup_table=[],
            agent_id=0,
            current_week=0,
        )
        input_dim = int(sample_x.numel())

        self.agents = [
            BareBonesPickerNet(input_dim=input_dim, num_teams=cfg.num_teams, agent_id=agent_id)
            for agent_id in range(cfg.num_agents)
        ]
        for model in self.agents:
            model.eval()

    def play_game(
        self,
        matchup_table: List[FeatureMatchupRow],
        winning_teams_by_week: List[List[int]],
    ) -> SurvivorGameResult:
        rows_by_week: List[List[FeatureMatchupRow]] = [[] for _ in range(self.cfg.max_weeks)]
        for week_id, team_id, win_prob in matchup_table:
            if 0 <= week_id < self.cfg.max_weeks:
                rows_by_week[week_id].append((week_id, team_id, win_prob))

        matchup_rows_seen: List[FeatureMatchupRow] = []
        active_agents = list(range(self.cfg.num_agents))
        picked_teams_by_agent = [set() for _ in range(self.cfg.num_agents)]
        survivors_by_week: List[List[int]] = []
        eliminated_by_week: List[int] = []

        with torch.no_grad():
            for week_id in range(self.cfg.max_weeks):
                if len(active_agents) <= 1:
                    break
                round_start_agents = active_agents.copy()

                matchup_rows_seen.extend(rows_by_week[week_id])
                week_winners = set()
                if week_id < len(winning_teams_by_week):
                    week_winners = set(winning_teams_by_week[week_id])

                next_active_agents: List[int] = []
                contestant_picks = {}
                for agent_id in active_agents:
                    prior_picks = picked_teams_by_agent[agent_id]
                    if len(prior_picks) >= self.cfg.num_teams:
                        continue

                    x = featurize(
                        cfg=self.feature_cfg,
                        contestant_picks=contestant_picks,
                        matchup_table=matchup_rows_seen,
                        agent_id=agent_id,
                        current_week=week_id,
                    )
                    pick_probs = self.agents[agent_id](
                        x,
                        unavailable_team_ids=list(prior_picks),
                    )
                    picked_team = int(torch.multinomial(pick_probs, num_samples=1).item())
                    picked_teams_by_agent[agent_id].add(picked_team)
                    contestant_picks[agent_id] = picked_team
                    if picked_team in week_winners:
                        next_active_agents.append(agent_id)

                if len(next_active_agents) == 0 and len(round_start_agents) > 0:
                    # If everyone is eliminated in the final played round,
                    # treat all round entrants as co-winners.
                    active_agents = round_start_agents
                    eliminated_by_week.append(0)
                    survivors_by_week.append(active_agents.copy())
                    break

                eliminated_by_week.append(len(active_agents) - len(next_active_agents))
                active_agents = next_active_agents
                survivors_by_week.append(active_agents.copy())

        winner_ids = active_agents.copy()
        return SurvivorGameResult(
            weeks_played=len(survivors_by_week),
            winner_ids=winner_ids,
            survivors=active_agents,
            survivors_by_week=survivors_by_week,
            winning_teams_by_week=winning_teams_by_week,
            eliminated_by_week=eliminated_by_week,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple survivor simulation with 1,000 agents.")
    parser.add_argument("--num-agents", type=int, default=1_000)
    parser.add_argument("--num-teams", type=int, default=32)
    parser.add_argument("--max-weeks", type=int, default=18)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = SurvivorConfig(
        num_agents=args.num_agents,
        num_teams=args.num_teams,
        max_weeks=args.max_weeks,
    )

    # Matchups and game winners are generated outside the environment.
    schedule = build_round_robin_schedule(
        num_weeks=cfg.max_weeks,
        num_teams=cfg.num_teams,
    )
    outcomes_by_week = sample_game_outcomes(schedule.games_by_week)
    winning_teams_by_week = extract_weekly_winners(outcomes_by_week)

    env = SurvivorEnvironment(cfg)
    result = env.play_game(
        matchup_table=schedule.feature_rows,
        winning_teams_by_week=winning_teams_by_week,
    )

    print(f"Weeks played: {result.weeks_played}")
    print(f"Num winners: {len(result.winner_ids)}")
    for week_index, survivors in enumerate(result.survivors_by_week):
        print(f"Week {week_index} survivors: {survivors}")


if __name__ == "__main__":
    main()
