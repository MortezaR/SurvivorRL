import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class Config:
    num_contestants: int
    num_teams: int
    num_weeks: int

MatchupRow = Tuple[int, int, float]  # (week_id, team_id, win_probability)
PICKS_CONTESTANT_CAP = 100

def featurize(
    cfg: Config,
    contestant_picks: Dict[int, List[int]],  # {contestant_id: [picked_team_id, ...]}, active only
    matchup_table: List[MatchupRow],   # [(week_id, team_id, win_prob), ...]
    agent_id: int,                     # contestant_id
    current_week: int,                 # current week index
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns a single feature vector x: [D]

    Encodes:
      1) picks_mat: [100, T] one-hot picks of active contestants (capped),
         with row 0 always reserved for this agent
      2) matchup_odds: [W, T] week x team win probabilities
      3) current_week: [W] one-hot current week
    """
    C, T, W = cfg.num_contestants, cfg.num_teams, cfg.num_weeks

    # If no device is provided, default to GPU.
    if device is None:
        device = torch.device("cuda")

    # 1) contestant->picked team matrix: [100, T]
    # If contestants exceed cap, leave picks input as zeros.
    picks_mat = torch.zeros((PICKS_CONTESTANT_CAP, T), device=device, dtype=dtype)
    # Row 0 is always this agent.
    self_picks = contestant_picks.get(agent_id, [])
    for tid in self_picks:
        picks_mat[0, tid] = 1.0

    if C <= PICKS_CONTESTANT_CAP:
        max_rows = C
        # Remaining rows are other contestants in stable id order.
        other_ids = sorted(cid for cid in contestant_picks.keys())
        row = 1
        for cid in other_ids:
            for tid in contestant_picks[cid]:
                picks_mat[row, tid] = 1.0
            row += 1

    # 2) matchup odds table: [W, T]
    matchup_odds = torch.zeros((W, T), device=device, dtype=dtype)
    for week_id, team_id, win_prob in matchup_table:
        if 0 <= week_id < W and 0 <= team_id < T:
            if isinstance(win_prob, torch.Tensor):
                matchup_odds[week_id, team_id] = win_prob.to(device=device, dtype=dtype)
            else:
                matchup_odds[week_id, team_id] = float(win_prob)

    # 3) current week one-hot: [W]
    current_week_oh = torch.zeros((W,), device=device, dtype=dtype)
    if 0 <= current_week < W:
        current_week_oh[current_week] = 1.0


    # Flatten into one vector
    x = torch.cat([
        picks_mat.flatten(),      # 100*T
        matchup_odds.flatten(),   # W*T
        current_week_oh.flatten(),   # W
    ], dim=0)

    return x  # [D]

class PickerNet(nn.Module):
    def __init__(self, agent_id, cfg: Config):
        super().__init__()
        self.agent_id = agent_id
        self.cfg = cfg
        self.input_dim = (
            (PICKS_CONTESTANT_CAP * cfg.num_teams)
            + (cfg.num_weeks * cfg.num_teams)
            + cfg.num_weeks
        )
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, cfg.num_teams)

    def forward(
        self,
        contestant_picks: Dict[int, List[int]],
        matchup_table: List[MatchupRow],
        current_week: int,
        unavailable_team_ids: Optional[List[int]] = None,
    ):
        model_device = self.fc1.weight.device
        model_dtype = self.fc1.weight.dtype
        x = featurize(
            cfg=self.cfg,
            contestant_picks=contestant_picks,
            matchup_table=matchup_table,
            agent_id=self.agent_id,
            current_week=current_week,
            device=model_device,
            dtype=model_dtype,
        )
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        if unavailable_team_ids:
            blocked = [tid for tid in unavailable_team_ids if 0 <= tid < logits.shape[-1]]
            if blocked and len(blocked) < logits.shape[-1]:
                logits = logits.clone()
                logits[blocked] = -1e9
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def add_gaussian_noise(model: torch.nn.Module, std: float = 0.01):
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * std
                    param.add_(noise)

    @staticmethod
    def mutated_copy(model: torch.nn.Module, std: float = 0.01):
        child = copy.deepcopy(model)
        PickerNet.add_gaussian_noise(child, std)
        return child
