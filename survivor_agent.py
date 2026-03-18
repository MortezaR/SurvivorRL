import copy
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class Config:
    num_contestants: int
    num_teams: int
    num_weeks: int

MatchupRow = Tuple[int, int, float]  # (week_id, team_id, win_probability)
PICKS_FEATURE_ROWS = 2

def featurize(
    cfg: Config,
    contestant_picks: Dict[int, List[int]],  # {contestant_id: [picked_team_id, ...]}, active only
    matchup_table: List[MatchupRow],   # [(week_id, team_id, win_prob), ...]
    agent_id: int,                     # contestant_id
    current_week: int,                 # current week index
    num_players: [int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns a single feature vector x: [D]

    Encodes:
      1) picks_mat: [2, T]
         - row 0: one-hot picks for this agent
         - row 1: elementwise average of one-hot picks for all other active agents
      2) matchup_odds: [W, T] week x team win probabilities
      3) current_week: [W] one-hot current week
    """
    T, W = cfg.num_teams, cfg.num_weeks

    # If no device is provided, default to GPU.
    if device is None:
        device = torch.device("cuda")

    num_players = torch.tensor(num_players, device=device, dtype=dtype)

    # 1) contestant pick features: [2, T]
    picks_mat = torch.zeros((PICKS_FEATURE_ROWS, T), device=device, dtype=dtype)
    # Row 0 is always this agent.
    self_picks = contestant_picks.get(agent_id, [])
    for tid in self_picks:
        picks_mat[0, tid] = 1.0

    other_ids = [cid for cid in contestant_picks.keys() if cid != agent_id]
    if other_ids:
        for cid in other_ids:
            for tid in contestant_picks[cid]:
                picks_mat[1, tid] += 1.0
        picks_mat[1] /= len(other_ids)

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
        picks_mat.flatten(),      # 2*T
        matchup_odds.flatten(),   # W*T
        current_week_oh.flatten(),   # W
        num_players.flatten()
    ], dim=0)

    return x  # [D]


def build_matchup_odds_tensor(
    cfg: Config,
    matchup_table: List[MatchupRow],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns matchup odds as [W, T] using the same layout as featurize().
    """

    T, W = cfg.num_teams, cfg.num_weeks
    if device is None:
        device = torch.device("cuda")

    matchup_odds = torch.zeros((W, T), device=device, dtype=dtype)
    for week_id, team_id, win_prob in matchup_table:
        if 0 <= week_id < W and 0 <= team_id < T:
            if isinstance(win_prob, torch.Tensor):
                matchup_odds[week_id, team_id] = win_prob.to(device=device, dtype=dtype)
            else:
                matchup_odds[week_id, team_id] = float(win_prob)

    return matchup_odds


def featurize_population(
    cfg: Config,
    contestant_pick_history: torch.Tensor,  # [B, T], bool or float
    matchup_odds: torch.Tensor,             # [W, T]
    current_week: int,
    num_players: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Batched version of featurize() that preserves the same feature layout.
    """

    batch_size = contestant_pick_history.shape[0]
    device = matchup_odds.device

    self_picks = contestant_pick_history.to(device=device, dtype=dtype)
    if batch_size > 1:
        other_pick_totals = self_picks.sum(dim=0, keepdim=True) - self_picks
        other_picks = other_pick_totals / float(batch_size - 1)
    else:
        other_picks = torch.zeros_like(self_picks)

    current_week_oh = torch.zeros((1, cfg.num_weeks), device=device, dtype=dtype)
    if 0 <= current_week < cfg.num_weeks:
        current_week_oh[0, current_week] = 1.0

    matchup_features = matchup_odds.reshape(1, -1).expand(batch_size, -1)
    week_features = current_week_oh.expand(batch_size, -1)
    num_players_feature = torch.full(
        (batch_size, 1),
        float(num_players),
        device=device,
        dtype=dtype,
    )

    return torch.cat(
        [
            self_picks,
            other_picks,
            matchup_features,
            week_features,
            num_players_feature,
        ],
        dim=1,
    )


def _stacked_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:

    return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias


def population_policy_forward(
    cfg: Config,
    parameter_tensors: Mapping[str, torch.Tensor],
    active_indices: torch.Tensor,
    contestant_pick_history: torch.Tensor,  # [B, T]
    matchup_odds: torch.Tensor,             # [W, T]
    current_week: int,
    num_players: int,
    unavailable_team_mask: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    """
    Batched equivalent of PickerNet.forward() over a slice of population tensors.
    """

    model_device = parameter_tensors["fc1.weight"].device
    model_dtype = parameter_tensors["fc1.weight"].dtype
    row_indices = active_indices.to(device=model_device, dtype=torch.long)

    x = featurize_population(
        cfg=cfg,
        contestant_pick_history=contestant_pick_history,
        matchup_odds=matchup_odds,
        current_week=current_week,
        num_players=num_players,
        dtype=model_dtype,
    )

    hidden_1 = torch.relu(
        _stacked_linear(
            x,
            parameter_tensors["fc1.weight"].index_select(0, row_indices),
            parameter_tensors["fc1.bias"].index_select(0, row_indices),
        )
    )
    hidden_2 = torch.relu(
        _stacked_linear(
            hidden_1,
            parameter_tensors["fc2.weight"].index_select(0, row_indices),
            parameter_tensors["fc2.bias"].index_select(0, row_indices),
        )
    )
    logits = _stacked_linear(
        hidden_2,
        parameter_tensors["fc3.weight"].index_select(0, row_indices),
        parameter_tensors["fc3.bias"].index_select(0, row_indices),
    )

    blocked_mask = unavailable_team_mask.to(device=model_device, dtype=torch.bool)
    if blocked_mask.numel() > 0:
        rows_with_available_teams = ~blocked_mask.all(dim=1, keepdim=True)
        effective_blocked_mask = blocked_mask & rows_with_available_teams
        if effective_blocked_mask.any():
            logits = logits.masked_fill(effective_blocked_mask, -1e9)

    return torch.softmax(logits, dim=-1)

class PickerNet(nn.Module):
    def __init__(self, agent_id, cfg: Config):
        super().__init__()
        self.agent_id = agent_id
        self.cfg = cfg
        self.input_dim = (
            (PICKS_FEATURE_ROWS * cfg.num_teams)
            + (cfg.num_weeks * cfg.num_teams)
            + cfg.num_weeks + 1
        )
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, cfg.num_teams)

    def forward(
        self,
        contestant_picks: Dict[int, List[int]],
        matchup_table: List[MatchupRow],
        current_week: int,
        num_players: int,
        unavailable_team_ids: List[int],
    ):
        model_device = self.fc1.weight.device
        model_dtype = self.fc1.weight.dtype
        x = featurize(
            cfg=self.cfg,
            contestant_picks=contestant_picks,
            matchup_table=matchup_table,
            agent_id=self.agent_id,
            current_week=current_week,
            num_players=num_players,
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
