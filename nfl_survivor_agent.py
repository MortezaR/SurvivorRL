import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class Config:
    max_contestants: int
    max_teams: int
    max_weeks: int

MatchupRow = Tuple[int, int, float]  # (week_id, team_id, win_probability)

def featurize(
    cfg: Config,
    contestant_picks: Dict[int, int],  # {contestant_id: picked_team_id}, active only
    matchup_table: List[MatchupRow],   # [(week_id, team_id, win_prob), ...]
    agent_id: int,                     # contestant_id
    current_week: int,                 # current week index
) -> torch.Tensor:
    """
    Returns a single feature vector x: [D]

    Encodes:
      1) picks_mat: [C, T] one-hot picks of active contestants
      2) matchup_odds: [W, T] week x team win probabilities
      3) agent_oh: [C] one-hot agent id
      4) current_week: [W] one-hot current week
    """
    C, T, W = cfg.max_contestants, cfg.max_teams, cfg.max_weeks

    # pick a device from any tensor input if possible; otherwise CPU
    device = None
    for _, _, win_prob in matchup_table:
        if isinstance(win_prob, torch.Tensor):
            device = win_prob.device
            break
    if device is None:
        device = torch.device("cpu")

    # 1) contestant->picked team matrix: [C, T]
    picks_mat = torch.zeros((C, T), device=device)
    for cid, tid in contestant_picks.items():
        if 0 <= cid < C and 0 <= tid < T:
            picks_mat[cid, tid] = 1.0

    # 2) matchup odds table: [W, T]
    matchup_odds = torch.zeros((W, T), device=device)
    for week_id, team_id, win_prob in matchup_table:
        if 0 <= week_id < W and 0 <= team_id < T:
            matchup_odds[week_id, team_id] = float(win_prob)

    # 3) agent id one-hot: [C]
    agent_oh = torch.zeros((C,), device=device)
    if 0 <= agent_id < C:
        agent_oh[agent_id] = 1.0

    # 4) current week one-hot: [W]
    current_week_oh = torch.zeros((W,), device=device)
    if 0 <= current_week < W:
        current_week_oh[current_week] = 1.0


    # Flatten into one vector
    x = torch.cat([
        picks_mat.flatten(),      # C*T
        matchup_odds.flatten(),   # W*T
        agent_oh.flatten(),       # C
        current_week_oh.flatten(),   # W
    ], dim=0)

    return x  # [D]

class BareBonesPickerNet(nn.Module):
    def __init__(self, input_dim, num_teams, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_teams)

    def forward(self, x, unavailable_team_ids: Optional[List[int]] = None):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
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
        BareBonesPickerNet.add_gaussian_noise(child, std)
        return child

# ---------------------------
# Example wiring
# ---------------------------
if __name__ == "__main__":
    cfg = Config(max_contestants=200, max_teams=32, max_weeks=18)
    C, T, W = cfg.max_contestants, cfg.max_teams, cfg.max_weeks

    # Example inputs (replace with real data)
    contestant_picks = {0: 3, 7: 10, 15: 3}     # active only
    matchup_table = [
        (0, 3, 0.55),  # week 0, team 3, win probability
        (0, 2, 0.48),  # week 0, team 2, win probability
        (1, 1, 0.62),  # week 1, team 1, win probability
    ]
    agent_id = 7

    x = featurize(cfg, contestant_picks, matchup_table, agent_id, current_week=0)
    model = BareBonesPickerNet(input_dim=x.numel(), num_teams=T, agent_id=agent_id)

    p_pick = model(x)  # [T], sums to 1
    print("Pick distribution:", p_pick)
    print("Sum:", p_pick.sum().detach().item())
