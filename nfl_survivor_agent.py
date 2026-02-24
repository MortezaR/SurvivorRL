from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

@dataclass
class Config:
    max_contestants: int
    max_teams: int
    max_weeks: int

def featurize(
    cfg: Config,
    contestant_picks: Dict[int, int],               # {contestant_id: picked_team_id}, active only
    matchups_by_week: List[List[Tuple[int, float]]],# week w: [(team_id, win_prob), ...]
    agent_id: int,                                   # contestant_id
) -> torch.Tensor:
    """
    Returns a single feature vector x: [D]

    Encodes:
      1) picks_mat: [C, T] one-hot picks of active contestants
      2) matchup_odds: [W, T] per-week per-team win probabilities (0 if not playing)
      3) agent_oh: [C] one-hot agent id
    """
    C, T, W = cfg.max_contestants, cfg.max_teams, cfg.max_weeks

    # pick a device from any tensor input if possible; otherwise CPU
    device = None
    for week in matchups_by_week:
        if len(week) > 0 and isinstance(week[0][1], torch.Tensor):
            device = week[0][1].device
            break
    if device is None:
        device = torch.device("cpu")

    # 1) contestant->picked team matrix: [C, T]
    picks_mat = torch.zeros((C, T), device=device)
    for cid, tid in contestant_picks.items():
        if 0 <= cid < C and 0 <= tid < T:
            picks_mat[cid, tid] = 1.0

    # 2) matchup odds table: [W, T]
    # Each entry is "odds of team t winning in week w" (0 if not playing / unknown)
    matchup_odds = torch.zeros((W, T), device=device)
    for w, entries in enumerate(matchups_by_week[:W]):
        for team_id, win_prob in entries:
            if 0 <= team_id < T:
                matchup_odds[w, team_id] = float(win_prob)

    # 3) agent id one-hot: [C]
    agent_oh = torch.zeros((C,), device=device)
    if 0 <= agent_id < C:
        agent_oh[agent_id] = 1.0

    # Flatten into one vector
    x = torch.cat([
        picks_mat.flatten(),      # C*T
        matchup_odds.flatten(),   # W*T
        agent_oh.flatten(),       # C
    ], dim=0)

    return x  # [D]

class BareBonesPickerNet(nn.Module):
    def __init__(self, input_dim, num_teams):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_teams)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# ---------------------------
# Example wiring
# ---------------------------
if __name__ == "__main__":
    cfg = Config(max_contestants=200, max_teams=32, max_weeks=18)
    C, T, W = cfg.max_contestants, cfg.max_teams, cfg.max_weeks

    # Example inputs (replace with real data)
    contestant_picks = {0: 3, 7: 10, 15: 3}     # active only
    prob_chart = torch.rand(T, T)               # placeholder; fill with your matrix
    matchups_by_week = [
        [(3, 10), (2, 5)],   # week 0
        [(1, 7)],            # week 1
    ]
    agent_id = 7

    x = featurize(cfg, contestant_picks, prob_chart, matchups_by_week, agent_id)
    model = BareBonesPickerNet(input_dim=x.numel(), num_teams=T, hidden=128)

    p_pick = model(x)  # [T], sums to 1
    print("Pick distribution:", p_pick)
    print("Sum:", p_pick.sum().detach().item())