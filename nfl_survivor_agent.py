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
    contestant_picks: Dict[int, int],         # {contestant_id: picked_team_id}, active only
    prob_chart: torch.Tensor,                  # [T, T] float, P(team_i beats team_j)
    matchups_by_week: List[List[Tuple[int,int]]],  # len<=W, each week: [(team_a, team_b), ...]
    agent_id: int,                             # contestant_id
) -> torch.Tensor:
    """
    Returns a single feature vector x: [D]
    """

    C, T, W = cfg.max_contestants, cfg.max_teams, cfg.max_weeks
    device = prob_chart.device

    # 1) contestant->picked team matrix: [C, T]
    picks_mat = torch.zeros((C, T), device=device)
    for cid, tid in contestant_picks.items():
        if 0 <= cid < C and 0 <= tid < T:
            picks_mat[cid, tid] = 1.0

    # 2) prob chart: [T, T]
    probs = prob_chart
    assert probs.shape == (T, T)

    # 3) matchups: [W, T, T] (undirected adjacency for each week)
    match_adj = torch.zeros((W, T, T), device=device)
    for w, games in enumerate(matchups_by_week[:W]):
        for a, b in games:
            if 0 <= a < T and 0 <= b < T:
                match_adj[w, a, b] = 1.0
                match_adj[w, b, a] = 1.0

    # 4) agent id one-hot: [C]
    agent_oh = torch.zeros((C,), device=device)
    if 0 <= agent_id < C:
        agent_oh[agent_id] = 1.0

    # Flatten everything into one vector
    x = torch.cat([
        picks_mat.flatten(),      # C*T
        probs.flatten(),          # T*T
        match_adj.flatten(),      # W*T*T
        agent_oh.flatten(),       # C
    ], dim=0)

    return x  # [D]

class BareBonesPickerNet(nn.Module):
    """
    Output: per-team sigmoid scores, then normalized to sum to 1 (a distribution).
    User request conflicts slightly: sigmoid != distribution by itself, so we do:
      p_raw = sigmoid(...)
      p = p_raw / sum(p_raw)
    """
    def __init__(self, input_dim: int, num_teams: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_teams),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [D] or [B, D]
        logits = self.net(x)
        p_raw = torch.sigmoid(logits)  # [T] or [B, T]
        p = p_raw / (p_raw.sum(dim=-1, keepdim=True) + 1e-8)
        return p

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
    print("Sum:", float(p_pick.sum()))