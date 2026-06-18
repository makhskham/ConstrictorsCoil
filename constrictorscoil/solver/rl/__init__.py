import os

import numpy as np
import torch
import torch.nn as nn

from constrictorscoil.base.direc import Direc
from constrictorscoil.base.pos import Pos
from constrictorscoil.solver.base import BaseSolver

_GRID = 6
_NUM_CH = 4
_EXTRA = 9
_MLP_IN = 64 * _GRID * _GRID + _EXTRA  # 2313: 64 conv filters * 36 cells + 9 extra
_ACTIONS = 3

_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "rl_model.pt")
)

_DIR_IDX = {Direc.UP: 0, Direc.LEFT: 1, Direc.DOWN: 2, Direc.RIGHT: 3}

_TURN_LEFT = {
    Direc.UP: Direc.LEFT,
    Direc.LEFT: Direc.DOWN,
    Direc.DOWN: Direc.RIGHT,
    Direc.RIGHT: Direc.UP,
}
_TURN_RIGHT = {
    Direc.UP: Direc.RIGHT,
    Direc.RIGHT: Direc.DOWN,
    Direc.DOWN: Direc.LEFT,
    Direc.LEFT: Direc.UP,
}


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(_NUM_CH, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(_MLP_IN, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, _ACTIONS),
        )

    def forward(self, grid, extra):
        x = self.conv(grid)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra], dim=1)
        return self.mlp(x)


def _encode(snake):
    game_map = snake.map
    head = snake.head()
    food = game_map.food
    slen = snake.len()
    # 1-based index from head: head=1, tail=slen
    body_idx = {(p.x, p.y): i + 1 for i, p in enumerate(snake.bodies)}

    grid = np.zeros((_NUM_CH, _GRID, _GRID), dtype=np.float32)
    for r in range(_GRID):
        for c in range(_GRID):
            # playfield is 1-indexed in new map (walls at border)
            pos = Pos(r + 1, c + 1)
            if pos == food:
                grid[0][r][c] = 1.0
            key = (pos.x, pos.y)
            if key in body_idx:
                idx = body_idx[key]
                grid[1][r][c] = idx / slen
                if idx == 1:
                    grid[2][r][c] = 1.0
                if idx != slen:  # danger: occupied and not the tail
                    grid[3][r][c] = 1.0

    extra = np.zeros(_EXTRA, dtype=np.float32)
    d_idx = _DIR_IDX.get(snake.direc, -1)
    if d_idx >= 0:
        extra[d_idx] = 1.0

    # x increases downward, y increases rightward
    dx = food.x - head.x
    dy = food.y - head.y
    extra[4] = 1.0 if dx < 0 else 0.0  # food is up
    extra[5] = 1.0 if dy < 0 else 0.0  # food is left
    extra[6] = 1.0 if dx > 0 else 0.0  # food is down
    extra[7] = 1.0 if dy > 0 else 0.0  # food is right
    extra[8] = (abs(dx) + abs(dy)) / (2.0 * _GRID)

    return grid, extra


class RLSolver(BaseSolver):
    def __init__(self, snake):
        super().__init__(snake)
        self._net = _Net()
        self._net.load_state_dict(torch.load(_MODEL_PATH, weights_only=True))
        self._net.eval()

    def next_direc(self):
        grid, extra = _encode(self._snake)
        g = torch.tensor(grid).unsqueeze(0)
        e = torch.tensor(extra).unsqueeze(0)
        with torch.no_grad():
            q = self._net(g, e)
        action = q.argmax(dim=1).item()
        cur = self._snake.direc
        if action == 0:
            return cur
        elif action == 1:
            return _TURN_LEFT[cur]
        else:
            return _TURN_RIGHT[cur]
