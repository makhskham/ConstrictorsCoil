import os
import random
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch

from constrictorscoil.base import Direc, Map, PointType, Pos, Snake
from constrictorscoil.base.point import PointType
from constrictorscoil.solver.base import BaseSolver

_GRID_ROWS = 8
_GRID_COLS = 8
_NUM_GRID_CH = 4
_NUM_EXTRA = 9
_NUM_ACTIONS = 3

_DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
_DEVICE = torch.device(_DEVICE_NAME)

_INIT_DIREC = Direc.RIGHT
_INIT_BODIES = [Pos(1, 4), Pos(1, 3), Pos(1, 2), Pos(1, 1)]
_INIT_TYPES = [PointType.HEAD_R, PointType.BODY_HOR, PointType.BODY_HOR, PointType.BODY_HOR]

_HEAD_TYPES = (PointType.HEAD_L, PointType.HEAD_U, PointType.HEAD_R, PointType.HEAD_D)
_BODY_TYPES = (
    PointType.BODY_LU, PointType.BODY_UR, PointType.BODY_RD,
    PointType.BODY_DL, PointType.BODY_HOR, PointType.BODY_VER,
)

_DIREC_IDX = {Direc.UP: 0, Direc.LEFT: 1, Direc.DOWN: 2, Direc.RIGHT: 3}

_TURN_LEFT = {
    Direc.LEFT: Direc.DOWN, Direc.DOWN: Direc.RIGHT,
    Direc.RIGHT: Direc.UP, Direc.UP: Direc.LEFT,
}
_TURN_RIGHT = {
    Direc.LEFT: Direc.UP, Direc.UP: Direc.RIGHT,
    Direc.RIGHT: Direc.DOWN, Direc.DOWN: Direc.LEFT,
}


def _action_to_direc(action, cur_direc):
    if action == 0:
        return cur_direc
    if action == 1:
        return _TURN_LEFT[cur_direc]
    return _TURN_RIGHT[cur_direc]


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(_NUM_GRID_CH, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        conv_out = 64 * _GRID_ROWS * _GRID_COLS
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(conv_out + _NUM_EXTRA, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, _NUM_ACTIONS),
        )

    def forward(self, x):
        grid_len = _NUM_GRID_CH * _GRID_ROWS * _GRID_COLS
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        grid_2d = x[:, :grid_len].view(-1, _NUM_GRID_CH, _GRID_ROWS, _GRID_COLS)
        conv_out = self.conv(grid_2d).view(grid_2d.size(0), -1)
        q_vals = self.mlp(torch.cat((conv_out, x[:, grid_len:]), dim=1))
        if squeeze:
            q_vals = q_vals.squeeze(0)
        return q_vals


class _Exp:
    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    @staticmethod
    def batch(exps):
        states = torch.stack([e.state for e in exps]).to(_DEVICE)
        actions = torch.tensor([e.action for e in exps]).to(_DEVICE)
        rewards = torch.tensor([e.reward for e in exps], dtype=torch.float).to(_DEVICE)
        next_states = torch.stack([e.next_state for e in exps]).to(_DEVICE)
        dones = torch.tensor([e.done for e in exps], dtype=torch.float).to(_DEVICE)
        return states, actions, rewards, next_states, dones


class DQNTorchSolver(BaseSolver):
    MODEL_PATH = os.path.join(".", "model.pt")

    NUM_EPISODES = 100_000
    NUM_EPISODES_FOR_AVG = 200
    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY_EPISODES = 50_000
    MAX_STEPS_PER_EPISODE = 500
    STEPS_PER_LEARN = 4
    MAX_STEPS_NO_FOOD_BASE = _GRID_ROWS * _GRID_COLS
    MAX_MEM = 100_000
    BATCH = 64
    LR = 0.001
    SOFT_UPDATE = 0.005
    GAMMA = 0.99
    MAX_GRAD = 10.0

    def __init__(self, snake):
        if snake is not None:
            super().__init__(snake)
        self._eval_net = None

    def next_direc(self):
        if self._eval_net is None:
            self._eval_net = self._load_net()
        state = self._make_state(self.snake, self.map)
        with torch.no_grad():
            q_vals = self._eval_net(state.to(_DEVICE))
        return _action_to_direc(int(q_vals.argmax().item()), self.snake.direc)

    def close(self):
        pass

    def train_standalone(self):
        q_net = _Net().to(_DEVICE)
        q_target = _Net().to(_DEVICE)
        q_target.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=self.LR)
        mem = deque(maxlen=self.MAX_MEM)

        epsilon = self.EPSILON_INIT
        reward_hist = []
        score_hist = []
        start = time.monotonic()

        num_params = sum(p.numel() for p in q_net.parameters())
        num_mb = sum(p.numel() * p.element_size() for p in q_net.parameters()) / (1024 * 1024)
        print(f"Model: {num_params:,} parameters ({num_mb:.1f} MB) | Device: {_DEVICE_NAME}")
        print("-" * 97)

        for episode in range(self.NUM_EPISODES):
            snake, game_map = self._new_snake()
            state = self._make_state(snake, game_map)
            tot_reward = 0.0
            steps_no_food = 0
            prev_len = snake.len()

            for step in range(self.MAX_STEPS_PER_EPISODE):
                q_vals = q_net(state.to(_DEVICE))
                action = self._pick_action(q_vals, epsilon)
                direc = _action_to_direc(action, snake.direc)

                if not game_map.has_food():
                    game_map.create_rand_food()
                snake.move(direc)
                if not game_map.has_food() and not snake.dead and not game_map.is_full():
                    game_map.create_rand_food()

                done = snake.dead or game_map.is_full()
                next_state = self._make_state(snake, game_map)

                reward = -0.01
                if snake.dead:
                    reward = -100.0
                elif game_map.is_full():
                    reward = 100.0
                elif snake.len() > prev_len:
                    reward = 10.0
                    prev_len = snake.len()
                    steps_no_food = 0
                else:
                    steps_no_food += 1

                if steps_no_food >= self.MAX_STEPS_NO_FOOD_BASE + snake.len():
                    reward = -100.0
                    done = True

                mem.append(_Exp(state, action, reward, next_state, done))
                state = next_state
                tot_reward += reward

                if len(mem) >= self.BATCH and (step + 1) % self.STEPS_PER_LEARN == 0:
                    self._learn(random.sample(mem, self.BATCH), q_net, q_target, optimizer)

                if done:
                    break

            for q_p, t_p in zip(q_net.parameters(), q_target.parameters()):
                t_p.data.copy_(self.SOFT_UPDATE * q_p.data + (1 - self.SOFT_UPDATE) * t_p.data)

            epsilon = max(
                self.EPSILON_MIN,
                self.EPSILON_INIT
                - episode * (self.EPSILON_INIT - self.EPSILON_MIN) / self.EPSILON_DECAY_EPISODES,
            )

            reward_hist.append(tot_reward)
            score_hist.append(snake.len())
            self._print_progress(episode, reward_hist, score_hist, time.monotonic() - start)

        print()
        print("-" * 97)
        self._save(q_net)

    def _new_snake(self):
        game_map = Map(_GRID_ROWS + 2, _GRID_COLS + 2)
        snake = Snake(game_map, _INIT_DIREC, list(_INIT_BODIES), list(_INIT_TYPES))
        game_map.create_rand_food()
        return snake, game_map

    def _make_state(self, snake, game_map):
        rows = game_map.num_rows - 2
        cols = game_map.num_cols - 2
        food_ch = torch.zeros(rows, cols)
        body_ch = torch.zeros(rows, cols)
        head_ch = torch.zeros(rows, cols)
        danger_ch = torch.zeros(rows, cols)

        snake_len = snake.len()
        tail = snake.tail()
        body_idx = {(p.x, p.y): i + 1 for i, p in enumerate(snake.bodies)}

        for i in range(1, game_map.num_rows - 1):
            for j in range(1, game_map.num_cols - 1):
                ri, ci = i - 1, j - 1
                pos = Pos(i, j)
                t = game_map.point(pos).type
                if t == PointType.FOOD:
                    food_ch[ri][ci] = 1.0
                elif t in _HEAD_TYPES:
                    head_ch[ri][ci] = 1.0
                    body_ch[ri][ci] = 1.0 / snake_len
                    danger_ch[ri][ci] = 1.0
                elif t in _BODY_TYPES:
                    idx = body_idx.get((i, j), 0)
                    body_ch[ri][ci] = idx / snake_len
                    if tail is None or (i != tail.x or j != tail.y):
                        danger_ch[ri][ci] = 1.0

        grid_channels = torch.stack((food_ch, body_ch, head_ch, danger_ch))

        direc_vec = torch.zeros(4)
        if snake.direc in _DIREC_IDX:
            direc_vec[_DIREC_IDX[snake.direc]] = 1.0

        food_direc = torch.zeros(4)
        food_dist = torch.zeros(1)
        if game_map.food is not None:
            head = snake.head()
            dx = game_map.food.x - head.x
            dy = game_map.food.y - head.y
            if dx < 0:
                food_direc[0] = 1.0
            if dy < 0:
                food_direc[1] = 1.0
            if dx > 0:
                food_direc[2] = 1.0
            if dy > 0:
                food_direc[3] = 1.0
            food_dist[0] = (abs(dx) + abs(dy)) / (2 * max(rows, cols))

        return torch.cat((grid_channels.flatten(), direc_vec, food_direc, food_dist))

    def _pick_action(self, q_vals, epsilon=None):
        if epsilon is not None and random.random() < epsilon:
            return random.randrange(_NUM_ACTIONS)
        return int(q_vals.argmax().item())

    def _learn(self, batch, q_net, q_target, optimizer):
        states, actions, rewards, next_states, dones = _Exp.batch(batch)
        with torch.no_grad():
            next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
            q_target_vals = q_target(next_states).gather(1, next_actions).squeeze(1)
        y = rewards + self.GAMMA * q_target_vals * (1 - dones)
        q_pred = q_net(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(q_pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), self.MAX_GRAD)
        optimizer.step()

    def _save(self, net):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        tmp = self.MODEL_PATH + f".tmp.{ts}"
        torch.save(net.state_dict(), tmp)
        print(f"Model saved to {tmp}")
        print(f"Rename to {self.MODEL_PATH} to use for inference.")

    def _load_net(self):
        if not os.path.isfile(self.MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")
        net = _Net().to(_DEVICE)
        net.load_state_dict(torch.load(self.MODEL_PATH, weights_only=True, map_location=_DEVICE))
        net.eval()
        return net

    def _print_progress(self, episode, reward_hist, score_hist, elapsed_secs):
        avg_r = float(np.mean(reward_hist[-self.NUM_EPISODES_FOR_AVG:]))
        avg_s = float(np.mean(score_hist[-self.NUM_EPISODES_FOR_AVG:]))
        elapsed_mins = int(elapsed_secs / 60)
        print(
            "\r"
            + " | ".join([
                f"Episode: {episode + 1:>6}",
                f"Avg reward/score: {avg_r:>7.2f} / {avg_s:>5.2f}",
                f"Elapsed: {elapsed_mins:>4} mins",
                f"Device: {_DEVICE_NAME}",
            ]),
            end="",
        )
        if (episode + 1) % self.NUM_EPISODES_FOR_AVG == 0:
            print()
