# Constrictor's Coil

![Constrictor's Coil](assets/ConstrictorsCoil.png)

[![Project Presentation](https://img.shields.io/badge/Website-Behind_the_Scenes-blue)](https://aisnakegame.my.canva.site/)

A Snake game AI built in Python, implementing three distinct solvers ranging from classical graph algorithms to deep reinforcement learning. The Hamilton solver achieves near-perfect play, filling 99.9% of the map on average.

> Not because it's a snake. Because every successful action literally constricts future possibilities. The snake isn't just growing. It is tightening the noose around itself.

---

## About

The project focuses on the artificial intelligence of the [Snake][snake-wiki] game. The snake's goal is to eat food continuously and fill the map with its body as quickly as possible. This version is written in Python for a user-friendly GUI and cleaner algorithm implementations.

The game runs on an 8x8 grid (64 cells). The snake starts at the top-left corner facing right, with an initial length of 4. It can eat at most 60 pieces of food before filling the entire map.

---

## Solvers

Three solvers are implemented, each making different tradeoffs between completeness and computational cost.

| Solver | Strategy | Average Length | Average Steps |
| :----: | :------: | :------------: | :-----------: |
| [Hamilton][doc-hamilton] | Hamiltonian cycle with shortcuts | 63.93 / 64 | 717.83 |
| [Greedy][doc-greedy] | Shortest path with safety lookahead | 60.15 / 64 | 904.56 |
| [DQN][doc-dqn] (experimental) | Deep Q-Network | 24.44 / 64 | 131.69 |

Results averaged over 1000 episodes each.

---

## Algorithm Details

### Hamilton Solver

![][demo-hamilton]

Builds a Hamiltonian cycle across all 64 cells, then directs the snake along it. Because following the full cycle is inefficient, the solver takes shortcuts when it can safely skip ahead in the path without getting cut off.

The cycle is constructed once at the start by fixing the snake's initial head and body positions, then computing the longest path from head to tail, and joining the endpoints into a cycle. Shortcut eligibility is checked at each step using the path index difference between the snake's head and the food.

This guarantees the snake never collides with itself and eventually fills the entire board. The average length of 63.93 reflects the rare cases where shortcuts slightly perturb the finishing position.

### Greedy Solver

![][demo-greedy]

At each step, the solver attempts to eat food along the shortest path. Before committing, it simulates the move and checks whether the snake's head can still reach its own tail afterward (a proxy for long-term survival). If the safety check fails, the snake wanders by following the longest path to its tail instead.

Decision sequence per step:
1. Find shortest path P1 from head to food. If none, go to step 4.
2. Simulate snake S2 eating food along P1.
3. Find longest path from S2's head to its tail. If it exists, take direction P1[0]. Otherwise, go to step 4.
4. Find longest path P3 from S1's head to its tail. If it exists, take direction P3[0]. Otherwise, go to step 5.
5. Move in the direction that maximizes distance from food.

### DQN Solver

![][demo-dqn]

Uses a Deep Q-Network trained with three optimizations applied simultaneously: Double DQN, Prioritized Experience Replay, and Dueling Network Architecture. The agent is trained in relative direction mode (the snake's frame of reference, not the map's), which leverages the game's rotational symmetry to improve sample efficiency.

The state vector has two components. The global state is an 8x8x4 binary encoding of each cell's content (head, body, food, or empty). The local state is a 3-value binary vector indicating whether the cell directly ahead, left, or right of the head is dangerous.

The DQN solver is experimental. It plateaus well below the classical solvers because the credit assignment problem is hard: the reward for each food piece is separated from the eventual collision by hundreds of steps.

---

## Benchmarks

Tested with 1000 episodes per solver. Two metrics:

- **Average Length:** how far the snake grew before dying or finishing (max: 64)
- **Average Steps:** total moves taken

| Solver | Average Length | Average Steps |
| :----: | :------------: | :-----------: |
| Hamilton | 63.93 | 717.83 |
| Greedy | 60.15 | 904.56 |
| DQN | 24.44 | 131.69 |

The Hamilton solver is 6% longer on average than Greedy, but uses 21% fewer steps, because shortcuts aggressively reduce wasted movement. The DQN solver underperforms both classical approaches by a wide margin under default training conditions.

---

## Installation

Requirements: Python 3.6+ with [Tkinter][doc-tkinter] installed.

```bash
pip install -r requirements.txt
python run.py [-h]
```

Run a specific solver:

```bash
# Hamilton (default)
python run.py -s hamilton

# Greedy
python run.py -s greedy

# DQN (TensorFlow, requires trained model in logs/)
python run.py -s dqn

# DQN PyTorch (requires model.pt in project root)
python run.py -s dqn_torch

# Benchmark mode (1000 episodes, no GUI)
python run.py -s hamilton -m bcmk
```

Train the PyTorch DQN solver:

```bash
python train.py
# optional flags:
python train.py --episodes 100000 --out model.pt
```

After training completes, rename the output file to `model.pt` in the project root.

Run unit tests:

```bash
python -m pytest
```

---

## Project Structure

```
ConstrictorsCoil/
├── run.py               # entrypoint, CLI argument parsing
├── train.py             # standalone PyTorch DQN training script
├── requirements.txt
├── constrictorscoil/
│   ├── game.py          # game loop, config, modes
│   ├── gui.py           # Tkinter rendering
│   ├── base/
│   │   ├── snake.py     # snake state and movement
│   │   ├── map.py       # game grid, cell types
│   │   ├── point.py     # coordinate primitive
│   │   ├── pos.py       # position utilities
│   │   └── direc.py     # direction enum
│   └── solver/
│       ├── base.py      # BaseSolver interface
│       ├── path.py      # BFS shortest path, heuristic longest path
│       ├── greedy.py    # Greedy solver
│       ├── hamilton.py  # Hamilton solver
│       ├── dqn/         # DQN solver (TensorFlow)
│       └── dqn_torch/   # DQN solver (PyTorch, Double DQN)
├── docs/
│   ├── algorithms.md    # algorithm writeups with diagrams
│   └── images/          # demo GIFs, architecture diagrams
├── tests/
└── tools/
```

---

## Design Decisions

**Why Hamiltonian cycle for the strongest solver?** The snake game on a finite grid is a path-covering problem. A Hamiltonian cycle visits every cell exactly once, making it impossible to trap yourself. The only cost is inefficiency: without shortcuts, the snake spirals through the whole board even when the food is one step away. The shortcut logic recovers most of that efficiency while preserving safety.

**Why relative direction for DQN?** The game map is symmetric under rotation. Training in the snake's own reference frame means the network sees equivalent situations as equivalent states rather than four different encodings depending on which way the snake is facing. Experimental results confirm this roughly doubles training efficiency.

**Why BFS for shortest path and a heuristic for longest?** Shortest path on an unweighted grid is exactly what BFS solves optimally. Longest path on a general graph is NP-hard. The heuristic used here extends the shortest path by iteratively pushing path segments outward when space allows, producing good-enough results for the 8x8 grid without exponential cost.

---

## License

See the [LICENSE](./LICENSE) file for license rights and limitations.

Built by Makhsuma Khamzaliyeva.

---

[snake-wiki]: https://en.wikipedia.org/wiki/Snake_(video_game)

[doc-tkinter]: https://docs.python.org/3/library/tkinter.html
[doc-algorithms]: ./docs/algorithms.md
[doc-greedy]: ./docs/algorithms.md#greedy-solver
[doc-hamilton]: ./docs/algorithms.md#hamilton-solver
[doc-dqn]: ./docs/algorithms.md#dqn-solver

[demo-hamilton]: ./docs/images/solver_hamilton.gif
[demo-greedy]: ./docs/images/solver_greedy.gif
[demo-dqn]: ./docs/images/solver_dqn.gif
