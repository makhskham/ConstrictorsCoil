import argparse

from constrictorscoil.base.direc import Direc
from constrictorscoil.base.point import PointType
from constrictorscoil.base.pos import Pos
from constrictorscoil.game import Game, GameConf, GameMode


def main():
    dict_solver = {
        "greedy": "GreedySolver",
        "hamilton": "HamiltonSolver",
        "dqn": "DQNSolver",
        "dqn_torch": "DQNTorchSolver",
        "rl": "RLSolver",
    }

    dict_mode = {
        "normal": GameMode.NORMAL,
        "bcmk": GameMode.BENCHMARK,
        "train_dqn": GameMode.TRAIN_DQN,
        "train_dqn_gui": GameMode.TRAIN_DQN_GUI,
    }

    parser = argparse.ArgumentParser(description="Run snake game agent.")
    parser.add_argument(
        "-s",
        default="hamilton",
        choices=dict_solver.keys(),
        help="name of the solver to direct the snake (default: hamilton)",
    )
    parser.add_argument(
        "-m",
        default="normal",
        choices=dict_mode.keys(),
        help="game mode (default: normal)",
    )
    args = parser.parse_args()

    conf = GameConf()
    conf.solver_name = dict_solver[args.s]
    conf.mode = dict_mode[args.m]

    if args.s == "rl":
        conf.map_rows = 6
        conf.map_cols = 6
        conf.init_bodies = [Pos(1, 3), Pos(1, 2), Pos(1, 1)]
        conf.init_types = [PointType.HEAD_R, PointType.BODY_HOR, PointType.BODY_HOR]
        conf.info_str = (
            "<w/a/s/d>: snake direction\n"
            "<space>: pause/resume\n"
            "<r>: restart    <esc>: exit\n"
            "-----------------------------------\n"
            "status: %s\n"
            "episode: %d   step: %d\n"
            "length: %d/%d (6x6)\n"
            "-----------------------------------"
        )

    print(f"Solver: {conf.solver_name}   Mode: {conf.mode}")

    Game(conf).run()


if __name__ == "__main__":
    main()
