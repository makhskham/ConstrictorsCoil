import argparse

from constrictorscoil.solver.dqn_torch import DQNTorchSolver


def main():
    parser = argparse.ArgumentParser(description="Train the PyTorch DQN solver.")
    parser.add_argument(
        "--episodes",
        metavar="<n>",
        default=DQNTorchSolver.NUM_EPISODES,
        type=int,
        help=f"number of training episodes (default: {DQNTorchSolver.NUM_EPISODES})",
        dest="episodes",
    )
    parser.add_argument(
        "--out",
        metavar="<path>",
        default=DQNTorchSolver.MODEL_PATH,
        type=str,
        help=f"output model path (default: {DQNTorchSolver.MODEL_PATH})",
        dest="out",
    )
    args = parser.parse_args()

    solver = DQNTorchSolver(None)
    solver.NUM_EPISODES = args.episodes
    solver.MODEL_PATH = args.out
    solver.train_standalone()


if __name__ == "__main__":
    main()
