import argparse

import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ##### TRAINING PARAMS #####
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=5)  # TD(n), for n-step returns
    parser.add_argument(
        "--target-update-freq", type=int, default=32
    )  # TD target update interval
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument(
        "--step-per-epoch", type=int, default=100
    )  # transitions made each epoch
    parser.add_argument(
        "--step-per-collect", type=int, default=100
    )  # transitions gathered before doing updates in an epoch
    parser.add_argument("--update-per-step", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument(
        "--test-num", type=int, default=1
    )  # episodes per policy evaluation, or parallel test envs

    ##### MISCELLANEOUS ######
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.001)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.95,
        help="the expected winning rate",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=1,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="random",
        choices=["random", "minimax", "self"],
        help="the policy used by the opponent agent",
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=1,
        help="the search depth for minimax policy",
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]
