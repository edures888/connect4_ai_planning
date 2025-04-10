import logging
import os
from typing import Optional, Tuple

import numpy as np
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import (
    BasePolicy,
)
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

from src.mcts_collector import MCTSCollector

from .config_parser import get_args
from .environment import get_env
from .agents import get_agents
from .eval import watch
import datetime

DEBUG = os.environ.get("DEBUG")
level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:

    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy_manager, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
    )

    # ======== collector setup =========
    train_collector = MCTSCollector(
        policy_manager,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        # exploration_noise=True,
    )
    test_collector = MCTSCollector(policy_manager, test_envs)
    # exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(
        args.logdir,
        "connect4",
        "dqn",
        datetime.datetime.now().strftime("%H_%M_%S-%d%m"),
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    tb_logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy_manager):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir,
                "connect4",
                "dqn",
                f"{datetime.datetime.now().strftime('%H_%M_%S-%d%m')}_policy.pth",
            )
        torch.save(
            policy_manager.policies[agents[args.agent_id - 1]].state_dict(),
            model_save_path,
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy_manager.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy_manager.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    if args.watch:  # skip training, can refactor this elsewhere
        return None, policy_manager.policies[agents[args.agent_id - 1]]

    # trainer
    result = OffpolicyTrainer(
        policy_manager,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=tb_logger,
        # test_in_train=False,
        reward_metric=reward_metric,
        verbose=True,
        show_progress=True,
        test_in_train=True,
    ).run()

    return result, policy_manager.policies[agents[args.agent_id - 1]]


if __name__ == "__main__":
    args = get_args()
    _, agent = train_agent(args)
    # if args.watch:
    # watch(args, agent)
