from typing import Union
import numpy as np
import math
import torch
import torch.nn.functional as F
import gymnasium as gym
from tianshou.policy import BasePolicy

from src.environment import PettingZooC4MCTSWrapper


class MCTSNode:
    def __init__(self, env, is_terminal, prior=1.0, parent=None, action=None):
        self.prior = prior
        self.parent = parent
        self.action = action
        self.last_player = None
        self.env = env

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_terminal = is_terminal

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        """
        Choose child node with highest PUCT score
        c_puct: confidence coefficient for exploration
        """
        best_score = -float("inf")
        best_action = -1
        best_child = None
        sqrt_parent_visits = math.sqrt(max(1, self.visit_count))

        # find best action and child node
        for action, child in self.children.items():
            # Predictor + UCT -- Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            ucb_score = child.value() + c_puct * child.prior * sqrt_parent_visits / (
                1 + child.visit_count
            )

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, actions, priors):
        """
        Adds new child nodes using action space
        priors: P(s,a)
        """
        for action, prior in zip(actions, priors):
            if action not in self.children:
                env_copy = PettingZooC4MCTSWrapper.clone_env(self.env)
                _, _, _, done, _ = env_copy.step(action)
                self.children[action] = MCTSNode(
                    env_copy, done, prior=prior, parent=self, action=action
                )

    def update(self, value):
        """
        update node values during backpropagation
        value: rollout value
        """
        self.visit_count += 1
        self.value_sum += value

    def get_visit_count_distribution(self, temperature=1.0):
        """
        Returns distribution P(a|s) using visit counts
        temperature: exploration parameter,
          set to 0 for deterministically choosing action with highest visit count
        """
        visits = {action: child.visit_count for action, child in self.children.items()}

        # set highest visit count action to have probability 1.0
        if temperature == 0:
            action = max(visits.items(), key=lambda x: x[1])[0]
            dist = {a: 0.0 for a in visits}
            dist[action] = 1.0
            return dist

        # compute temperature skewed distribution
        counts = np.array([visits[a] for a in sorted(visits.keys())])
        counts = counts ** (1.0 / temperature)
        total = counts.sum()
        if total == 0:
            probs = np.ones_like(counts) / len(counts)
        else:
            probs = counts / total

        return {a: probs[i] for i, a in enumerate(sorted(visits.keys()))}


class MCTS:
    def __init__(
        self,
        model: BasePolicy,
        env: PettingZooC4MCTSWrapper,
        num_simulations=100,
        c_puct=1.0,
        temperature=1.0,
        max_depth=10,
        rollout_depth=10,
    ):
        """
        model: Policy
        num_simulations: Number of rollouts
        c_puct: Confidence coefficient for exploration in PUCT
        temperature:
        """
        self.model = model
        self.env = env
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.root = MCTSNode(env, is_terminal=False)

    def choose_action(self):
        for _ in range(self.num_simulations):
            self.search()

        if not self.root.children:
            raise Exception("No children")
        action_prob = self.root.get_visit_count_distribution(self.temperature)
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        action = np.random.choice(actions, p=probs)
        return action

    def search(self):
        depth = 0
        node = self.root
        path = []

        # -------- Selection & Expansion Phase --------
        while depth < self.max_depth and not node.is_terminal and node.expanded():
            path.append(node)
            action, child = node.select_child()
            node = child
            depth += 1

        # -------- Expansion Phase --------
        if not node.is_terminal:
            obs, _, _, _, _ = node.env.last()
            with torch.no_grad():
                q_values = self.model(obs)  # Shape: [batch_size, num_actions]
                policy = F.softmax(q_values, dim=-1).detach().cpu().numpy()[0]
            legal_actions = [i for i, _ in enumerate(obs["action_mask"])]
            legal_policy = [policy[a] for a in legal_actions]
            legal_policy_sum = sum(legal_policy)
            if legal_policy_sum > 0:
                legal_policy = [p / legal_policy_sum for p in legal_policy]
            else:
                legal_policy = [1.0 / len(legal_actions) for _ in legal_actions]
            node.expand(legal_actions, legal_policy)
            _, node = node.select_child(self.c_puct)

        # -------- Backpropagation Phase --------
        rollout_returns = self.rollout(node.env)

        # -------- Backpropagation Phase --------
        for path_node in reversed(path):
            path_node.update(rollout_returns)
            # negate action value for opposing player
            value = -value

        return

    def rollout(self, env):
        """
        Run a rollout (simulation) until a terminal state is reached or rollout_depth is reached.
        Returns cumulative reward.
        """
        cumulative_reward = 0.0
        current_depth = 0
        done = False
        obs, _, _, _, _ = env.last()
        # Continue simulation with random actions
        while (not done) and (current_depth < self.rollout_depth):
            actions = [i for i, _ in enumerate(obs["action_mask"])]
            action = np.random.choice(actions)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            current_depth += 1
        return cumulative_reward

    def update_root(self, action):
        """
        Updates root of tree to children for search tree reuse
        Subtree transfer is appropriate given deterministic environment
        action: Action taken by model during actual transition
        """
        self.root = self.root.children[action]
