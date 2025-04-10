from tianshou.env import PettingZooEnv
from pettingzoo.classic import connect_four_v3


def get_env(render_mode=None):
    env = PettingZooC4MCTSWrapper(render_mode=render_mode)
    # env = PettingZooEnv(connect_four_v3.env(render_mode=render_mode))
    return env


class PettingZooC4MCTSWrapper(PettingZooEnv):
    """Custom class to wrap PettingZoo ConnectFour environment
    to allow for state/environment cloning necessary for MCTS"""

    def __init__(self, render_mode=None):
        super().__init__(connect_four_v3.env(render_mode=render_mode))
        self.render_mode = render_mode

    @classmethod
    def clone_env(cls, initial_env):
        new_env = cls()
        # must access raw unwrapped environment,
        # otherwise new `board` attribute is created instead
        unwrapped_env = getattr(new_env.env, "unwrapped")
        unwrapped_env.board = initial_env.env.board[:]
        unwrapped_env.agents = initial_env.env.possible_agents[:]
        # previous rewards should not matter for return backpropagation

        # self.env.rewards = {i: 0 for i in self.agents}
        # self.env._cumulative_rewards = {name: 0 for name in self.agents}

        unwrapped_env.terminations = initial_env.env.terminations.copy()
        unwrapped_env.truncations = initial_env.env.truncations.copy()
        unwrapped_env.infos = initial_env.env.infos.copy()
        unwrapped_env.agent_selection = initial_env.env.agent_selection
        return new_env
