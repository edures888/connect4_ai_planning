from tianshou.env import PettingZooEnv
from pettingzoo.classic import connect_four_v3


def get_env(render_mode=None):
    env = PettingZooEnv(connect_four_v3.env(render_mode=render_mode))
    return env
