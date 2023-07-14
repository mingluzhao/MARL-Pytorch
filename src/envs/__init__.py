from functools import partial
#from .multiagentenv import MultiAgentEnv
#from .starcraft2.starcraft2 import StarCraft2Env
from .mape.multiagent.environment import MultiAgentEnv
from .mape.make_env import make_env
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
#REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["mape"] = partial(env_fn, env=make_env)

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
