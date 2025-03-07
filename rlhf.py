from agent import Agent, Environment, AgentEnvironment
from reward_model import RewardModel
from typing import Callable, TypeVar, Iterable
from torch import Tensor
import multiprocessing

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions


def rlhf(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
         preference_oracle: Callable[[Iterable[S], Iterable[S]], Tensor]):
    agent_environment = AgentEnvironment(agent, environment, lambda s: float(reward_model.reward(s)[0]))

    preference_database = []
    rl_process = multiprocessing.Process()
