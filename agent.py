import time
from typing import Callable, TypeVar, Iterable, Generic
import random
import warnings

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions


class Agent(Generic[S, A]):
    def __init__(self, state: S):
        self.state = state
        self.most_recent_action = None

    def policy(self, state: S) -> Iterable[tuple[float, A]]:
        pass

    def act(self) -> A:
        action_distribution = self.policy(self.state)
        probabilities, actions = zip(*action_distribution)
        action = random.choices(actions, probabilities, k=1)[0]
        self.most_recent_action = action
        return action

    def observe(self, new_state: S, reward: float):
        """
        The environment is fully observable. So every observation gives the entire environment state
        :param new_state: The state of the environment
        :param reward: RL reward given to the agent at this time step
        """
        self.state = new_state

class Environment(Generic[S, A]):
    def __init__(self, state: S):
        self.state = state

    def transition_func(self, state: S, action: A) -> S:
        pass

    def transition(self, action: A) -> S:
        self.state = self.transition_func(self.state, action)
        return self.state

    def is_terminal(self) -> bool:
        return False


class AgentEnvironment(Generic[S, A]):
    def __init__(self, agent: Agent[S, A], environment: Environment[S, A],
                 reward_func: Callable[[S], float] = lambda s: 0, step_limit: int = 200, sleep_time=None):
        if agent.state != environment.state:
            warnings.warn(f"The state for the agent and environment are different. "
                          f"{agent.state=}, {environment.state=}, "
                          f"The agent's state was changed to the environment state", stacklevel=2)
            agent.state = environment.state
        self.agent = agent
        self.environment = environment
        self.reward_func= reward_func
        self.step_limit = step_limit
        self.sleep_time = sleep_time

    def step(self) -> S:
        action = self.agent.act()
        new_state = self.environment.transition(action)
        self.agent.observe(new_state, self.reward_func(new_state))
        if self.environment.is_terminal():
            return None
        return new_state

    def loop(self):
        initial_state = self.environment.state
        while True:
            self.environment.state = self.agent.state =  initial_state
            elapsed_steps = 0
            while True:
                if self.environment.is_terminal() or elapsed_steps > self.step_limit:
                    break
                self.step()
                elapsed_steps += 1
                if self.sleep_time:
                    time.sleep(self.sleep_time)

