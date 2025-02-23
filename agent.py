from typing import Callable, TypeVar, Iterable, Generic
import random
import warnings

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions


class Agent(Generic[S, A]):
    def __init__(self, state: S, policy: Callable[[S], Iterable[tuple[float, A]]]):
        self.state = state
        self.policy = policy

    def act(self) -> A:
        action_distribution = self.policy(self.state)
        probabilities, actions = zip(*action_distribution)
        action = random.choices(actions, probabilities, k=1)[0]
        return action

    def observe(self, state: S):
        """
        The environment is fully observable. So every observation gives the entire environment state
        :param state: The state of the environment
        """
        self.state = state

class Environment(Generic[S, A]):
    def __init__(self, state: S, transition_func: Callable[[S, A], S]):
        self.state = state
        self.transition_func = transition_func

    def transition(self, action: A) -> S:
        self.state = self.transition_func(self.state, action)
        return self.state


class AgentEnvironment(Generic[S, A]):
    def __init__(self, agent: Agent[S, A], environment: Environment[S, A]):
        if agent.state != environment.state:
            warnings.warn(f"The state for the agent and environment are different. "
                          f"{agent.state=}, {environment.state=}, "
                          f"The agent's state was changed to the environment state", stacklevel=2)
            agent.state = environment.state
        self.agent = agent
        self.environment = environment

    def step(self) -> S:
        action = self.agent.act()
        new_state = self.environment.transition(action)
        self.agent.observe(new_state)
        return new_state
