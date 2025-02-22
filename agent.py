from typing import Callable, TypeVar

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions

class Agent:
    def __init__(self, state: S, transition: Callable[[S, A], S], policy: Callable[[S], A]):
        self.state = state
        self.transition = transition
        self.policy = policy

    def act(self) -> tuple[A, S]:
        action = self.policy(self.state)
        self.state = self.transition(self.state, action)
        return action, self.state