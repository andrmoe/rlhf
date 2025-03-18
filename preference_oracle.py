from typing import TypeVar, Generic, Callable, Any
from torch import Tensor
from rlhf_message import RLHFMessage


S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions


class PreferenceOracle(Generic[S]):
    def __init__(self):
        self.preference_callback = None
        self.next_pair_callback = None

    def register_callbacks(self, preference_callback: Callable[[RLHFMessage[S]], None],
                           next_pair_callback: Callable[[RLHFMessage[S]], RLHFMessage[S]]):
        self.preference_callback = preference_callback
        self.next_pair_callback = next_pair_callback

    def start(self):
        pass
