from typing import TypeVar, Generic
import torch

S = TypeVar('S')


class RLHFMessage(Generic[S]):
    def __init__(self):
        self.reward_model_weights: dict[str, torch.nn.Parameter] = {}
        self.trajectories: [[S]] = []
        self.preference = torch.zeros(2)
        self.save = False

    def __str__(self):
        return f'{self.reward_model_weights=}, {self.trajectories=}, {self.preference=}, {self.save=}'