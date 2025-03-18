from typing import TypeVar, Generic
import torch

S = TypeVar('S')


class RLHFMessage(Generic[S]):
    def __init__(self):
        self.reward_model_weights: dict[str, torch.nn.Parameter] = {}
        self.trajectory0: [S] = []
        self.trajectory1: [S] = []
        self.preference = torch.zeros(2)
        self.save = False

    def __str__(self):
        return (f'{self.reward_model_weights=}, {self.trajectory0=}, '
                f'{self.trajectory1=}, {self.preference=}, {self.save=}')