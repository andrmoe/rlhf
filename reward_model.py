from typing import TypeVar, Generic, Callable
import numpy as np
import torch
from torch import tensor, Tensor, nn

S = TypeVar('S')


class RewardModel(Generic[S], nn.Module):
    def __init__(self, ensemble: [nn.Module], encode: Callable[[S], Tensor]):
        super().__init__()
        self.ensemble = nn.ModuleList(ensemble)
        self.encode = encode

    def reward(self, state: S) -> Tensor:
        encoded = self.encode(state)
        rewards = torch.stack([reward_model(encoded) for reward_model in self.ensemble])
        return torch.stack((torch.mean(rewards), torch.var(rewards)))

    def loss(self, traj0: [S], traj1: [S], preference: Tensor) -> float:
        exp_sum0 = np.exp(sum([self.reward(state) for state in traj0]))
        exp_sum1 = np.exp(sum([self.reward(state) for state in traj1]))
        denominator = exp_sum0 + exp_sum1
        return - preference[0]*np.log(exp_sum0/denominator) - preference[1]*np.log(exp_sum1/denominator)


class RewardTable(nn.Module):
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.table = nn.Parameter(torch.randn(shape))

    def forward(self, indices: Tensor) -> Tensor:
        return self.table[indices[0], indices[1]]


class GridWorldRewardModel(RewardModel[tuple[int, int]]):
    def __init__(self, shape: tuple[int, int], ensemble_size: int):
        super().__init__([RewardTable(shape) for _ in range(ensemble_size)], lambda s: tensor(s, dtype=torch.long))
