import time
from typing import TypeVar, Generic, Callable
import numpy as np
import torch
from torch import tensor, Tensor, nn
import multiprocessing

S = TypeVar('S')


class RewardModel(Generic[S], nn.Module):
    def __init__(self, ensemble: [nn.Module]):
        super().__init__()
        self.ensemble = nn.ModuleList(ensemble)

    def encode(self, state: S) -> Tensor:
        pass

    def encode_trajectory(self, trajectory: [S]) -> Tensor:
        return torch.stack([self.encode(state) for state in trajectory])

    def reward(self, state: S) -> Tensor:
        encoded = self.encode(state)
        rewards = torch.stack([reward_model(encoded) for reward_model in self.ensemble])
        return torch.mean(rewards)

    def trajectory_variance(self, trajectory: [S]) -> float:
        trajectory_rewards = [torch.tensor([reward_model(self.encode(state)) for reward_model in self.ensemble])
                              for state in trajectory]
        variances = torch.tensor([torch.var(episode_rewards) for episode_rewards in trajectory_rewards])
        return float(torch.sum(variances))

    def exp_sum(self, trajectory: Tensor) -> Tensor:
        rewards = tensor([[predictor(state) for state in trajectory] for predictor in self.ensemble])
        return torch.exp(torch.sum(rewards, dim=1))


    def loss(self, traj0: Tensor, traj1: Tensor, preference: Tensor) -> Tensor:
        exp_sum0 = self.exp_sum(traj0)
        exp_sum1 = self.exp_sum(traj1)
        denominator = exp_sum0 + exp_sum1
        return - torch.mean(preference[0]*torch.log(exp_sum0/denominator) + preference[1]*np.log(exp_sum1/denominator))

class RewardTable(nn.Module):
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.table = nn.Parameter(torch.randn(shape))

    def forward(self, indices: Tensor) -> Tensor:
        return self.table[indices[0], indices[1]]


def train(preference_pipe: multiprocessing.Pipe,
          model_weights_pipe: multiprocessing.Pipe):
    preference_database = []
    # TODO: Implement training
    preference_pipe[0].close()
    model_weights_pipe[1].close()
    steps = 0
    while True:
        if False:
            print('Sending updated model weights')
            model_weights_pipe[0].send({})
        if preference_pipe[1].poll(0):
            new_preference_data = preference_pipe[1].recv()
            print(f"Reward model received new training data ({new_preference_data})")
        time.sleep(0.01)
        steps += 1

class GridWorldRewardModel(RewardModel[tuple[int, int]]):
    def __init__(self, shape: tuple[int, int], ensemble_size: int):
        super().__init__([RewardTable(shape) for _ in range(ensemble_size)])

    def encode(self, s: S) -> Tensor:
        return tensor(s, dtype=torch.long)
