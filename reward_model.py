import time
from typing import TypeVar, Generic
import numpy as np
import torch
from torch import tensor, Tensor, nn
from torch.utils.data import DataLoader, Dataset
import multiprocessing
from rlhf_message import RLHFMessage

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
        return self(encoded)

    def forward(self, encoded_traj) -> Tensor:
        rewards = torch.stack([reward_model(encoded_traj) for reward_model in self.ensemble])
        return torch.mean(rewards)

    def trajectory_variance(self, trajectory: [S]) -> float:
        trajectory_rewards = [torch.tensor([reward_model(self.encode(state)) for reward_model in self.ensemble])
                              for state in trajectory]
        variances = torch.tensor([torch.var(episode_rewards) for episode_rewards in trajectory_rewards])
        return float(torch.sum(variances))

    def exp_sum(self, trajectory: Tensor, predictor: nn.Module) -> Tensor:
        reward_sum = 0
        for state in trajectory:
            reward_sum = reward_sum + predictor(state)

        return torch.exp(reward_sum)


    def loss(self, traj0: Tensor, traj1: Tensor, preference: Tensor) -> Tensor:
        loss = 0
        for predictor in self.ensemble:
            exp_sum0 = self.exp_sum(traj0[0], predictor)
            exp_sum1 = self.exp_sum(traj1[0], predictor)
            denominator = exp_sum0 + exp_sum1
            loss = loss - torch.mean(preference[0][0]*torch.log(exp_sum0/denominator) + preference[0][1]*torch.log(exp_sum1/denominator))
        return loss

class RewardTable(nn.Module):
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.table = nn.Parameter(torch.randn(shape))

    def forward(self, indices: Tensor) -> Tensor:
        entry = self.table[indices[0], indices[1]]
        return entry


class PreferenceDataset(Dataset):
    def __init__(self):
        self.data_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        return self.data_list[idx]


def train(model: RewardModel[S], preference_pipe: multiprocessing.Pipe, model_weights_pipe: multiprocessing.Pipe):
    preference_pipe[0].close()
    model_weights_pipe[1].close()

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = PreferenceDataset()
    # Wait for the first feedback
    preference_pipe[1].poll()
    rlhf_message: RLHFMessage[S] = preference_pipe[1].recv()
    t0 = rlhf_message.trajectory0
    t1 = rlhf_message.trajectory1
    pref = rlhf_message.preference

    print(f"Reward model received new training data ({t0, t1, pref})")
    dataset.data_list.append((model.encode(t0), model.encode(t1), pref))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.train()

    rlhf_message = RLHFMessage[S]()

    while True:
        for batch_index, (traj0, traj1, preference) in enumerate(data_loader):
            optimiser.zero_grad()
            loss = model.loss(traj0, traj1, preference)
            #print(loss)
            loss.backward()
            optimiser.step()
        print('Sending updated model weights')
        rlhf_message.reward_model_weights = dict(model.named_parameters())
        model_weights_pipe[0].send(rlhf_message)
        if preference_pipe[1].poll(0):
            rlhf_message: RLHFMessage[S] = preference_pipe[1].recv()
            print(f"Reward model received new training data ({rlhf_message.trajectory0, rlhf_message.trajectory1, rlhf_message.preference})")
            dataset.data_list.append((model.encode(rlhf_message.trajectory0), model.encode(rlhf_message.trajectory1), rlhf_message.preference))
            data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        print(len(dataset))

class GridWorldRewardModel(RewardModel[tuple[int, int]]):
    def __init__(self, shape: tuple[int, int], ensemble_size: int):
        super().__init__([RewardTable(shape) for _ in range(ensemble_size)])

    def encode(self, s: S) -> Tensor:
        return tensor(s, dtype=torch.long)
