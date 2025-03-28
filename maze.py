import torch
from torch import Tensor
from gridworld import GridWorld

small_maze = GridWorld(-Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 0, 1, 0, 1, 0, 0, 0, -1, 1],
                               [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                               [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                               [1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                               [1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                               [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                               [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                               [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                               (5, 5))

open_world = GridWorld(torch.zeros((9, 9)), (5, 5))