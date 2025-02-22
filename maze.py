from torch import Tensor
from gridworld import GridWorld, visualise_gridworld

small_maze = GridWorld(-Tensor([[1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                               [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                               [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
                               [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                               [1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                               [1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                               [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                               [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                               [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                               (5, 5))


visualise_gridworld(small_maze)