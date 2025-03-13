import time

import numpy as np
import torch
import random
from maze import small_maze, open_world
from gridworld import GridWorld
from agent import AgentEnvironment, Agent, Environment
from qlearning import QLearningAgent
import visual
from reward_model import GridWorldRewardModel
import threading
from rlhf import rlhf
from gui import Gui

def manual_control(state: tuple[int, int]) -> tuple[tuple[float, int]]:
    east = 0
    north = 1
    west = 2
    south = 3

    while True:
        wasd = input("WASD:")
        if wasd == 'd':
            return ((1.0, east),)
        elif wasd == 'w':
            return ((1.0, north),)
        elif wasd == 'a':
            return ((1.0, west),)
        elif wasd == 's':
            return ((1.0, south),)


def reward_func(world: GridWorld, state: tuple[int, int]) -> float:
    if world.world_data[state] == world.terminal_square:
        return 10
    else:
        return -1


def q_learning_demo():
    world = small_maze
    agent = QLearningAgent(small_maze.state, range(4))
    agent_environment = AgentEnvironment(agent, world, lambda s: reward_func(world, s), sleep_time=0.01)
    t = threading.Thread(target=agent_environment.loop, args=[], daemon=True)
    t.start()
    visual.visualise_gridworld([lambda: visual.agent_in_gridworld(world),
                                lambda: visual.q_values_in_matrix(world, agent),
                                lambda: visual.rewards_in_matrix(agent_environment)],
                               ["World", "Expected Future Returns (Q)", "Reward Function"],
                               [(-1, 3), (-3, 10), (-1, 10)])


def euclidean_preference_oracle(t1: [tuple[int, int]], t2: [tuple[int, int]]) -> torch.Tensor:
    target = (0, 0)
    sq_dist1 = sum([(x-target[0])**2 + (y-target[1])**2 for x, y in t1])
    sq_dist2 = sum([(x-target[0])**2 + (y-target[1])**2 for x, y in t2])
    if sq_dist1 == sq_dist2:
        return torch.tensor([0.5,0.5])
    elif sq_dist1 > sq_dist2:
        return torch.tensor([1,0])
    elif sq_dist1 < sq_dist2:
        return torch.tensor([0,1])

def rlhf_demo():
    world = small_maze
    agent = QLearningAgent(small_maze.state, range(4), exploration_rate=0.3)
    reward_model = GridWorldRewardModel(world.world_data.shape, 10)
    trajectory_length = 10
    gui = Gui(world, trajectory_length)
    rlhf(agent, world, reward_model, gui, trajectory_length)


if __name__ == '__main__':
    #q_learning_demo()
    rlhf_demo()
