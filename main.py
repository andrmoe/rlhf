
import numpy as np
import torch
import random
from maze import small_maze, open_world
from gridworld import GridWorld
from agent import AgentEnvironment, Agent, Environment
from qlearning import QLearningAgent
from visual import visualise_gridworld
from reward_model import GridWorldRewardModel
import threading
from rlhf import rlhf


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
    visualise_gridworld(agent_environment)


def rlhf_demo():
    world = small_maze
    agent = QLearningAgent(small_maze.state, range(4))
    reward_model = GridWorldRewardModel(world.world_data.shape, 10)
    rlhf(agent, world, reward_model, lambda t1, t2: torch.tensor([0.5, 0.5]))


if __name__ == '__main__':
    #q_learning_demo()
    rlhf_demo()
