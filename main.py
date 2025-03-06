import time

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


def agent_loop(agent_environment: AgentEnvironment[tuple[int, int], int]):
    elapsed_steps = 0
    while True:
        if agent_environment.environment.is_terminal() or elapsed_steps > 200:
            elapsed_steps = 0
            while True:
                agent_environment.environment.state = random.randint(0, 9), random.randint(0, 9)
                if small_maze.world_data[agent_environment.environment.state] != small_maze.obstacle:
                    break

        agent_environment.step()
        elapsed_steps += 1
        #time.sleep(0.01)


def reward_func(world: GridWorld, state: tuple[int, int]) -> float:
    if world.world_data[state] == world.terminal_square:
        return 10
    else:
        return -1


if __name__ == '__main__':
    world = open_world
    agent = QLearningAgent(small_maze.state, range(4))
    reward_model = GridWorldRewardModel(world.world_data.shape, 10)
    agent_environment = AgentEnvironment(agent, world, lambda s: float(reward_model.reward(s)[0]))
    t = threading.Thread(target=agent_loop, args=[agent_environment], daemon=True)
    t.start()
    visualise_gridworld(agent_environment)
