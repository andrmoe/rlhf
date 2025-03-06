import time

import numpy as np
import torch
import random
from maze import small_maze
from gridworld import GridWorld
from agent import AgentEnvironment, Agent, Environment
from qlearning import QLearningAgent
import threading
import matplotlib.pyplot as plt
from matplotlib import animation


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


def q_values_in_matrix(world: GridWorld, agent: QLearningAgent[tuple[int, int], int]) -> torch.Tensor:
    matrix = torch.zeros_like(world.world_data)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            if (x, y) in agent.q_table:
                matrix[x, y] = float(torch.max(agent.q_table[(x, y)]))
    return matrix


def visualise_gridworld(world: GridWorld, agent: QLearningAgent[tuple[int, int], int]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    agent_dot, = ax1.plot([], [], 'ro', markersize=20)
    gridworld_mesh = ax1.pcolormesh(world.world_data)
    ax1.invert_yaxis()
    ax1.set_title("World")
    ax2.set_title("Expected Future Returns (Q)")
    matrix = q_values_in_matrix(world, agent)
    q_mesh = ax2.pcolormesh(matrix.numpy(), vmin=-10, vmax=10)
    ax2.invert_yaxis()

    def update(frame):
        agent_dot.set_data([world.state[1] + 0.5], [world.state[0] + 0.5])
        matrix = q_values_in_matrix(world, agent)
        q_mesh.set_array(matrix.numpy().flatten())
        return agent_dot, q_mesh
    ani = animation.FuncAnimation(fig, update, frames=range(2), interval=10)
    plt.colorbar(q_mesh, ax=ax2)
    plt.show()


def agent_loop(agent_environment: AgentEnvironment[tuple[int, int], int]):
    while True:
        if agent_environment.environment.is_terminal():
            while True:
                agent_environment.environment.state = random.randint(0, 9), random.randint(0, 9)
                if small_maze.world_data[agent_environment.environment.state] != small_maze.obstacle:
                    break

        agent_environment.step()
        time.sleep(0.01)


def reward_func(world: GridWorld, state: tuple[int, int]) -> float:
    if world.world_data[state] == world.terminal_square:
        return 10
    else:
        return -1


if __name__ == '__main__':
    agent = QLearningAgent(small_maze.state, range(4))
    agent_environment = AgentEnvironment(agent, small_maze, lambda s: reward_func(small_maze, s))
    t = threading.Thread(target=agent_loop, args=[agent_environment], daemon=True)
    t.start()
    visualise_gridworld(small_maze, agent)
