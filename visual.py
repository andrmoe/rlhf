from copy import deepcopy

import torch
from agent import AgentEnvironment
from gridworld import GridWorld
from qlearning import QLearningAgent
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Callable
import numpy as np


def agent_in_gridworld(world: GridWorld) -> np.ndarray:
    matrix = deepcopy(world.world_data.numpy())
    matrix[world.state] = 3
    return matrix


def q_values_in_matrix(world: GridWorld, agent: QLearningAgent[tuple[int, int], int]) -> np.ndarray:
    matrix = np.zeros(world.world_data.shape)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            if (x, y) in agent.q_table:
                matrix[x, y] = float(torch.max(agent.q_table[(x, y)]))
    return matrix


def rewards_in_matrix(agent_environment: AgentEnvironment[tuple[int, int], int]) -> np.ndarray:
    matrix = np.zeros(agent_environment.environment.world_data.shape)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            matrix[x, y] = agent_environment.reward_func((x, y))
    return matrix


def visualise_gridworld(matrix_funcs: [tuple[Callable[[], np.ndarray]]], titles: [str], ranges: [tuple[float, float]]):
    fig, axes = plt.subplots(1, len(matrix_funcs), figsize=(10, 5))
    meshes = [ax.pcolormesh(matrix_func(), vmin=ran[0], vmax=ran[1]) for ax, matrix_func, ran in zip(axes, matrix_funcs, ranges)]
    for ax, title in zip(axes, titles):
        ax.invert_yaxis()
        ax.set_title(title)

    def update(frame):
        return [mesh.set_array(matrix_func()) for mesh, matrix_func in zip(meshes, matrix_funcs)]
    ani = animation.FuncAnimation(fig, update, frames=range(2), interval=10)
    for mesh, ax in zip(meshes, axes):
        plt.colorbar(mesh, ax=ax)
    plt.show()