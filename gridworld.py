import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import animation
from copy import deepcopy


class GridWorld:
    def __init__(self, world_data: Tensor, agent_pos: tuple[int, int]):
        self.world_data = world_data
        self.agent_pos = agent_pos
        self.east = 0
        self.north = 1
        self.west = 2
        self.south = 3
        self.obstacle = -1


    def move(self, direction: int) -> bool:
        f"""
        Move the agent in a particular direction. Agent remains stationary if the movement is impossible.
        :param direction An int between 0 and 3. east={self.east}, north={self.north}, west={self.west}, south={self.south}.
        :return True if movement happened, False if blocked.
        """
        if direction == self.east:
            new_position = self.agent_pos[0], self.agent_pos[1] + 1
        elif direction == self.north:
            new_position = self.agent_pos[0] - 1, self.agent_pos[1]
        elif direction == self.west:
            new_position = self.agent_pos[0], self.agent_pos[1] - 1
        elif direction == self.south:
            new_position = self.agent_pos[0] + 1, self.agent_pos[1]
        else:
            raise ValueError(
                f"Direction must be between 0 and 3, east={self.east}, north={self.north}, west={self.west}, south={self.south}")

        # The agent doesn't move if the direction is into the edge of the world
        if (direction == self.east and self.agent_pos[1] == self.world_data.shape[1] - 1) or (
                direction == self.north and self.agent_pos[0] == 0) or (
                direction == self.west and self.agent_pos[1] == 0) or (
                direction == self.south and self.agent_pos[0] == self.world_data.shape[0] - 1):
            return False

        # The agent doesn't move if the direction is into an obstacle
        if self.world_data[new_position] == self.obstacle:
            return False

        self.agent_pos = new_position
        return True


def visualise_gridworld(world: GridWorld):
    fig, ax = plt.subplots()
    agent, = ax.plot([], [], 'ro', markersize=20)
    plt.gca().invert_yaxis()
    plt.pcolormesh(world.world_data)
    def update(frame):
        agent.set_data([world.agent_pos[1]+0.5], [world.agent_pos[0]+0.5])
        return agent
    ani = animation.FuncAnimation(fig, update, frames=range(2), interval=10)
    plt.show()