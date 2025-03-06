import torch
from agent import AgentEnvironment
from gridworld import GridWorld
from qlearning import QLearningAgent
import matplotlib.pyplot as plt
from matplotlib import animation

def q_values_in_matrix(world: GridWorld, agent: QLearningAgent[tuple[int, int], int]) -> torch.Tensor:
    matrix = torch.zeros_like(world.world_data)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            if (x, y) in agent.q_table:
                matrix[x, y] = float(torch.max(agent.q_table[(x, y)]))
    return matrix


def rewards_in_matrix(agent_environment: AgentEnvironment[tuple[int, int], int]) -> torch.Tensor:
    matrix = torch.zeros_like(agent_environment.environment.world_data)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            matrix[x, y] = agent_environment.reward_func((x, y))
    return matrix


def visualise_gridworld(agent_environment: AgentEnvironment[tuple[int, int], int]):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    agent_dot, = ax1.plot([], [], 'ro', markersize=20)
    world = agent_environment.environment
    gridworld_mesh = ax1.pcolormesh(world.world_data)
    ax1.invert_yaxis()
    ax1.set_title("World")
    ax2.set_title("Expected Future Returns (Q)")
    ax3.set_title("Reward Function")
    agent = agent_environment.agent
    matrix = q_values_in_matrix(world, agent)
    q_mesh = ax2.pcolormesh(matrix.numpy(), vmin=-10, vmax=10)
    ax2.invert_yaxis()
    rewards = rewards_in_matrix(agent_environment)
    reward_mesh = ax3.pcolormesh(rewards.detach().numpy(), vmin=-3, vmax=3)
    ax3.invert_yaxis()


    def update(frame):
        agent_dot.set_data([world.state[1] + 0.5], [world.state[0] + 0.5])
        matrix = q_values_in_matrix(world, agent)
        q_mesh.set_array(matrix.numpy().flatten())
        reward_mesh.set_array(rewards.detach().numpy().flatten())
        return agent_dot, q_mesh, reward_mesh
    ani = animation.FuncAnimation(fig, update, frames=range(2), interval=10)
    plt.colorbar(q_mesh, ax=ax2)
    plt.colorbar(reward_mesh, ax=ax3)
    plt.show()