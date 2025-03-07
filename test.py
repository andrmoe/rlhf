import torch

from gridworld import GridWorld
from torch import Tensor
from agent import Agent, Environment, AgentEnvironment
from reward_model import GridWorldRewardModel

def test_gridworld():
    data = Tensor([[ 0, -1, -1],
                   [ 0,  0, -1],
                   [-1,  0,  0]])
    world = GridWorld(data, (0, 0))
    assert world.move(world.east) == False
    assert world.state == (0, 0)
    assert world.move(world.north) == False
    assert world.state == (0, 0)
    assert world.move(world.west) == False
    assert world.state == (0, 0)
    assert world.move(world.south) == True
    assert world.state == (1, 0)
    assert world.move(world.south) == False
    assert world.state == (1, 0)
    assert world.move(world.east) == True
    assert world.state == (1, 1)
    assert world.move(world.south) == True
    assert world.state == (2, 1)
    assert world.move(world.south) == False
    assert world.state == (2, 1)
    assert world.move(world.east) == True
    assert world.state == (2, 2)
    assert world.move(world.east) == False
    assert world.state == (2, 2)


def test_generic_agent_environment():
    # S = A = bool
    agent = Agent(False, lambda state: ((0, state), (1, not state)))
    environment = Environment(False, lambda state, action: action)
    agent_environment = AgentEnvironment(agent, environment)
    assert agent.state == environment.state == False
    next_state = agent_environment.step()
    assert next_state == agent.state == environment.state == True


def test_reward_model_loss():
    reward_model = GridWorldRewardModel((5,5), 10)
    trajectory = [(i, i) for i in range(5)]
    encoded_traj = reward_model.encode_trajectory(trajectory)

    assert reward_model.exp_sum(encoded_traj).shape == (10,)
    assert reward_model.loss(trajectory, trajectory, torch.tensor([0.5, 0.5])) == -torch.log(torch.tensor(0.5))