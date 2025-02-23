from gridworld import GridWorld
from torch import Tensor
from agent import Agent, Environment, AgentEnvironment

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