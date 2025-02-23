from maze import small_maze
from gridworld import visualise_gridworld, GridWorld
from agent import AgentEnvironment, Agent
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
    while True:
        agent_environment.step()

if __name__ == '__main__':
    agent = Agent(small_maze.state, manual_control)
    agent_environment = AgentEnvironment(agent, small_maze)
    t = threading.Thread(target=agent_loop, args=[agent_environment], daemon=True)
    t.start()
    visualise_gridworld(small_maze)
