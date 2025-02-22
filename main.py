from maze import small_maze
from gridworld import visualise_gridworld, GridWorld
import threading


def manual_control(world: GridWorld):
    while True:
        wasd = input("WASD:")
        if wasd == 'd':
            world.move(world.east)
        elif wasd == 'w':
            world.move(world.north)
        elif wasd == 'a':
            world.move(world.west)
        elif wasd == 's':
            world.move(world.south)
        print(world.agent_pos)

if __name__ == '__main__':
    t = threading.Thread(target=manual_control, args=[small_maze], daemon=True)
    t.start()
    visualise_gridworld(small_maze)
