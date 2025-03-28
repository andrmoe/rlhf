from torch import Tensor
from agent import Environment


class GridWorld(Environment[tuple[int, int], int]):
    def __init__(self, world_data: Tensor, agent_pos: tuple[int, int]):
        super().__init__(agent_pos)
        self.world_data = world_data
        self.east = 0
        self.north = 1
        self.west = 2
        self.south = 3
        self.obstacle = -1
        self.terminal_square = 1


    def move(self, direction: int) -> bool:
        f"""
        Move the agent in a particular direction. Agent remains stationary if the movement is impossible.
        :param direction An int between 0 and 3. east={self.east}, north={self.north}, west={self.west}, south={self.south}.
        :return True if movement happened, False if blocked.
        """
        if direction == self.east:
            new_position = self.state[0], self.state[1] + 1
        elif direction == self.north:
            new_position = self.state[0] - 1, self.state[1]
        elif direction == self.west:
            new_position = self.state[0], self.state[1] - 1
        elif direction == self.south:
            new_position = self.state[0] + 1, self.state[1]
        else:
            raise ValueError(
                f"Direction must be between 0 and 3, east={self.east}, north={self.north}, west={self.west}, south={self.south}")

        # The agent doesn't move if the direction is into the edge of the world
        if (direction == self.east and self.state[1] == self.world_data.shape[1] - 1) or (
                direction == self.north and self.state[0] == 0) or (
                direction == self.west and self.state[1] == 0) or (
                direction == self.south and self.state[0] == self.world_data.shape[0] - 1):
            return False

        # The agent doesn't move if the direction is into an obstacle
        if self.world_data[new_position] == self.obstacle:
            return False

        self.state = new_position
        return True

    def transition(self, action: int) -> tuple[int, int]:
        self.move(action)
        return self.state

    def is_terminal(self) -> bool:
        return bool(self.world_data[self.state] == self.terminal_square)

