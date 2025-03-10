import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import animation
import torch
import numpy as np
from matplotlib.widgets import Button
from preference_oracle import PreferenceOracle
from visual import agent_in_gridworld
from gridworld import GridWorld
from copy import deepcopy





class Gui(PreferenceOracle[tuple[int, int]]):
    def __init__(self, world: GridWorld, trajectory_length: int):
        super().__init__()
        self.world = deepcopy(world)
        self.trajectory_length = trajectory_length
        self.trajectories = None

    def animate_trajectory(self, frame: int, traj: [tuple[int, int]]) -> np.ndarray:
        if frame < self.trajectory_length:
            self.world.state = traj[frame]
            return agent_in_gridworld(self.world)
        else:
            return self.world.world_data.numpy()

    def feedback(self, preference: torch.Tensor):
        self.preference_callback(self.trajectories[0], self.trajectories[1], preference)
        self.trajectories = self.next_pair_callback()

    def left(self, _):
        self.feedback(torch.tensor([1,0]))

    def right(self, _):
        self.feedback(torch.tensor([0,1]))

    def start(self):
        while self.trajectories is None:
            self.trajectories = self.next_pair_callback()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        meshes = [ax.pcolormesh(self.world.world_data.numpy(), vmin=-1, vmax=3) for ax in axes]
        buttons = [('Left', self.left), ('Right', self.right)]
        if buttons:
            fig.subplots_adjust(bottom=0.2)
            button_axes = [fig.add_axes([(i + 1) / (len(buttons) + 1), 0.05, 0.1, 0.075]) for i in range(len(buttons))]
            button_widgets = [Button(ax, t[0]) for ax, t in zip(button_axes, buttons)]
            for button, t in zip(button_widgets, buttons):
                button.on_clicked(t[1])
        for ax in axes:
            ax.invert_yaxis()

        def update(frame):
            if self.trajectories:
                return [mesh.set_array(self.animate_trajectory(frame, traj)) for mesh, traj in
                        zip(meshes, self.trajectories)]

        ani = animation.FuncAnimation(fig, update, frames=range(self.trajectory_length+1), interval=100)
        for mesh, ax in zip(meshes, axes):
            plt.colorbar(mesh, ax=ax)
        plt.show()