import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import numpy as np
from matplotlib.widgets import Button
from preference_oracle import PreferenceOracle
from visual import agent_in_gridworld
from gridworld import GridWorld
from copy import deepcopy
from rlhf_message import RLHFMessage

class Gui(PreferenceOracle[tuple[int, int]]):
    def __init__(self, world: GridWorld, trajectory_length: int):
        super().__init__()
        self.world = deepcopy(world)
        self.trajectory_length = trajectory_length
        self.trajectories = None
        self.model_params = {}

    def animate_trajectory(self, frame: int, traj: [tuple[int, int]]) -> np.ndarray:
        if frame < self.trajectory_length:
            self.world.state = traj[frame]
            return agent_in_gridworld(self.world)
        else:
            return self.world.world_data.numpy()

    def feedback(self, preference: torch.Tensor):
        rlhf_message = RLHFMessage()
        rlhf_message.trajectories = self.trajectories
        rlhf_message.preference = preference
        self.preference_callback(rlhf_message)
        rlhf_message: RLHFMessage[tuple[int, int]] = self.next_pair_callback(rlhf_message)
        self.model_params = rlhf_message.reward_model_weights
        self.trajectories = rlhf_message.trajectories

    def left(self, _):
        self.feedback(torch.tensor([1,0]))

    def indifferent(self, _):
        self.feedback(torch.tensor([0.5, 0.5]))

    def right(self, _):
        self.feedback(torch.tensor([0,1]))

    def start(self):
        while self.trajectories is None:
            rlhf_message: RLHFMessage[tuple[int, int]] = self.next_pair_callback(RLHFMessage[tuple[int, int]]())
            self.trajectories = rlhf_message.trajectories
        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        world_meshes = [ax.pcolormesh(self.world.world_data.numpy(), vmin=-1, vmax=3) for ax in axes[1, :]]
        model_meshes = [ax.pcolormesh(np.zeros(self.world.world_data.shape), vmin=-1, vmax=3) for ax in axes[0, :]]
        axes[0, 0].set_title('Mean Reward')
        axes[0, 1].set_title('Variance in Reward')

        buttons = [('Left', self.left), ('Indifferent', self.indifferent), ('Right', self.right)]
        if buttons:
            fig.subplots_adjust(bottom=0.2)
            button_shift = -0.1
            button_axes = [fig.add_axes([(i + 1) / (len(buttons) + 1) + button_shift, 0.05, 0.1, 0.075]) for i in range(len(buttons))]
            button_widgets = [Button(ax, t[0]) for ax, t in zip(button_axes, buttons)]
            for button, t in zip(button_widgets, buttons):
                button.on_clicked(t[1])

        for ax in axes.flatten():
            ax.invert_yaxis()

        def update(frame):
            if self.trajectories:
                if self.model_params:
                    params = np.array([param.detach().numpy() for param in self.model_params.values()])
                    two_params = np.mean(params, axis=0), np.var(params, axis=0)
                    model_update = [mesh.set_array(param) for mesh, param in zip(model_meshes, two_params)]
                else:
                    model_update = []
                return [mesh.set_array(self.animate_trajectory(frame, traj)) for mesh, traj in
                    zip(world_meshes, self.trajectories)] + model_update

        ani = animation.FuncAnimation(fig, update, frames=range(self.trajectory_length+1), interval=100)
        for mesh, ax in zip(model_meshes, axes[0, :]):
            plt.colorbar(mesh, ax=ax)
        for mesh, ax in zip(world_meshes, axes[1, :]):
            plt.colorbar(mesh, ax=ax)
        #plt.tight_layout()
        plt.show()