import time

from agent import Agent, Environment, AgentEnvironment
from reward_model import RewardModel, train
from preference_oracle import PreferenceOracle
from typing import Callable, TypeVar, Iterable, Generic
from torch import Tensor
import multiprocessing
from copy import deepcopy
from random import random
import torch

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions

def run_rlhf_agent_env(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
                       trajectory_length: int, req_human_feedback_pipe: multiprocessing.Pipe,
                       model_weights_pipe: multiprocessing.Pipe):
    initial_state = environment.state
    step_limit = 200
    feedback_probability = 0.1
    trajectories: [[S]] = []
    evaluated_trajectories: [[S]] = []
    req_human_feedback_pipe[1].close()
    model_weights_pipe[0].close()
    reward_model.eval()
    with torch.no_grad():
        model_weights = dict(reward_model.named_parameters())
        while True:
            environment.state = agent.state = initial_state
            elapsed_steps = 0
            trajectory: [S] = [initial_state]
            if model_weights_pipe[1].poll(0):
                model_weights = model_weights_pipe[1].recv()
                print('Received new model weights')
                for param, old_param in zip(model_weights.values(), dict(reward_model.named_parameters()).values()):
                    pass
                    #print(torch.norm(param-old_param))
                reward_model.load_state_dict(model_weights)
            while True:
                if environment.is_terminal() or elapsed_steps > step_limit:
                    break

                if len(trajectory) == trajectory_length and trajectory not in evaluated_trajectories:
                    this_traj = trajectory[:trajectory_length]
                    other_candidates = [traj for traj in trajectories if traj not in evaluated_trajectories and traj != this_traj]
                    if random() < feedback_probability and other_candidates:
                        other_traj = max(other_candidates, key=reward_model.trajectory_variance)

                        print(f'Requesting human feedback on: {this_traj}, {other_traj}')
                        req_human_feedback_pipe[0].send((this_traj, other_traj, model_weights))
                        evaluated_trajectories.append(trajectory)
                    trajectories.append(this_traj)

                action = agent.act()
                new_state = environment.transition(action)
                reward = reward_model.reward(new_state)
                trajectory.append(new_state)
                agent.observe(new_state, float(reward))
                elapsed_steps += 1


def rlhf(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
         preference_oracle: PreferenceOracle[S], trajectory_length: int):
    req_human_feedback_pipe = multiprocessing.Pipe()
    model_weights_pipe = multiprocessing.Pipe()
    preference_pipe = multiprocessing.Pipe()

    rl_process = multiprocessing.Process(target=run_rlhf_agent_env, daemon=True,
                                         args=(agent, environment, deepcopy(reward_model), trajectory_length,
                                               req_human_feedback_pipe, model_weights_pipe))
    reward_process = multiprocessing.Process(target=train, daemon=True, args=(reward_model, preference_pipe, model_weights_pipe))
    rl_process.start()
    reward_process.start()

    def preference_callback(traj0: [S], traj1: [S], preference: Tensor):
        feedback_triple = (traj0, traj1, preference)
        print(f'Sending human preference: {feedback_triple}')
        preference_pipe[0].send(feedback_triple)

    def next_pair_callback() -> tuple[[S], [S], dict[str, torch.nn.Parameter]]:
        if req_human_feedback_pipe[1].poll():
            pair_and_reward_model_params = req_human_feedback_pipe[1].recv()
            print(f"Received traj pair ({pair_and_reward_model_params[0]}, {pair_and_reward_model_params[1]})")
            return pair_and_reward_model_params


    preference_oracle.register_callbacks(preference_callback, next_pair_callback)
    preference_pipe[1].close()
    req_human_feedback_pipe[0].close()
    preference_oracle.start()
