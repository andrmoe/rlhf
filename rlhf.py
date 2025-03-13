import time

import numpy as np

from agent import Agent, Environment, AgentEnvironment
from reward_model import RewardModel, train
from preference_oracle import PreferenceOracle
from typing import Callable, TypeVar, Iterable, Generic
from torch import Tensor
import multiprocessing
from copy import deepcopy
from random import random
import torch
import heapq

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions

def run_rlhf_agent_env(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
                       trajectory_length: int, req_human_feedback_pipe: multiprocessing.Pipe,
                       next_eval_pipe: multiprocessing.Pipe,  model_weights_pipe: multiprocessing.Pipe):
    initial_state = environment.state
    step_limit = 200
    feedback_probability = 0.1
    trajectories: [[S]] = []
    evaluated_trajectories: [[S]] = []
    req_human_feedback_pipe[1].close()
    next_eval_pipe[0].close()
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

                if len(trajectory) == trajectory_length and trajectory not in evaluated_trajectories and trajectory not in trajectories:
                    this_traj = trajectory[:trajectory_length]
                    trajectories.append(this_traj)
                if next_eval_pipe[1].poll(0):
                    traj0, traj1 = next_eval_pipe[1].recv()
                    # We could add both trajectories here, but it might be better to enable comparison with an old trajectory
                    if traj0 not in evaluated_trajectories:
                        evaluated_trajectories.append(traj0)
                    if traj1 not in evaluated_trajectories:
                        evaluated_trajectories.append(traj1)
                    # candidates = [traj for traj in trajectories if traj not in evaluated_trajectories and traj != this_traj]
                    if len(trajectories) >= 2:
                        ps = np.array([np.exp(reward_model.trajectory_variance(t)) for t in trajectories])
                        ps /= np.sum(ps)
                        print(ps)
                        first_traj_index = np.random.choice(list(range(len(trajectories))), p=ps)
                        second_traj_index = np.random.choice(list(range(len(trajectories))), p=ps)
                        first_traj = trajectories[first_traj_index]
                        second_traj = trajectories[second_traj_index]
                        #pair = tuple(heapq.nlargest(2, trajectories, key=reward_model.trajectory_variance))
                        print(f'Requesting human feedback on: {first_traj}, {second_traj}')
                    else:
                        first_traj = second_traj = trajectory_length*[initial_state]
                    req_human_feedback_pipe[0].send((first_traj, second_traj, model_weights))
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
    next_eval_pipe = multiprocessing.Pipe()

    rl_process = multiprocessing.Process(target=run_rlhf_agent_env, daemon=True,
                                         args=(agent, environment, deepcopy(reward_model), trajectory_length,
                                               req_human_feedback_pipe, next_eval_pipe, model_weights_pipe))
    reward_process = multiprocessing.Process(target=train, daemon=True, args=(reward_model, preference_pipe, model_weights_pipe))
    rl_process.start()
    reward_process.start()

    def preference_callback(traj0: [S], traj1: [S], preference: Tensor):
        feedback_triple = (traj0, traj1, preference)
        print(f'Sending human preference: {feedback_triple}')
        preference_pipe[0].send(feedback_triple)

    def next_pair_callback(prev_traj0: [S], prev_traj1: [S]) \
            -> tuple[[S], [S], dict[str, torch.nn.Parameter]]:
        next_eval_pipe[0].send((prev_traj0, prev_traj1))
        req_human_feedback_pipe[1].poll()
        pair_and_reward_model_params = req_human_feedback_pipe[1].recv()
        print(f"Received traj pair ({pair_and_reward_model_params[0]}, {pair_and_reward_model_params[1]})")
        return pair_and_reward_model_params


    preference_oracle.register_callbacks(preference_callback, next_pair_callback)
    preference_pipe[1].close()

    req_human_feedback_pipe[0].close()
    next_eval_pipe[1].close()

    preference_oracle.start()
