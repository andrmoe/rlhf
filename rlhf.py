import time

from agent import Agent, Environment, AgentEnvironment
from reward_model import RewardModel, train
from preference_oracle import PreferenceOracle
from typing import Callable, TypeVar, Iterable, Generic
from torch import Tensor
import multiprocessing
from copy import deepcopy
from random import random

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions

def run_rlhf_agent_env(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
                       trajectory_length: int, req_human_feedback_pipe: multiprocessing.Pipe,
                       model_weights_pipe: multiprocessing.Pipe):
    initial_state = environment.state
    step_limit = 200
    feedback_probability = 0.01
    trajectories: [[S]] = []
    req_human_feedback_pipe[1].close()
    model_weights_pipe[0].close()
    while True:
        environment.state = agent.state = initial_state
        elapsed_steps = 0
        trajectory: [S] = [initial_state]
        while True:
            if environment.is_terminal() or elapsed_steps > step_limit:
                break
            if model_weights_pipe[1].poll(0):
                new_model_weights = model_weights_pipe[1].recv()
                print('Received new model weights')
                reward_model.load_state_dict(new_model_weights)
            if len(trajectory) == trajectory_length:
                this_traj = trajectory[:trajectory_length]
                if random() < feedback_probability:
                    other_traj = max(trajectories, key=reward_model.trajectory_variance)

                    print(f'Requesting human feedback on: {this_traj}, {other_traj}')
                    req_human_feedback_pipe[0].send((this_traj, other_traj))
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
    reward_process = multiprocessing.Process(target=train, daemon=True, args=(preference_pipe, model_weights_pipe))
    rl_process.start()
    reward_process.start()

    def preference_callback(traj0: [S], traj1: [S], preference: Tensor):
        feedback_triple = (traj0, traj1, preference)
        print(f'Sending human preference: {feedback_triple}')
        preference_pipe[0].send(feedback_triple)

    def next_pair_callback() -> tuple[[S], [S]]:
        if req_human_feedback_pipe[1].poll(20):
            pair = req_human_feedback_pipe[1].recv()
            print(f"Received traj pair ({pair})")
            return pair


    preference_oracle.register_callbacks(preference_callback, next_pair_callback)
    preference_pipe[1].close()
    req_human_feedback_pipe[0].close()
    preference_oracle.start()
