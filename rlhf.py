import time

from agent import Agent, Environment, AgentEnvironment
from reward_model import RewardModel, train
from typing import Callable, TypeVar, Iterable
from torch import Tensor
import multiprocessing

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions


def pretend_rl(req_human_feedback_pipe: multiprocessing.Pipe,
               model_weights_pipe: multiprocessing.Pipe):
    # TODO: Implement training
    req_human_feedback_pipe[1].close()
    model_weights_pipe[0].close()
    steps = 0
    while True:
        if steps % 10 == 0:
            print('Sending trajectory pair to be evaluated')
            req_human_feedback_pipe[0].send(f"Trajectory pair to be evaluated {steps}")
        if model_weights_pipe[1].poll(0):
            new_preference_data = model_weights_pipe[1].recv()
            print(f"Received updated reward model ({new_preference_data})")
        time.sleep(0.1)
        steps += 1


def rlhf(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
         preference_oracle: Callable[[Iterable[S], Iterable[S]], Tensor]):
    agent_environment = AgentEnvironment(agent, environment, lambda s: float(reward_model.reward(s)[0]))

    preference_database = []
    req_human_feedback_pipe = multiprocessing.Pipe()
    model_weights_pipe = multiprocessing.Pipe()
    preference_pipe = multiprocessing.Pipe()

    rl_process = multiprocessing.Process(target=pretend_rl, args=(req_human_feedback_pipe, model_weights_pipe),
                                         daemon=True)
    reward_process = multiprocessing.Process(target=train, args=(preference_pipe, model_weights_pipe),
                                             daemon=True)
    rl_process.start()
    reward_process.start()
    while True:
        preference_pipe[1].close()
        req_human_feedback_pipe[0].close()
        steps = 0
        current_pair = None
        while True:
            if current_pair:
                time.sleep(5)
                print('Sending human preference')
                preference_pipe[0].send(f"Human preference {current_pair}, {steps}")
                current_pair = None
            if model_weights_pipe[1].poll(0):
                current_pair = req_human_feedback_pipe[1].recv()
                print(f"Received traj pair ({current_pair})")
            steps += 1
