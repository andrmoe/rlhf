import time

from agent import Agent, Environment, AgentEnvironment
from reward_model import RewardModel, train
from typing import Callable, TypeVar, Iterable
from torch import Tensor
import multiprocessing
from copy import deepcopy

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions

def run_rlhf_agent_env(agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S],
                       req_human_feedback_pipe: multiprocessing.Pipe, model_weights_pipe: multiprocessing.Pipe):
    agent_environment = RlhfAgentEnvironment(agent, environment, reward_model)
    initial_state = agent_environment.environment.state
    while True:
        agent_environment.environment.state = agent_environment.agent.state = initial_state
        elapsed_steps = 0
        while True:
            if agent_environment.environment.is_terminal() or elapsed_steps > agent_environment.step_limit:
                break
            agent_environment.step()
            elapsed_steps += 1
            if agent_environment.sleep_time:
                time.sleep(agent_environment.sleep_time)


class RlhfAgentEnvironment(AgentEnvironment[S, A]):
    def __init__(self, agent: Agent[S, A], environment: Environment[S, A], reward_model: RewardModel[S]):
        super().__init__(agent, environment, lambda s: float(reward_model.reward(s)[0]))


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
    rlhf_agent_env = RlhfAgentEnvironment(agent, environment, deepcopy(reward_model))
    preference_database = []
    req_human_feedback_pipe = multiprocessing.Pipe()
    model_weights_pipe = multiprocessing.Pipe()
    preference_pipe = multiprocessing.Pipe()

    rl_process = multiprocessing.Process(target=run_rlhf_agent_env, args=(agent, environment,
                                                                                deepcopy(reward_model),
                                                                                req_human_feedback_pipe,
                                                                                model_weights_pipe),
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
