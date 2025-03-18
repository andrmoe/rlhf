from operator import indexOf

import numpy as np

from agent import Agent
from typing import TypeVar, Iterable
from torch import Tensor
import torch

S = TypeVar("S")  # Type for states
A = TypeVar("A")  # Type for actions


class QLearningAgent(Agent[S, A]):
    def __init__(self, state: S, possible_actions=Iterable[A], discount_factor: float = 0.9, learning_rate: float = 0.1,
                exploration_rate: float = 0.01, temperature: float = 0.5):
        super().__init__(state)
        # The tensors here are probabilities of taking an action from self.possible_actions
        self.possible_actions = tuple(possible_actions)
        self.q_table: dict[S, Tensor] = {state: torch.zeros(len(self.possible_actions))}
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.temperature = temperature

    def policy(self, state: S) -> Iterable[tuple[float, A]]:
        max_value = torch.max(self.q_table[state])
        # max_one_hot = (self.q_table[state] == max_value).int()
        # max_value_count = torch.sum(max_one_hot)
        # if max_value_count == len(self.possible_actions):
        #     return zip(torch.ones(len(self.possible_actions))/len(self.possible_actions), self.possible_actions)
        # non_optimal_probability = self.exploration_rate/(len(self.possible_actions)-max_value_count)
        # action_distribution = torch.full_like(self.q_table[state], non_optimal_probability)
        # action_distribution += max_one_hot * (1 - self.exploration_rate)/max_value_count
        action_distribution = torch.softmax(self.q_table[state]/self.temperature, dim=0)
        return zip(action_distribution, self.possible_actions)

    def observe(self, new_state: S, reward: float):
        if new_state not in self.q_table:
            self.q_table[new_state] = torch.zeros(len(self.possible_actions))
        if self.most_recent_action is not None:
            temp_diff_target = reward + self.discount_factor * max(self.q_table[new_state])
            action_index = indexOf(self.possible_actions, self.most_recent_action)
            if action_index is None:
                raise Exception(F"The most recent action {self.most_recent_action} is not in the "
                                F"QLearningAgent's possible_actions")
            prev_q = self.q_table[self.state][action_index]
            self.q_table[self.state][action_index] = (1-self.learning_rate)*prev_q+self.learning_rate*temp_diff_target

        self.state = new_state

