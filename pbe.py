import tensorflow as tf
import numpy as np
import DQN


class Worker:
    def __init__(self, idx, pop_statistics):
        self.idx = idx
        self.dqn = DQN.DQN()
        self.population_stats = pop_statistics

    def eval(self):
        loss = self.dqn.train().loss


pop_statisctics = []
