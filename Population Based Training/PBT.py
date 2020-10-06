import tensorflow as tf
from agent import Agent
import random

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
# TRAIN_ITERS = 100

hyper_parameters = {'tau': 0.005, 'gamma': 0.99, 'actor_learning_rate': 0.001, 'critic_learning_rate': 0.002,
                    'number_of_minions': 1}

population = {'agent': 'agent', 'agent_id': int, 'agent_mean_score': float}  # 16 workers


def exploit(population):
    sorted_population = sorted(population, key=lambda i: i['agent_mean_score'], reverse=True)
    best_agents = sorted_population[:3]
    worst_agents = sorted_population[:-3]
    agent_to_exploit
    print('hi')
