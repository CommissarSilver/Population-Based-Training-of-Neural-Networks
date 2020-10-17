import tensorflow as tf
from agent import Agent
from tensorflow.keras import backend as K
import random
from coordinator import Coordinator
import numpy as np

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
# TRAIN_ITERS = 100

hyper_parameters = {'tau': 0.005, 'gamma': 0.99, 'actor_learning_rate': 0.001, 'critic_learning_rate': 0.002,
                    'number_of_minions': 1}

population = {'coordinator': 'agent', 'coordinator_id': int, 'coordinator_mean_score': float}  # 16 workers


def exploit(population):
    sorted_population = sorted(population, key=lambda i: i.mean_scores, reverse=True)
    best_coordinators = sorted_population[:2]
    worst_coordinators = sorted_population[-2:]
    coordinators = best_coordinators + worst_coordinators
    coordinator_to_exploit = random.choice(best_coordinators)
    # save agent's model
    for agent in coordinator_to_exploit.agents:
        agent.save_models()
    exploited_hyper_parameters = coordinator_to_exploit.hyper_parameters
    # for each other agent, load their models here
    for coordinator in worst_coordinators:
        for i, agent in enumerate(coordinator.agents):
            agent.network.load_weights(coordinator_to_exploit.agents[i].network.checkpoint_file)
            agent.hyper_parameters = exploited_hyper_parameters
            K.set_value(agent.network.optimizer.learning_rate,
                        coordinator_to_exploit.agents[i].network.optimizer.learning_rate)
    for coordinator in worst_coordinators:
        explore(coordinator, coordinator_to_exploit)
    print('exploit done')


def explore(coordinator, best_coordinator):
    for i, agent in enumerate(coordinator.agents):
        agent.hyper_parameters = best_coordinator.agents[i].hyper_parameters
        x = best_coordinator.agents[i].hyper_parameters['learning_rate'] * random.uniform(0.8, 1.2)
        # K.set_value(agent.network.optimizer.learning_rate, x)
        agent.network.optimizer.learning_rate.assign(x)
        print(agent.network.optimizer.learning_rate)
        agent.hyper_parameters['learning_rate'] = x
    print('explore done')


# env1 = Coordinator('Checkers-v0',
#                    {'learning_rate': 0.0001, 'discount_factor': 0.99, 'unroll_length': 50, 'minions_num': 5},
#                    coordinator_id=0)
coordinators = []
for i in range(8):
    coordinators.append(Coordinator('Checkers-v0',
                                    {'learning_rate': round(random.uniform(0.000025, 0.0025), 4),
                                     'discount_factor': 0.99, 'unroll_length': 50,
                                     'minions_num': 5},
                                    coordinator_id=i))
f = open('{}-PBT.csv'.format('Checkers-v0'), 'a')
f.write('Coordinator ID,Episode Number,Total Reward\n')
f.close()
for j in range(1, 10000):
    for coordinator in coordinators:
        coordinator.play()
    if j % 200 == 0:
        print(j)
        exploit(coordinators)
