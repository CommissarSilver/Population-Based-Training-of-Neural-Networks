import tensorflow as tf
from agent import Agent
from tensorflow.keras import backend as K
import random
from coordinator import Coordinator
import numpy as np

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
# TRAIN_ITERS = 100
population = {'coordinator': 'agent', 'coordinator_id': int, 'coordinator_mean_score': float}  # 16 workers


def exploit(population):
    sorted_population = sorted(population, key=lambda i: i.mean_scores, reverse=True)
    best_coordinators = sorted_population[:2]
    worst_coordinators = sorted_population[-3:]
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

    for coordinator in population:
        coordinator.episode_rewards.clear()


def explore(coordinator, best_coordinator):
    for i, agent in enumerate(coordinator.agents):
        agent.hyper_parameters = best_coordinator.agents[i].hyper_parameters
        new_learning_rate = best_coordinator.agents[i].hyper_parameters['learning_rate'] * random.uniform(0.8, 1.2)
        agent.network.optimizer.learning_rate.assign(new_learning_rate)
        agent.hyper_parameters['learning_rate'] = new_learning_rate
        agent.hyper_parameters['entropy_coefficient'] = best_coordinator.agents[i].hyper_parameters[
                                                            'entropy_coefficient'] * random.uniform(0.8, 1.2)
        agent.hyper_parameters['critic_coefficient'] = best_coordinator.agents[i].hyper_parameters[
                                                           'critic_coefficient'] * random.uniform(0.8, 1.2)


coordinators = []
f = open('{}-PBT.csv'.format('Checkers-v0'), 'a')
f.write('Coordinator ID,Episode Number,Agent ID,Agent Reward,Agent Learning Rate,Agent Entropy, Agent Critic '
        'Coefficient,Total Reward\n')
f.close()
for i in range(16):
    coordinators.append(Coordinator('Checkers-v0',
                                    {'learning_rate': round(random.uniform(0.000025, 0.005), 4),
                                     'entropy_coefficient': round(random.uniform(0.00001, 0.001), 4),
                                     'critic_coefficient': round(random.uniform(0.1, 0.5), 4),
                                     'discount_factor': 0.99, 'unroll_length': 50,
                                     'minions_num': 5},
                                    coordinator_id=i))
for j in range(1, 1000):
    for coordinator in coordinators:
        coordinator.play()
        print(f'Coordinator {coordinator.id} Finished Episode {j}')
    if j % 200 == 0:
        print('***{} Episodes Have Passed. Initiating Exploitation***'.format(j))
        exploit(coordinators)
