import tensorflow as tf
import A2C
import random
import threading
import numpy as np

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


class Worker:
    def __init__(self, input_shape, possible_actions, hyper_params, id):
        self.master_agent = A2C.Agent(input_shape, possible_actions, hyper_params)
        self.id = id

    def explore(self):
        self.master_agent.learning_rate *= random.uniform(0.8, 1.2)

    def exploit(self, best_worker):
        self.master_agent.learning_rate = best_worker.master_agent.learning_rate
        self.master_agent.model.set_weights(best_worker.model.get_weights())


def learn(worker, number_of_epochs):
    worker_losses = []
    for epoch in range(number_of_epochs):
        worker_losses.append(worker['worker'].master_agent.train())
    worker['score'] = np.mean(worker_losses, axis=0)


def create_population(population_size, initial_hyper_params):
    input_shape = (84, 84, 4)
    possible_actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    population = {'{0}'.format(i): {'worker': Worker, 'score': int} for i in range(population_size)}
    for i in range(population_size):
        population['{0}'.format(i)]['worker'] = Worker(input_shape=input_shape, possible_actions=possible_actions,
                                                       hyper_params=initial_hyper_params, id=i)
        population['{0}'.format(i)]['score'] = 0
    return population


pop = create_population(10, {'learning_rate': 0.001, 'discount_factor': 0.95, 'minions_num': 1})
threads = []
for worker_id in pop:
    thread = threading.Thread(target=learn, args=(pop[worker_id], 10))
    threads.append(thread)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print('hi')
