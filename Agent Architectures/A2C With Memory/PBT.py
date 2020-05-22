import tensorflow as tf
import A2C
import random


gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def create_worker(TRAIN_ITERS, best_worker_id, hyper_params, i, queue):
    print("Worker {} Starting".format(i))
    worker = A2C.Agent((84, 84, 4), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], hyper_params)  # builds an agent with the given
    # hyper_params
    worker.model.load_weights('Agent-{0} Model.h5'.format(best_worker_id))  # loads weights from the best agent so far
    worker_id = i
    print(" -> Worker {} Training".format(i))
    returns = worker.train(TRAIN_ITERS, i)  # trains for the given number of iterations
    print("         -> Worker {} returns".format(i), returns)
    worker.model.save('Agent-{0} Model.h5'.format(worker_id))  # saves the agents  model
    queue.put({'agent_id': worker_id,
               'hyper_params': worker.hyper_params,
               'returns': returns})


def explore(hyper_params):
    j = random.uniform(0.7, 1.5)  # perturbs the agents learning rate
    hyper_params['learning_rate'] *= j
    return hyper_params


def exploit(training_results):
    best_agent = sorted(training_results, key=lambda i: i['returns'], reverse=True)[0]  # sort the agents according
    # the their mean return
    for agent in training_results:
        agent['agent_id'] = best_agent['agent_id']
        agent['hyper_params'] = explore(best_agent['hyper_params'])
        agent['returns'] = []
    return training_results
# # def analyze(results):
