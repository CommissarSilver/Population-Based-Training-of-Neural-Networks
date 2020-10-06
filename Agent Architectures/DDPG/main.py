import gym
import PBT
import multiprocessing
from multiprocessing import Process
from agent import Agent

if __name__ == '__main__':
    hyper_parameters = {'tau': 0.005, 'gamma': 0.99, 'actor_learning_rate': 0.001, 'critic_learning_rate': 0.002}
    agents = [Agent(hyper_parameters, input_dims=8, env_name='LunarLanderContinuous-v2', n_actions=2, id=i) for i in
              range(4)]
    queue = multiprocessing.Queue()
    processes = []
    for j in range(4):
        agent = agents[j]
        process = Process(target=PBT.train_agent, args=(agent,))
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    print('hi')
