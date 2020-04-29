import numpy as np
import DQN
import environment
from collections import deque
import stack_frames
import random
import matplotlib.pyplot as plt


class Worker:
    def __init__(self, hyper_params, idx):
        self.dqn = DQN.DQN(state_shape=[84, 84, 4], h_parameters=hyper_params, pretrain_length=10, stack_size=4,
                           possible_actions=possible_actions)
        self.idx = idx

    def step(self, number_of_steps):
        for i in range(number_of_steps):
            self.dqn.train()

    def eval(self, number_of_steps):
        latest_losses_mean = np.mean(self.dqn.losses[-number_of_steps:])
        return latest_losses_mean

    def explore(self):
        self.dqn.learning_rate += random.uniform(-0.01, 0.01)
        self.dqn.gamma += random.uniform(-0.01, 0.01)

    def exploit(self, best_worker):
        self.dqn.learning_rate = best_worker.dqn.learning_rate
        self.dqn.gamma = best_worker.dqn.gamma


game, possible_actions = environment.create_environment()
h_parameters = {'learning_rate': 0.01,
                'gamma': 0.01,
                'batch_size': 10,
                'max_buffer_size': 1000,
                'epsilon': 0.01}


# agent = DQN.DQN(state_shape=[84, 84, 4], h_parameters=h_parameters, pretrain_length=50, stack_size=4,
#                 possible_actions=possible_actions)
# worker1 = Worker(h_parameters)
# worker1.step(100)
# worker1.eval(10)


def play_game(game, agent, epsilon):
    # Arguments:
    #   agent: an agent instance
    #   epsilon: exploration rate
    # Returns:
    #   -
    # Implements:
    #   playing DOOM!

    # create an environment for agent to play in

    game.new_episode()

    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
    # get current state
    state = game.get_state()
    # Since after creating an enviroment we have only one frame to go with, copy it 4 times
    stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, True)
    # take action
    action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
    # get reward
    reward = game.make_action(action)
    # determine if we're in the terminal state or not
    done = game.is_episode_finished()
    # get the next state
    next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
    # add it to agent's replay buffer
    agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))

    # until we reach the terminal state:
    while not game.is_episode_finished():
        state = game.get_state()
        stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
        action = agent.get_action(np.asarray(stacked_frames).reshape(1, 84, 84, 4), epsilon)
        reward = game.make_action(action)
        done = game.is_episode_finished()
        next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
        agent.add_experience((np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, done))


def pbe(workers):

    losses = []
    min_loss = 1000000000000000
    for j in range(20):
        print(j)
        for worker in workers:
            worker.step(200)
            worker_loss = worker.eval(20)
            if worker_loss < min_loss:
                for other_worker in workers:
                    if other_worker != worker:
                        other_worker.exploit(worker)
                min_loss = worker_loss
                losses.append(min_loss)
            else:
                worker.explore()
    return losses

workers = [Worker(h_parameters, 1), Worker(h_parameters, 2), Worker(h_parameters, 3)]
losses = pbe(workers)
x = [i for i in range(len(workers[0].dqn.losses))]
plt.plot(x, workers[0].dqn.losses, label=[str])
plt.show()