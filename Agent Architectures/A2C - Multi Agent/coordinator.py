import tensorflow as tf
from agent import Agent
import gym
import ma_gym
import numpy as np


class Coordinator:
    def __init__(self, environment_name, initial_hyper_parameters, coordinator_id):
        self.environment_name = environment_name
        self.id = coordinator_id
        self.environment = gym.make(environment_name)
        self.observation = self.environment.reset()
        self.number_of_agents = len(self.environment.get_action_meanings())
        self.output_dims = len(self.environment.get_action_meanings()[0])
        self.observation_dims = self.environment.observation_space[0].shape[0]
        self.hyper_parameters = initial_hyper_parameters
        self.agents = [Agent(self.observation_dims, self.output_dims, self.hyper_parameters, agent_id=i,
                             coordinator_id=coordinator_id) for i in range(self.number_of_agents)]
        self.episode_finished = False
        self.mean_scores = 0
        self.episode_number = 0
        self.episode_rewards = []
        self.total_reward = 0
        self.episode_finished = False

    def play(self, unroll_length=20, show_env=False):
        self.episode_finished = False
        steps = 0
        while not self.episode_finished:
            actions = []

            for i, observation in enumerate(self.observation):
                self.agents[i].states.append(observation)
                actions.append(self.agents[i].choose_action(observation))

            next_state, rewards, dones, info = self.environment.step(actions)

            for i, reward in enumerate(rewards):
                self.agents[i].rewards.append(reward)
                self.total_reward += reward

            if show_env:
                self.environment.render()
            self.observation = next_state

            if dones == [True for i in range(len(self.agents))]:
                self.episode_finished = True
                self.episode_number += 1
                self.observation = self.environment.reset()
                self.episode_rewards.append(self.total_reward)
                self.total_reward = 0
                self.mean_scores = np.mean(self.episode_rewards)
                f = open('{}-Single-5000.csv'.format(self.environment_name), 'a')
                f.write('{},{},{}\n'.format(self.id, self.episode_number, self.episode_rewards[-1]))

            steps += 1

        for agent in self.agents:
            agent.learn()
        #     for agent in self.agents:
        #         f = open('{}-PBT.csv'.format(self.environment_name), 'a')
        #         f.write(
        #             '{},{},{},{},{},{}\n'.format(self.id, agent.id, agent.network.optimizer.learning_rate.numpy(),
        #                                          self.episode_number,
        #                                          np.sum(agent.rewards),
        #                                          total_score))
        #
        #         agent.learn()
        #         # if episode_number % 100 == 0:
        #         agent.save_models()
        #     scores.append(total_score)
        #     print('{} episode {} finished'.format(self.id, self.episode_number))


env1 = Coordinator('Checkers-v0',
                   {'learning_rate': 0.0001, 'discount_factor': 0.99, 'unroll_length': 50, 'minions_num': 5},
                   coordinator_id=0)
while env1.episode_number<5000:
    env1.play()
    print(env1.episode_number)
