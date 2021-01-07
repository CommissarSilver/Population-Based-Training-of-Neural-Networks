import tensorflow as tf
from agent import Agent
import gym
import ma_gym
import numpy as np
import random


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

    def play(self, unroll_length=2, show_env=False):
        self.episode_finished = False
        steps = 0
        while (steps < unroll_length and not self.episode_finished):
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
                for agent in self.agents:
                    f = open('{}-Non PBT.csv'.format('Checkers-v0'), 'a')
                    f.write(
                        '{},{},{},{},{},{},{},{}\n'.format(self.id, self.episode_number, agent.id,
                                                           np.sum(agent.rewards),
                                                           agent.hyper_parameters['learning_rate'],
                                                           agent.hyper_parameters['entropy_coefficient'],
                                                           agent.hyper_parameters['critic_coefficient'],
                                                           self.episode_rewards[-1]))
                    f.close()

            steps += 1

        for agent in self.agents:
            agent.learn()


f = open('{}-Non PBT.csv'.format('Checkers-v0'), 'a')
f.write('Coordinator ID,Episode Number,Agent ID,Agent Reward,Agent Learning Rate,Agent Entropy, Agent Critic '
        'Coefficient,Total Reward\n')
f.close()
coordinator1 = Coordinator('Checkers-v0',
                           {'learning_rate': 0.00001,
                            'entropy_coefficient': 0.001,
                            'critic_coefficient': 0.3,
                            'discount_factor': 1,
                            'unroll_length': 5,
                            'minions_num': 5},
                           coordinator_id=0)

while coordinator1.episode_number < 2000:
    coordinator1.play()
    print(coordinator1.episode_number)
