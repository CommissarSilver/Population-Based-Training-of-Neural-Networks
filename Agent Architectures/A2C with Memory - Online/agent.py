import tensorflow as tf
import tensorflow_probability as tfp
from network import ActorNetwork, CriticNetwork
import threading
import random
import numpy as np
import gym

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent:
    def __init__(self, environment, initial_hyper_parameters, id, log_file_name):
        self.environment_name = environment
        self.environment = gym.make('{}'.format(self.environment_name))
        self.episode_finished = False
        self.log_file_name = log_file_name
        self.state = self.environment.reset()
        self.next_state = []
        self.cum_sum = 0
        self.episode_num = 0
        self.episode_rewards = []
        self.id = id

        self.hyper_parameters = initial_hyper_parameters
        # these are the parameters we want to use with population based training
        self.actor_learning_rate = self.hyper_parameters['actor_learning_rate']
        self.critic_learning_rate = self.hyper_parameters['critic_learning_rate']
        self.discount_factor = self.hyper_parameters['discount_factor']
        # We're going to use one network for all of our minions
        self.actor_network = ActorNetwork(observation_dims=4, output_dims=2, name=f'Agent {self.id} Actor')
        self.critic_network = CriticNetwork(observation_dims=4, output_dims=1, name=f'Agent {self.id} Critic')
        self.actor_network.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.actor_learning_rate))
        self.critic_network.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.critic_learning_rate))
        # Since Actor-Critic is an on-policy method, we will not use a replay buffer
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.losses = []
        self.scores = []

    def save_models(self):
        # print('... saving models ...')
        self.actor_network.save_weights(self.actor_network.checkpoint_file)
        self.critic_network.save_weights(self.critic_network.checkpoint_file)

    def load_models(self):
        # print('... loading models ...')
        self.actor_network.load_weights(self.actor_network.checkpoint_file)
        self.critic_network.load_weights(self.critic_network.checkpoint_file)

    def choose_action(self, state):
        action_logits = self.actor_network(tf.convert_to_tensor([state]))
        action_probabilities = tf.nn.softmax(action_logits)
        action_distribution = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        action = action_distribution.sample()

        return int(action.numpy()[0])

    def play(self):
        done = False
        while not done:
            action_to_take = self.choose_action(self.state)
            next_state, reward, done, _ = self.environment.step(action_to_take)
            # self.environment.render()
            self.cum_sum += reward
            self.learn(self.state, action_to_take, reward, next_state, done)
            self.state = next_state

        if done:
            self.state = self.environment.reset()
            self.episode_num += 1
            self.episode_rewards.append(self.cum_sum)
            self.scores.append(self.cum_sum)

            f = open(f'{self.environment_name}-{self.log_file_name}.csv', 'a')
            f.write(f'{self.id},'
                    f'{self.episode_num},'
                    f'{self.cum_sum},'
                    f'{self.actor_network.optimizer.learning_rate.numpy()},'
                    f'{self.critic_network.optimizer.learning_rate.numpy()}\n')
            f.close()

            if self.episode_num % 50 == 0:
                print(self.id, ' -> ', self.episode_num, ' -> ', np.mean(self.episode_rewards))
                self.episode_rewards.clear()

            self.cum_sum = 0

    def learn(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Start calculating the Actor and Critic losses for each minion's experience
            action_logits = self.actor_network(tf.convert_to_tensor([state]))
            state_values = self.critic_network(tf.convert_to_tensor([state]))
            next_state_values = self.critic_network(tf.convert_to_tensor([next_state]))
            action_probabilities = tf.nn.softmax(action_logits)
            # We'll be using an advantage function
            action_distributions = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
            log_probs = action_distributions.log_prob(action)
            advantage = reward + self.discount_factor * next_state_values * (1 - int(done)) - state_values
            entropy = -1 * tf.math.reduce_sum(action_probabilities * tf.math.log(action_probabilities))
            actor_loss = -log_probs * advantage - self.hyper_parameters['entropy_coefficient'] * entropy
            critic_loss = advantage ** 2

            # Optimize master's network with the mean of all the losses
            actor_grads = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
            critic_grads = tape2.gradient(critic_loss, self.critic_network.trainable_variables)
            self.actor_network.optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))
            self.critic_network.optimizer.apply_gradients(zip(critic_grads, self.critic_network.trainable_variables))
            self.losses.append(actor_loss.numpy())


# agent1 = Agent('CartPole-v0', {'actor_learning_rate': 0.0001,
#                                'critic_learning_rate': 0.0005,
#                                'entropy_coefficient': 0,
#                                'critic_coefficient': 0.1,
#                                'discount_factor': 0.99, 'unroll_length': 5,
#                                'minions_num': 1},
#                0,
#                'Non-PBT')
# while agent1.episode_num < 850:
#     agent1.play()
