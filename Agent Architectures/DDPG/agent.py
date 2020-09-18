import tensorflow as tf
import threading
import os

from replay_buffer import ReplayBuffer
from network import ActorNetwork, CriticNetwork
from minion import Minion


class Agent:
    def __init__(self, hyper_parameters, input_dims=8, env_name=None, n_actions=2, max_size=10000, batch_size=512):
        self.hyper_parameters = hyper_parameters
        self.gamma = self.hyper_parameters['gamma']
        self.tau = self.hyper_parameters['tau']

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.minions = [Minion(self, env_name, i) for i in range(self.hyper_parameters['number_of_minions'])]

        self.actor = ActorNetwork(observation_dims=input_dims, output_dims=n_actions, name='actor')
        self.critic = CriticNetwork(observation_dims=input_dims, output_dims=n_actions, name='critic')
        self.target_actor = ActorNetwork(observation_dims=input_dims, output_dims=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(observation_dims=input_dims, output_dims=n_actions, name='target_critic')

        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyper_parameters['actor_learning_rate']))
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyper_parameters['critic_learning_rate']))
        self.target_actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyper_parameters['actor_learning_rate']))
        self.target_critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyper_parameters['critic_learning_rate']))

        self.update_network_parameters(tau=1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def play(self):
        # self.network.load_weights('weights.h5')
        # print('model loaded')
        processes = []
        lock = threading.Lock()
        # Start each minion to gather experience from the environment
        for minion in self.minions:
            process = threading.Thread(target=minion.play, args=(lock,))
            processes.append(process)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        # Collect the gathered experiences

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            self.play()
            return
        self.play()
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            next_critic_value = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * next_critic_value * (1 - dones)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        print('loss ops')
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()


hyper_parameters = {'tau': 0.001, 'gamma': 0.99, 'actor_learning_rate': 0.000025, 'critic_learning_rate': 0.00025,
                    'number_of_minions': 10}

agent = Agent(hyper_parameters, input_dims=8, env_name='LunarLanderContinuous-v2', n_actions=2)
for i in range(1000000):
    agent.learn()
