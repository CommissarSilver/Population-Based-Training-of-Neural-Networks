import tensorflow as tf
import tensorflow_probability as tfp
from network import ActorCriticNetwork
from minion import Minion
import threading
import numpy as np


class Agent:
    def __init__(self, observation_dims, output_dims, initial_hyper_parameters, agent_id, coordinator_id):
        self.id = agent_id
        self.hyper_parameters = initial_hyper_parameters
        # these are the parameters we want to use with population based training
        self.learning_rate = self.hyper_parameters['learning_rate']
        self.discount_factor = self.hyper_parameters['discount_factor']
        self.unroll_length = self.hyper_parameters['unroll_length']
        # Just using the minions to gather experience from different instances of environment
        # self.minions = [Minion(self, environment, i) for i in range(initial_hyper_parameters['minions_num'])]
        # We're going to use one network for all of our minions
        self.network = ActorCriticNetwork(observation_dims=observation_dims, output_dims=output_dims,
                                          name='Coordinator {} - Agent {}'.format(coordinator_id, agent_id))
        self.network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        # Since Actor-Critic is an on-policy method, we will not use a replay buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.discounted_rewards = []
        self.losses = []

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        if len(rewards) == 0:
            print('we have a problem')

        for t in range(len(rewards)):
            discounted_sum = 0
            discount = 1

            for k in range(t, len(rewards)):
                discounted_sum += rewards[k] * discount
                discount *= self.discount_factor
            discounted_rewards[t] = discounted_sum
        # Normalize discounter rewards
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards) if np.std(discounted_rewards) > 0 else 1
        discounted_rewards = (discounted_rewards - mean) / std

        return discounted_rewards

    def choose_action(self, state):
        action_logits = self.network(tf.convert_to_tensor([state]))[1]
        action_probabilities = tf.nn.softmax(action_logits)
        action_distribution = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        action = action_distribution.sample()
        self.actions.append(int(action.numpy()[0]))

        return int(action.numpy()[0])

    def save_models(self):
        # print('... saving models ...')
        self.network.save_weights(self.network.checkpoint_file)

    def load_models(self):
        # print('... loading models ...')
        self.network.load_weights(self.network.checkpoint_file)

    def learn(self):
        # x = self.rewards.pop()
        discounted_rewards = self.discount_rewards(self.rewards)

        with tf.GradientTape() as tape:
            for exp_num in range(len(self.states)):
                # Start calculating the Actor and Critic losses for each minion's experience
                state_values, action_logits = self.network(tf.convert_to_tensor([self.states[exp_num]]))
                action_probabilities = tf.nn.softmax(action_logits)
                # We'll be using an advantage function
                advantage = discounted_rewards[exp_num] - state_values
                action_distributions = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
                log_probs = action_distributions.log_prob(self.actions[exp_num])
                entropy = -1 * tf.math.reduce_sum(action_probabilities * tf.math.log(action_probabilities))
                actor_loss = tf.math.reduce_sum(-1 * discounted_rewards[exp_num] * log_probs) - 0.0001 * entropy
                critic_loss = tf.math.reduce_sum(advantage ** 2)
                total_loss = actor_loss + 0.4 * critic_loss

                self.losses.append(total_loss)
            # Optimize master's network with the mean of all the losses
            entire_loss = tf.reduce_mean(self.losses)
            grads = tape.gradient(entire_loss, self.network.trainable_variables)
            self.network.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        self.states.clear()
        self.actions.clear()
        self.losses.clear()
        self.rewards.clear()
