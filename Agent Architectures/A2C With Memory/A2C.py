import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from model import build_network
from minion import Minion
import threading


class Master:
    def __init__(self, environment, initial_hyper_parameters):
        self.hyper_parameters = initial_hyper_parameters

        self.learning_rate = self.hyper_parameters['learning_rate']
        self.discount_factor = self.hyper_parameters['discount_factor']
        self.unroll_length = self.hyper_parameters['unroll_length']

        self.minions = [Minion(self, environment, i) for i in range(initial_hyper_parameters['minions_num'])]

        self.network = build_network(8, 4)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.states = []
        self.actions = []
        self.discounted_rewards = []
        self.losses = []

    def learn(self):
        # self.network.load_weights('weights.h5')
        print('model loaded')
        while True:
            self.states.clear()
            self.actions.clear()
            self.discounted_rewards.clear()
            processes = []
            queue = []
            for minion in self.minions:
                process = threading.Thread(target=minion.play, args=(queue,))
                processes.append(process)
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            for minion_num in range(len(self.minions)):
                minion_experience = queue.pop()
                self.states.append(minion_experience['states'])
                self.actions.append(minion_experience['actions'])
                self.discounted_rewards.append(minion_experience['discounted_rewards'])

            self.losses.clear()

            with tf.GradientTape() as tape:
                for exp_num in range(len(self.states)):
                    state_values, action_logits = self.network(np.asarray(self.states[exp_num]))
                    action_probabilities = tf.nn.softmax(action_logits)

                    advantage = self.discounted_rewards[exp_num] - state_values
                    action_distributions = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
                    log_probs = action_distributions.log_prob(self.actions[exp_num])
                    entropy = -1 * tf.math.reduce_sum(action_probabilities * tf.math.log(action_probabilities))
                    actor_loss = tf.math.reduce_sum(
                        -1 * self.discounted_rewards[exp_num] * log_probs) - 0.0001 * entropy
                    critic_loss = tf.math.reduce_sum(advantage ** 2)
                    total_loss = actor_loss + 0.1 * critic_loss
                    self.losses.append(total_loss)

                entire_loss = tf.reduce_mean(self.losses)
                grads = tape.gradient(entire_loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))


master = Master('LunarLander-v2',
                {'learning_rate': 0.00005, 'discount_factor': 0.99, 'unroll_length': 50, 'minions_num': 5})
master.learn()
