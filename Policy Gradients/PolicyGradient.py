import tensorflow as tf
import numpy as np
import NeuralModel
from collections import deque
import play_episode


class PGAgent:
    # the agent training with the simplest policy gradient algorithm
    def __init__(self, state_shape, actions, hyperparams, batch_size):
        self.batch_size = batch_size  # number of episodes to train the agent on in each iteration
        self.input_shape = state_shape  # agent's NN input shape
        self.possible_actions = actions  # actions which the agent can take in environment
        self.num_actions = len(self.possible_actions[0])
        self.learning_rate = hyperparams['learning_rate']  # learning rate
        self.discount_rate = hyperparams['discount_rate']  # discount rate
        self.model = NeuralModel.NeuralModel(self.input_shape, self.num_actions)  # agent's NN
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)  # adam optimizer
        self.states = deque(maxlen=batch_size)  # agent's memory for the states it has visited
        self.actions = deque(maxlen=batch_size)  # agent's memory for the action it has taken in each state
        self.discounted_rewards = deque(maxlen=batch_size)  # discounted rewards of the actions taken in states

    def predict_action_probs(self, state, in_training):
        # returns action probabilities if it's not training and logits if otherwise
        return self.model(state, in_training)

    def train(self):
        # train the agent on what it has stored in its memory
        for j in range(self.batch_size):
            states, actions, discounted_rewards = self.states[j], self.actions[j], self.discounted_rewards[j]

            with tf.GradientTape() as tape:
                neg_log_probs = tf.nn.softmax_cross_entropy_with_logits(actions,
                                                                        self.model(np.asarray(states),
                                                                                   in_training=True))
                loss = tf.math.reduce_mean(tf.math.multiply(neg_log_probs, discounted_rewards))

            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))


agent = PGAgent([84, 84, 4], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], {'learning_rate': 0.00001, 'discount_rate': 0.95},
                batch_size=30)
while True:
    for i in range(agent.batch_size):
        temp_states, temp_actions, temp_discounted_rewards = play_episode.play_episode(agent)
        agent.states.append(temp_states)
        agent.actions.append(temp_actions)
        agent.discounted_rewards.append(temp_discounted_rewards)
        print(i)
    agent.train()
    agent.states.clear()
    agent.actions.clear()
    agent.discounted_rewards.clear()
