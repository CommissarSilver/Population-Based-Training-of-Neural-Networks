import tensorflow as tf
import numpy as np
import environment
import model
import stack_frames
import threading
from collections import deque
import copy
from matplotlib import pyplot as plt

# needed changes and review:
# unify the hyperparams thingies. there are too many instances that are floating around.
#       is number of minions a hyperparam?????
#       is discount factor a hyperparam?????
#       is number of steps a hyperparam?????


class Master:
    def __init__(self, input_shape, possible_actions, hyper_params):
        self.model = model.build_model(input_shape, len(
            possible_actions))  # Master Agent's neural model
        self.hyper_params = hyper_params
        # we'll tune these in PBT
        self.learning_rate = self.hyper_params['learning_rate']
        # we'll tune these in PBT
        self.discount_factor = self.hyper_params['discount_factor']
        self.minions = [Minion(self, i) for i in range(
            self.hyper_params['minions_num'])]  # static for now
        self.n_steps = self.hyper_params['n_steps']  # we'll tune these in PBT
        self.memory = []  # contains minions experiences. resets after each training epoch
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def train(self, iters_num, master_id):
        rewards_sum = []
        print('Master agent ', master_id, ' starting training')
        for iteration in range(iters_num):
            threads = []

            for minion in self.minions:
                threads.append(threading.Thread(target=minion.play))
            print(' ->Starting Minion threads')
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            print(' ->Minion threads terminated. Training on ',
                  self.n_steps, ' steps')

            for experience in self.memory:
                states, actions, rewards = experience  # get the experiences
                rewards_sum.append(np.sum(rewards))
                with tf.GradientTape() as tape:
                    states_values, actions_probabilities = self.model(
                        np.asarray(states))

                    discounted_sum = 0
                    returns = []
                    for r in rewards:  # calculate discounted rewards
                        # i think this needs some tinkering. is the logic correct?
                        discounted_sum = r + self.discount_factor * discounted_sum
                        returns.append(discounted_sum)
                    returns = np.asarray(returns)

                    returns = (returns - np.mean(returns)) / \
                        (np.std(returns) + 0.001)  # normalize rewards

                    responsible_actions = tf.reduce_sum(
                        actions_probabilities * actions)  # get the actions the were
                    # actually executed

                    advantage = tf.reduce_sum(
                        returns - states_values)  # advantage function
                    # to force the
                    entropy = - \
                        tf.reduce_sum(actions_probabilities *
                                      tf.math.log(actions_probabilities))
                    # the agent to explore more and prevent premature degenerate probabilities
                    critic_loss = advantage ** 2
                    actor_loss = - \
                        tf.reduce_sum(tf.math.log(
                            responsible_actions) * advantage)

                    loss = critic_loss + actor_loss - entropy * 0.01

                gradients, global_norm = tf.clip_by_global_norm(tape.gradient(loss, self.model.trainable_variables),
                                                                40.0)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))  # update the master model

            for minion in self.minions:
                print("->Updating Minion ", minion.minion_id, " network")
                minion.update_network(self.model)  # update the minion model
            self.memory = []

        return np.mean(rewards_sum)


class Minion:
    def __init__(self, master: Master, id):
        self.master = master
        self.model = tf.keras.models.clone_model(master.model)
        self.environment, self.possible_actions = environment.create_environment()
        self.environment.new_episode()
        self.actions = [0, 1, 2]
        self.game_start = True
        self.state = []
        self.minion_id = id

    def step(self):  # takes a single step in the environment after being called. returns state observed,
        # action taken and reward received.
        if self.game_start:
            self.state = deque([np.zeros((84, 84), dtype=np.int)
                                for _ in range(4)], maxlen=4)
            self.state = stack_frames.stack_frames(
                self.state, self.environment.get_state().screen_buffer, True)
            self.game_start = False
        else:
            self.state = stack_frames.stack_frames(
                self.state, self.environment.get_state().screen_buffer, False)

        reshaped_state = np.asarray(self.state).T

        state_value, actions_probabilities = self.model.predict(
            np.expand_dims(reshaped_state, axis=0))

        action_distribution = actions_probabilities[0] / \
            np.sum(actions_probabilities[0])

        action_to_take = self.possible_actions[np.random.choice(
            self.actions, p=action_distribution)]

        reward = self.environment.make_action(action_to_take)

        return reshaped_state, action_to_take, reward

    def play(self):  # plays for the given number of steps
        if self.environment.is_episode_finished():
            print('     ->last session was finished')
            self.environment.new_episode()
        # appended the self.game_start in the if clause. seems it was something which i missed.
            self.game_start = True

        states, actions, rewards = [], [], []
        for step in range(self.master.n_steps):
            if not self.environment.is_episode_finished():
                reshaped_state, action, reward = self.step()
                states.append(reshaped_state)
                actions.append(action)
                rewards.append(reward)

        print('     ->Minion ', self.minion_id,
              ' has played ', len(rewards), ' steps')
        # appends the experience in master's memory
        self.master.memory.append((states, actions, rewards))

    def update_network(self, master_network):
        self.model.set_weights(master_network.get_weights())
        print('     ->Minion ', self.minion_id, ' network updated')

# master_agent = Master((84, 84, 4), [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                       {'learning_rate': 0.00001, 'discount_factor': 0.95, 'minions_num': 5, 'n_steps': 100})
# master_agent.model.load_weights('j200-entire_critic_loss.h5')
# print('Model loaded')
# for j in range(500):
#     print('EPOCH-->', j)
#     master_agent.train()
# master_model = master_agent.model
# master_model.save('j200-entire_critic_loss.h5')
