import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Minion:
    def __init__(self, master, environment_name, minion_id):
        self.id = minion_id
        self.master = master
        self.environment = gym.make('{}'.format(environment_name))
        self.n_actions = 2
        self.max_action = self.environment.action_space.high[0]
        self.min_action = self.environment.action_space.low[0]

        self.episode_finished = False
        self.state = self.environment.reset()

        self.episode_rewards = []
        self.cum_sum = 0
        self.episode_num = 1

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.master.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def play(self, threading_lock):
        with threading_lock:
            step = 0

            if not self.episode_finished:
                action_to_take = self.choose_action(self.state)
                next_state, reward, self.episode_finished, _ = self.environment.step(action_to_take)
                # self.environment.render()
                self.master.remember(self.state, action_to_take, reward, next_state, self.episode_finished)
                self.cum_sum += reward
                self.state = next_state
                step += 1

            if self.episode_finished:
                self.state = self.environment.reset()
                self.episode_rewards.append(self.cum_sum)
                self.master.episode_rewards.append(self.cum_sum)
                self.episode_num += 1
                # print(self.episode_num, ' -> ', self.id, ' -> ', np.mean(self.episode_rewards))

                if self.episode_num % 10 == 0:
                    f = open('LunarLander - Multi.csv', 'a')
                    f.write('{},{},{},{}\n'.format(self.master.id, self.id, self.episode_num,
                                                   np.mean(self.episode_rewards[-10:])))
                    f.close()

                if self.episode_num % 5 == 0:
                    print('Agent: ', self.master.id, ' -> ', 'Minion: ', self.id, ' -> ', 'Episode: ', self.episode_num,
                          ' -> ',
                          'Mean Rewards: ', ' -> ', np.mean(self.episode_rewards))
                    self.episode_rewards.clear()

                self.cum_sum = 0
                self.episode_finished = False

            return
