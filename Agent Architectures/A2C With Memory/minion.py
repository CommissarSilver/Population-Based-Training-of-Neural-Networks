import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Minion:
    def __init__(self, master, environment_name, minion_id):
        self.id = minion_id
        self.master = master
        self.environment = gym.make('{}'.format(environment_name))
        self.episode_finished = False

        self.state = self.environment.reset()
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = []
        self.cum_sum = 0
        self.episode_num = 1

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        if len(rewards) == 0:
            print('we have a problem')

        for t in range(len(rewards)):
            discounted_sum = 0
            discount = 1

            for k in range(t, len(rewards)):
                discounted_sum += rewards[k] * discount
                discount *= self.master.discount_factor
            discounted_rewards[t] = discounted_sum

        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards) if np.std(discounted_rewards) > 0 else 1
        discounted_rewards = (discounted_rewards - mean) / std

        return discounted_rewards

    def choose_action(self, state):
        action_logits = self.master.network(np.asarray([state]))[1]
        action_probabilities = tf.nn.softmax(action_logits)
        action_distribution = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        action = action_distribution.sample()

        return int(action.numpy()[0])

    def play(self, mp_queue):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        step = 0
        while (step < self.master.unroll_length and self.episode_finished == False):
            action_to_take = self.choose_action(self.state)
            next_state, reward, self.episode_finished, _ = self.environment.step(action_to_take)
            # self.environment.render()
            self.states.append(self.state)
            self.actions.append(action_to_take)
            self.rewards.append(reward)
            self.cum_sum += reward
            self.state = next_state
            step += 1

        if self.episode_finished:
            self.state = self.environment.reset()
            self.episode_rewards.append(self.cum_sum)
            self.episode_num += 1

            if self.episode_num % 20 == 0:
                f = open('LunarLander - Single.csv', 'a')
                f.write('{},{},{}\n'.format(self.id, self.episode_num, np.mean(self.episode_rewards[-20:])))
                f.close()

            if self.episode_num % 100 == 0:
                print(self.episode_num, ' -> ', self.id, ' -> ', np.mean(self.episode_rewards))
                self.episode_rewards.clear()

            self.cum_sum = 0
            self.episode_finished = False

        mp_queue.append(
            {'states': self.states, 'actions': self.actions, 'discounted_rewards': self.discount_rewards(self.rewards)})
