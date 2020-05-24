import tensorflow as tf
import numpy as np
import environment
import model
import stack_frames
import threading
from collections import deque
import copy
from matplotlib import pyplot as plt


class Master:
    def __init__(self, input_shape, possible_actions, hyper_params):
        self.model = model.build_model(input_shape, len(possible_actions))
        self.learning_rate = hyper_params['learning_rate']
        self.discount_factor = hyper_params['discount_factor']
        self.minions = [Minion(self) for _ in range(hyper_params['minions_num'])]

    def learn(self, epochs):
        total_rewards = []
        for epoch in range(epochs):
            print(epoch)
            epoch_rewards = []
            minion_threads = [threading.Thread(target=minion.play, args=(epoch_rewards,)) for minion in self.minions]
            for minion_thread in minion_threads:
                minion_thread.start()
            for minion_thread in minion_threads:
                minion_thread.join()
            total_rewards.append(np.average(epoch_rewards))
        return total_rewards


class Minion:
    def __init__(self, master: Master):
        self.model = master.model
        self.environment, self.possible_actions = environment.create_environment()
        self.actions = [0, 1, 2]
        self.gamma = master.discount_factor
        self.optimizer = tf.keras.optimizers.RMSprop(1e-4)

    def play(self, epoch_rewards):
        variables = self.model.trainable_variables
        self.environment.new_episode()
        game_terminated = False
        state = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
        state = stack_frames.stack_frames(state, self.environment.get_state().screen_buffer, True)
        episode_rewards = []
        step_num = 0
        while not  self.environment.is_episode_finished():
            with tf.GradientTape() as tape:

                state_value, actions_probabilities = self.model(np.asarray(state).reshape(1, 84, 84, 4))
                action_distribution = actions_probabilities[0].numpy() / np.sum(actions_probabilities[0].numpy())
                # print(action_distribution)
                action_to_take = self.possible_actions[np.random.choice(self.actions, p=action_distribution)]
                reward = self.environment.make_action(action_to_take, 4)
                episode_rewards.append(reward)
                if not self.environment.is_episode_finished():
                    next_state = stack_frames.stack_frames(copy.deepcopy(state),
                                                           self.environment.get_state().screen_buffer,
                                                           False)
                    state = copy.deepcopy(next_state)
                else:
                    next_state = np.zeros((84, 84, 4))
                    game_terminated = True

                next_state_value, next_action_probabilities = self.model(np.asarray(next_state).reshape(1, 84, 84, 4))
                responsible_action = tf.reduce_sum(actions_probabilities * action_to_take)

                advantage = reward + (self.gamma * next_state_value - state_value)
                entropy = -tf.reduce_sum(actions_probabilities * tf.math.log(actions_probabilities))
                critic_loss = 0.5 * advantage ** 2
                actor_loss = -tf.reduce_sum(tf.math.log(responsible_action) * advantage)

                loss = 0.5 * critic_loss + actor_loss - entropy * 0.01
                # print(loss)

            gradients, global_norm = tf.clip_by_global_norm(tape.gradient(loss, variables), 40.0)
            self.optimizer.apply_gradients(zip(gradients, variables))
        epoch_rewards.append(np.average(episode_rewards))


master_agent = Master((84, 84, 4), [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      {'learning_rate': 0.0001, 'discount_factor': 0.95, 'minions_num': 5})

total_rewards = master_agent.learn(200)
master_model = master_agent.model
master_model.save('j200.h5')
plt.plot([i for i in range(len(total_rewards))], total_rewards)
plt.show()
print('hi')
