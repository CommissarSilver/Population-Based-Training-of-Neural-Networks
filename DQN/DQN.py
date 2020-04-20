import tensorflow as tf
import numpy as np
import AgentModel
import stack_frames
import environment
from collections import deque
import random
import copy


# noinspection PyShadowingNames
class DQN:
    # Here we create our agent's DQN
    def __init__(self, state_shape, hyper_parameters, possible_actions):
        self.dqn_agent = AgentModel.Agent(input_shape=state_shape, output_shape=len(possible_actions),
                                          actions=possible_actions, hyper_parameters=hyper_parameters,
                                          target_network=False)
        self.dqn_target = AgentModel.Agent(input_shape=state_shape, output_shape=len(possible_actions),
                                           actions=possible_actions, hyper_parameters=hyper_parameters,
                                           target_network=True)
        self.batch_size = 3000  # hyper_parameters['batch_size']
        self.hyper_parameters = hyper_parameters
        self.possible_actions = possible_actions
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01  # hyper_parameters['epsilon']
        self.episode = 0
        self.gamma = 0.9  # hyperparameter
        self.frames_seen = 0

    def train(self):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = self.dqn_agent.memory.sample(
            self.batch_size)

        batch_next_state_q_values = self.dqn_target.model(batch_next_states)
        batch_next_state_actions = tf.argmax(batch_next_state_q_values, axis=1)
        targets = []

        for i, terminal in enumerate(batch_terminals):
            if terminal:
                targets.append(batch_rewards[i])
            else:
                targets.append(batch_rewards[i] + self.gamma * tf.reduce_sum(
                    batch_next_state_q_values[i] * self.possible_actions[batch_next_state_actions[i]]))

        with tf.GradientTape() as tape:
            batch_state_q_values = self.dqn_agent.model(batch_states)
            batch_action_qs = tf.math.reduce_sum(batch_state_q_values * batch_actions, axis=1)

            loss = tf.keras.losses.mse(targets, batch_action_qs)

        # print(loss)
        model_gradients = tape.gradient(loss, self.dqn_agent.model.trainable_variables)
        self.dqn_agent.optimizer.apply_gradients(zip(model_gradients, self.dqn_agent.model.trainable_variables))

    def get_td_error(self, state, next_state):
        dqn_out = tf.math.reduce_max(self.dqn_agent.model(state.reshape(1, 84, 84, 4)), axis=1)
        target_out = tf.math.reduce_max(self.dqn_target.model(next_state.reshape(1, 84, 84, 4)), axis=1)
        return (target_out.numpy() - dqn_out.numpy()) ** 2

    def get_action(self, state, model, evaluation=False):
        if evaluation:
            action_index = tf.argmax(self.dqn_agent.model(state.reshape(1, 84, 84, 4)), axis=1)
            return self.possible_actions[action_index]
        else:
            if random.random() < self.epsilon:
                action = random.choice(self.possible_actions)
            else:
                if model == 'dqn_agent':
                    action_qs = self.dqn_agent.model(state.reshape(1, 84, 84, 4))
                else:
                    action_qs = self.dqn_target.model(state.reshape(1, 84, 84, 4))
                action_index = tf.argmax(action_qs, axis=1).numpy()
                action = self.possible_actions[action_index[0]]

            if self.epsilon > self.min_epsilon:
                self.epsilon -= 0.0001

            return action

    def update_target_network(self):
        self.dqn_target.model.set_weights(self.dqn_agent.model.get_weights())

    def play(self, game, possible_actions, turns, frame_skip):
        for i in range(turns):
            game.new_episode()
            state = np.zeros((84, 84, 4))
            state = stack_frames.stack_frames(state, game.get_state().screen_buffer, True)
            action = self.get_action(state, 'dqn_agent')
            reward = game.make_action(action)
            for i in range(frame_skip):
                game.make_action(action)
            terminal = game.is_episode_finished()
            if terminal:
                next_state = np.zeros((84, 84, 4))
            else:

                while not game.is_episode_finished():
                    state = stack_frames.stack_frames(state, game.get_state().screen_buffer, True)
                    action = self.get_action(state, 'dqn_agent')
                    reward = game.make_action(action)
                    for i in range(frame_skip):
                        game.make_action(action)
                    terminal = game.is_episode_finished()
                    if terminal:
                        next_state = np.zeros((84, 84, 4))
                    else:
                        next_state = stack_frames.stack_frames(copy.deepcopy(state), game.get_state().screen_buffer,
                                                               False)
                    self.dqn_agent.memory.add(state, action, reward, next_state, terminal,
                                              self.get_td_error(state, next_state))
            self.dqn_agent.memory.add(state, action, reward, next_state, terminal, self.get_td_error(state, next_state))

    def true_play_get_action(self, state):

        action_qs = self.dqn_agent.model(np.asarray(state).reshape(1, 84, 84, 4))
        action_index = tf.argmax(action_qs, axis=1).numpy()
        action = self.possible_actions[action_index[0]]

        return action

    def true_play(self, game, turns):
        for i in range(turns):
            game.new_episode()
            state = np.zeros((84, 84, 4))
            state = stack_frames.stack_frames(state, game.get_state().screen_buffer, True)
            action = self.true_play_get_action(state)
            game.make_action(action)
            while not game.is_episode_finished():
                state = stack_frames.stack_frames(copy.deepcopy(state), game.get_state().screen_buffer, True)
                action = self.true_play_get_action(state)
                game.make_action(action)

    def fill_memory(self, game, pre_train_length, frame_skip):
        for i in range(pre_train_length):
            game.new_episode()
            state = np.zeros((84, 84, 4))
            state = stack_frames.stack_frames(state, game.get_state().screen_buffer, True)
            action = self.get_action(state, 'dqn_agent')
            reward = game.make_action(action)
            for i in range(frame_skip):
                game.make_action(action)
            terminal = game.is_episode_finished()
            if terminal:
                next_state = np.zeros((84, 84, 4))
            else:
                next_state = stack_frames.stack_frames(copy.deepcopy(state), game.get_state().screen_buffer, False)
                while not game.is_episode_finished():
                    state = stack_frames.stack_frames(state, game.get_state().screen_buffer, True)
                    action = self.get_action(state, 'dqn_agent')
                    reward = game.make_action(action)
                    for i in range(frame_skip):
                        game.make_action(action)
                    terminal = game.is_episode_finished()
                    if terminal:
                        next_state = np.zeros((84, 84, 4))
                    else:
                        next_state = stack_frames.stack_frames(copy.deepcopy(state), game.get_state().screen_buffer,
                                                               False)
                    self.dqn_agent.memory.add(state, action, reward, next_state, terminal,
                                              self.get_td_error(state, next_state))
            self.dqn_agent.memory.add(state, action, reward, next_state, terminal, self.get_td_error(state, next_state))


game, possible_actions = environment.create_environment()
dqn = DQN(state_shape=[84, 84, 4], hyper_parameters={'batch_size': 10, 'pretrain_length': 3},
          possible_actions=possible_actions)
dqn.fill_memory(game, 10, 4)
dqn.train()
i = 0
print('Behold! The humble beginnings of SkyNet (⌐■_■) ')
while True:
    i += 1
    print(i)
    dqn.play(game, possible_actions, 10, 4)
    dqn.train()
    print(dqn.epsilon)
    if i % 5 == 0:
        print('testing network')
        # dqn.true_play(game, 10)
        print(len(dqn.dqn_agent.memory.buffer.keys()))
    if i % 20 == 0:
        print('copying weights')
        dqn.update_target_network()

