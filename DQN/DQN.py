import tensorflow as tf
import numpy as np
import AgentModel
import stack_frames
import environment
from collections import deque
import random


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
        self.batch_size = 200  # hyper_parameters['batch_size']
        self.hyper_parameters = hyper_parameters
        self.possible_actions = possible_actions
        self.epsilon = 0.9
        self.max_epsilon = 0.9
        self.min_epsilon = 0.01  # hyper_parameters['epsilon']
        self.episode = 0
        self.gamma=0.9 #hyperparameter

    def train(self):
        batch = self.dqn_agent.memory.sample(self.batch_size)
        for exprerience in batch:
            states, actions, next_states, rewards, terminals = np.asarray(exprerience.state).reshape(1, 84, 84, 4), \
                                                               exprerience.action, np.asarray(exprerience.next_state)\
                                                                .reshape(1, 84, 84, 4),\
                                                               exprerience.reward,\
                                                               exprerience.done

            batch_next_state_q_values = self.dqn_target.model(next_states)
            batch_target_actions_indexes = tf.math.argmax(batch_next_state_q_values, axis=1)
            targets = []

            if not terminals:
                targets.append(rewards)
            else:
                targets.append(rewards + self.gamma * batch_next_state_q_values[batch_target_actions_indexes])

            with tf.GradientTape() as tape:
                batch_state_q_values = self.dqn_agent.model(states)
                batch_action_qs = tf.math.reduce_sum(batch_state_q_values * actions, axis=1)
                loss = tf.keras.losses.mse(targets, batch_action_qs)

            # print(loss)
            model_gradients = tape.gradient(loss, self.dqn_agent.model.trainable_variables)
            self.dqn_agent.optimizer.apply_gradients(zip(model_gradients, self.dqn_agent.model.trainable_variables))

    def get_action(self, state, model):
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            action_qs = model(np.asarray(state).reshape(1, 84, 84, 4))
            action_index = tf.argmax(action_qs, axis=1).numpy()
            action = self.possible_actions[action_index[0]]

        if self.epsilon > self.min_epsilon:
            self.epsilon -= 0.00001

        return action

    def update_target_network(self):
        self.dqn_target.model.set_weights(self.dqn_agent.model.get_weights())

    def play(self, game, possible_actions, turns):
        for i in range(turns):
            game.new_episode()
            frame_skip = 0
            stacked_frames = deque([np.zeros((84, 84, 4), dtype=np.int) for _ in range(4)], maxlen=4)
            state = stack_frames.stack_frames(stacked_frames, game.get_state().screen_buffer, True)
            action = self.get_action(state, self.dqn_agent.model)
            reward = game.make_action(action)
            next_state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
            next_action = self.get_action(next_state, self.dqn_target.model)
            agent_q = tf.math.reduce_sum(
                self.dqn_agent.model(np.asarray(state).reshape(1, 84, 84, 4)) * action, axis=1).numpy()
            target_q = tf.math.reduce_sum(
                self.dqn_target.model(np.asarray(next_state).reshape(1, 84, 84, 4)) * next_action, axis=1).numpy()
            td_error = ((target_q - agent_q) ** 2)[0]
            self.dqn_agent.memory.add(state=state, action=action, reward=reward, next_state=next_state,
                                      done=game.is_episode_finished(),
                                      td_error=td_error)
            while not game.is_episode_finished():
                state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
                action = self.get_action(state, self.dqn_agent.model)
                reward = game.make_action(action)
                terminal = game.is_episode_finished()
                if not terminal:
                    next_state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
                    next_action = self.get_action(next_state, self.dqn_target.model)
                    agent_q = tf.math.reduce_sum(
                        self.dqn_agent.model(np.asarray(state).reshape(1, 84, 84, 4)) * action,
                        axis=1).numpy()
                    target_q = tf.math.reduce_sum(
                        self.dqn_target.model(np.asarray(next_state).reshape(1, 84, 84, 4)) * next_action,
                        axis=1).numpy()
                    td_error = ((target_q - agent_q) ** 2)[0]
                    self.dqn_agent.memory.add(state, action, reward, next_state,
                                              game.is_episode_finished(),
                                              td_error)
                # else:
                #     next_state = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
                #     self.dqn_agent.memory.add(state, action, reward, next_state,
                #                               game.is_episode_finished(),
                #                               0.01)

    def true_play_get_action(self, state, model):

        action_qs = model(np.asarray(state).reshape(1, 84, 84, 4))
        action_index = tf.argmax(action_qs, axis=1).numpy()
        action = self.possible_actions[action_index[0]]

        return action

    def true_play(self, game,turns):
        for i in range(turns):
            game.new_episode()
            stacked_frames = deque([np.zeros((84, 84, 4), dtype=np.int) for _ in range(4)], maxlen=4)
            state = stack_frames.stack_frames(stacked_frames, game.get_state().screen_buffer, True)
            action = self.true_play_get_action(state, self.dqn_agent.model)
            reward = game.make_action(action)
            # next_state = stack_frames.stack_frames(stacked_frames, game.get_state().screen_buffer, False)
            while not game.is_episode_finished():
                state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
                action = self.true_play_get_action(state, self.dqn_agent.model)
                reward = game.make_action(action)

    def fill_memory(self, game, pre_train_length):
        for i in range(pre_train_length):
            game.new_episode()
            frame_skip = 0
            stacked_frames = deque([np.zeros((84, 84, 4), dtype=np.int) for _ in range(4)], maxlen=4)
            state = stack_frames.stack_frames(stacked_frames, game.get_state().screen_buffer, True)
            action = self.get_action(state, self.dqn_agent.model)
            reward = game.make_action(action)
            next_state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
            next_action = self.get_action(next_state, self.dqn_target.model)
            agent_q = tf.math.reduce_sum(
                self.dqn_agent.model(np.asarray(state).reshape(1, 84, 84, 4)) * action, axis=1).numpy()
            target_q = tf.math.reduce_sum(
                self.dqn_target.model(np.asarray(next_state).reshape(1, 84, 84, 4)) * next_action, axis=1).numpy()
            td_error = ((target_q - agent_q) ** 2)[0]
            self.dqn_agent.memory.add(state=state, action=action, reward=reward, next_state=next_state,
                                      done=game.is_episode_finished(),
                                      td_error=td_error)
            while not game.is_episode_finished():
                state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
                action = self.get_action(state, self.dqn_agent.model)
                reward = game.make_action(action)
                terminal = game.is_episode_finished()
                if not terminal:
                    next_state = stack_frames.stack_frames(state, game.get_state().screen_buffer, False)
                    next_action = self.get_action(next_state, self.dqn_target.model)
                    agent_q = tf.math.reduce_sum(
                        self.dqn_agent.model(np.asarray(state).reshape(1, 84, 84, 4)) * action,
                        axis=1).numpy()
                    target_q = tf.math.reduce_sum(
                        self.dqn_target.model(np.asarray(next_state).reshape(1, 84, 84, 4)) * next_action,
                        axis=1).numpy()
                    td_error = ((target_q - agent_q) ** 2)[0]
                    self.dqn_agent.memory.add(state, action, reward, next_state,
                                              game.is_episode_finished(),
                                              td_error)
                # else:
                #     next_state = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
                #     self.dqn_agent.memory.add(state, action, reward, next_state,
                #                               game.is_episode_finished(),
                #                               0.01)


game, possible_actions = environment.create_environment()
dqn = DQN([84, 84, 4], {'batch_size': 10, 'pretrain_length': 3}, possible_actions=possible_actions)
dqn.fill_memory(game, 2)
i = 0
while True:
    i += 1
    print(i)
    dqn.play(game, possible_actions, 10)
    dqn.train()
    print(dqn.epsilon)
    if i % 5 == 0:
        print('testing network')
        dqn.true_play(game,10)
    if i % 25 == 0:
        print('copying weights')
        dqn.update_target_network()
        dqn.epsilon+=0.5
