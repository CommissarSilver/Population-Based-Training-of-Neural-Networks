import tensorflow as tf
import numpy as np
import AgentModel
import stack_frames
import environment
import random
import copy


# noinspection PyShadowingNames
class DQN:
    # Here we create our agent's DQN
    def __init__(self, state_shape, hyper_parameters, possible_actions):
        self.input_shape = state_shape[:2]
        self.stack_size = state_shape[-1]
        self.hyper_parameters = hyper_parameters
        self.possible_actions = possible_actions

        self.batch_size = self.hyper_parameters['batch_size']
        self.epsilon = self.hyper_parameters['epsilon']
        self.epsilon_decay_rate = self.hyper_parameters['epsilon_decay_rate']
        self.min_epsilon = self.hyper_parameters['min_epsilon']
        self.frame_skip = self.hyper_parameters['frame_skip']
        self.gamma = self.hyper_parameters['gamma']
        self.learning_rate = self.hyper_parameters['learning_rate']

        self.dqn_agent = AgentModel.Minion(input_shape=(self.input_shape[0], self.input_shape[1], self.stack_size),
                                           output_shape=len(self.possible_actions),
                                           actions=self.possible_actions,
                                           learning_rate=self.learning_rate,
                                           a=self.hyper_parameters['a'],
                                           memory_capacity=10000,
                                           target_network=False)
        self.dqn_target = AgentModel.Minion(input_shape=(self.input_shape[0], self.input_shape[1], self.stack_size),
                                            output_shape=len(self.possible_actions),
                                            actions=self.possible_actions,
                                            learning_rate=self.learning_rate,
                                            a=self.hyper_parameters['a'],
                                            memory_capacity=10000,
                                            target_network=True)

    def train(self):
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = self.dqn_agent.replay_buffer.sample(
            self.batch_size)

        target_qs = self.dqn_target.model.predict(batch_next_states)
        target_qs_actions = target_qs.argmax(axis=1)
        target_qs = target_qs[range(self.batch_size), target_qs_actions]
        target_qs = batch_rewards + (self.gamma * target_qs * (1 - batch_terminals))

        with tf.GradientTape() as tape:
            q_values = self.dqn_agent.model(batch_states)
            qs = tf.reduce_sum(tf.multiply(q_values, batch_actions), axis=1)
            loss = tf.keras.losses.mse(qs, target_qs)

        gradients = tape.gradient(loss, self.dqn_agent.model.trainable_variables)
        self.dqn_agent.optimizer.apply_gradients(zip(gradients, self.dqn_agent.model.trainable_variables))

    def get_td_error(self, state, next_state, reward, terminal, action):

        dqn_qs = np.sum(self.dqn_agent.model.predict(
            state.reshape(1, self.input_shape[0], self.input_shape[1], self.stack_size)) * action)

        target_qs = self.dqn_target.model.predict(
            next_state.reshape(1, self.input_shape[0], self.input_shape[1], self.stack_size)).max(axis=1)

        targets = reward + (self.gamma * target_qs * (1 - terminal))

        return (dqn_qs - targets) ** 2

    def get_action(self, state, evaluation=False):
        if evaluation:
            action = self.possible_actions[self.dqn_agent.model.predict(state).argmax(axis=1)]
        else:

            if random.random() < self.epsilon:
                action = random.choice(self.possible_actions)
            else:
                action = self.possible_actions[
                    self.dqn_agent.model.predict(
                        state.reshape(1, self.input_shape[0], self.input_shape[1], self.stack_size)).argmax(axis=1)[0]]

            self.epsilon -= self.epsilon_decay_rate

        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

        return action

    def update_target_network(self):
        self.dqn_target.model.set_weights(self.dqn_agent.model.get_weights())

    def single_step(self, game, state):
        frame_skip = self.frame_skip
        action = self.get_action(state)

        for j in range(frame_skip):
            game.make_action(action)

        reward = game.make_action(action)
        terminal = game.is_episode_finished()

        if terminal:
            next_state = np.zeros((self.input_shape[0], self.input_shape[1], self.stack_size))
        else:
            next_state = stack_frames.stack_frames(copy.deepcopy(state), game.get_state().screen_buffer, False)

        error = self.get_td_error(state, next_state, reward, terminal, action)
        self.dqn_agent.replay_buffer.add_experience(state, action, reward, next_state, terminal, error)

        return next_state

    def fill_memory(self, game):
        num_frames = self.dqn_agent.replay_buffer.capacity
        i = 0
        game_start = True
        print(i)
        while i < num_frames:
            if game_start:
                game.new_episode()
                state = stack_frames.stack_frames(np.zeros((self.input_shape[0], self.input_shape[1], self.stack_size)),
                                                  game.get_state().screen_buffer, True)
                state = self.single_step(game, copy.deepcopy(state))
                game_start = False
                i += 1
            while not game.is_episode_finished():
                state = self.single_step(game, copy.deepcopy(state))
                i += 1
            if game.is_episode_finished():
                game_start = True

            self.epsilon = 1

    def play(self, game, number_of_turns=10):
        game_start = True
        i = 0
        while i < number_of_turns:

            if game_start:
                game.new_episode()
                state = stack_frames.stack_frames(np.zeros((self.input_shape[0], self.input_shape[1], self.stack_size)),
                                                  game.get_state().screen_buffer, True)
                state = self.single_step(game, copy.deepcopy(state))
                game_start = False

            while not game.is_episode_finished():
                state = self.single_step(game, copy.deepcopy(state))

            if game.is_episode_finished():
                game_start = True
                i += 1


hyper_parameters = {'epsilon': 0.001,
                    'epsilon_decay_rate': 0,
                    'learning_rate': 0.0001,
                    'batch_size': 30,
                    'pre_train_length': 10000,
                    'gamma': 0.9,
                    'min_epsilon': 0.001,
                    'frame_skip': 4,
                    'a': 0.6}

game, possible_actions = environment.create_environment()

dqn = DQN(state_shape=[84, 84, 4],
          hyper_parameters=hyper_parameters,
          possible_actions=possible_actions)
dqn.dqn_agent.model = tf.keras.models.load_model('dqn-frameskip-4.h5')
dqn.dqn_target.model = tf.keras.models.load_model('dqn-frameskip-4.h5')
dqn.fill_memory(game)
for j in range(100):
    dqn.train()
i = 0
print('Behold! The humble beginnings of SkyNet (⌐■_■) ')
while i < 601:
    i += 1
    print(i)
    dqn.play(game)
    for j in range(100):
        dqn.train()
    print(dqn.epsilon)
    if i % 20 == 0:
        print('copying weights')
        dqn.update_target_network()
    # if i % 25 == 0 and i < 101:
    #     dqn.epsilon += 0.5
    if i % 600 == 0:
        dqn.dqn_agent.model.save('dqn-frameskip-4.h5')
        print('model saved')
