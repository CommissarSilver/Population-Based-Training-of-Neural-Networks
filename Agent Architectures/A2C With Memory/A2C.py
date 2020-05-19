import tensorflow as tf
import numpy as np
import environment
import model
import stack_frames
import threading
from collections import deque


class Agent:
    def __init__(self, input_shape, possible_actions, hyper_params):
        self.model = model.build_model(input_shape, len(possible_actions))  # the agent's neural network
        self.discount_factor = hyper_params['discount_factor']
        self.learning_rate = hyper_params['learning_rate']
        self.memory_size = hyper_params['memory_size']
        self.train_steps = hyper_params['train_steps']
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
        self.minions = [Minion(self) for i in range(hyper_params['minions_num'])]

    def gather_experience(self):
        self.experiences = {'episode_{0}'.format(i): {'states': [], 'rewards': [], 'values': [], 'action_logprobs': []}
                            for i in range(self.memory_size)}
        print(' -Gathering Experience-')
        episode_num = 0
        for i in range(self.memory_size):
            print('     -Playing Episode {0}'.format(episode_num + 1))

            episode_num += 1

    def train(self):
        minion_threads = [threading.Thread(target=minion.play) for minion in self.minions]
        for minion_thread in minion_threads:
            minion_thread.start()
        for minion_thread in minion_threads:
            minion_thread.join()
        experiences = [minion.memory for minion in self.minions]
        losses=[]
        print(' Training')
        for experience in experiences:
            # steps_num = len(self.experiences[episode_id]['rewards']) % self.step_size

            returns = np.zeros((len(experience['rewards']), 1))
            ret = 0
            for r in reversed(range(len(experience['rewards']))):  # calculate discounted rewards for the entire episode
                ret = experience['rewards'][r] + self.discount_factor * ret
                returns[r] = ret
            returns = tf.linalg.normalize(returns)[0].numpy()

            with tf.GradientTape() as tape:
                state_values, action_logprobs = self.model(np.asarray(experience['states']))
                actor_loss = -1 * tf.multiply(action_logprobs,
                                              tf.math.subtract(tf.stop_gradient(state_values), returns))
                critic_loss = tf.math.pow(tf.math.subtract(state_values, returns), 2)
                loss = tf.reduce_sum(actor_loss) + tf.multiply(0.1, tf.reduce_sum(critic_loss))
                losses.append(loss)
        variables = self.model.trainable_variables
        gradients = tape.gradient(losses, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        for minion in self.minions:
            minion.update(self.model)
        return np.average(returns)

    def test(self):
        # test the agent to see how well it performs
        for minion in self.minions:
            minion.model.set_weights(self.model.get_weights())
        minion_threads = []

        for minion in self.minions:
            thread = threading.Thread(target=minion.play)
            minion_threads.append(thread)
            thread.start()


class Minion:
    def __init__(self, master_agent: Agent):
        self.model = tf.keras.models.clone_model(master_agent.model)
        self.environment, self.possible_actions = environment.create_environment()
        self.frame_skip = 4
        self.memory = {'states': [], 'rewards': []}

    def step(self, state):
        state_reshaped = np.asarray(state).T.reshape(1, 84, 84, 4)
        state_value, action_logprobs = self.model.predict(state_reshaped)
        actions_distribution = action_logprobs[0]  # get a softmax distribution over action logits
        actions = [0, 1, 2]
        action_to_take = self.possible_actions[np.random.choice(actions, p=actions_distribution)]  # select an action
        # from the possible actions according to the action distribution
        reward = self.environment.make_action(action_to_take, self.frame_skip)

        # store what happened in agent's memory
        self.memory['states'].append(np.asarray(state).T)
        self.memory['rewards'].append(reward)

    def play(self):
        self.memory = {'states': [], 'rewards': []}
        self.environment.new_episode()
        game_start = True

        if game_start:
            state = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
            state = stack_frames.stack_frames(state, self.environment.get_state().screen_buffer, True)
            self.step(state)
            game_start = False
        while not self.environment.is_episode_finished():
            state = stack_frames.stack_frames(state, self.environment.get_state().screen_buffer, False)
            self.step(state)

    def update(self, master_model):
        self.model.set_weights(master_model.get_weights())


def main(train_or_test):
    master_agent = Agent((84, 84, 4), [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         {'learning_rate': 0.9, 'discount_factor': 0.95, 'memory_size': 10, 'train_steps': 10,
                          'minions_num': 5})
    # master_agent.model.load_weights('Master-10.h5')
    print('     loading model')

    for i in range(100):
        print('Round: ', i)
        master_agent.train()
    # #
    master_agent.model.save('Master-10.h5')
    print('     model saved')


# main('train')
