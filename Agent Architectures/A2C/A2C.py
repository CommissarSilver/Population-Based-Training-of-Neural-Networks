import tensorflow as tf
import numpy as np
import environment
import model
import stack_frames
import threading


class MasterAgent:
    def __init__(self, input_shape, possible_actions, hyper_params, optimizer):

        self.agent_model = model.A2C(input_shape, possible_actions, hyper_params, optimizer)
        self.model = self.agent_model.model  # the agent's neural network
        self.discount_factor = hyper_params['discount_factor']
        self.number_of_slaves = hyper_params['slaves_num']
        self.experiences = []  # s list containing each master agents' slaves experiences
        self.slaves = [Agent(self) for i in range(self.number_of_slaves)]

    def gather_experience(self):

        for agent in self.slaves:
            agent.model.set_weights(self.model.get_weights())  # copy master agent's network weights to all the slave
            # agents
        slave_threads = []

        for slave in self.slaves:
            t = threading.Thread(target=slave.play, args=(1,))
            slave_threads.append(t)
            t.start()  # start salve agents

        for slave_thread in slave_threads:
            slave_thread.join()  # wait until all salve agents are done

        for slave in self.slaves:
            self.experiences.append(slave.episode_info)  # store slave agent's experience for training

    def train(self):
        self.gather_experience()

        for episode in self.experiences:
            returns = np.zeros((len(episode['rewards']), 1))
            ret = 0
            for r in reversed(range(len(episode['rewards']))):  # calculate discounted rewards for the entire episode
                ret = episode['rewards'][r] + self.discount_factor * ret
                returns[r] = ret
            returns = returns / np.max(returns)

            with tf.GradientTape() as tape:
                state_values, action_logprobs = self.model(np.asarray(episode['states']))
                actor_loss = -1 * action_logprobs * (state_values - returns)
                critic_loss = (state_values - returns) ** 2
                loss = tf.reduce_sum(actor_loss) + tf.multiply(0.1, tf.reduce_sum(critic_loss))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.agent_model.optimizer.apply_gradients(zip(gradients, variables))

            for agent in self.slaves:  # empty slaves agents' memory
                agent.episode_info = {'states': [], 'rewards': []}

    def test(self):
        # test the agent to see how well it performs
        for agent in self.slaves:
            agent.model.set_weights(self.model.get_weights())
        slave_threads = []

        for slave in self.slaves:
            t = threading.Thread(target=slave.play, args=(100,))
            slave_threads.append(t)
            t.start()


class Agent:
    def __init__(self, master_agent: MasterAgent):
        self.model = tf.keras.models.clone_model(master_agent.model)  # clone MasterAgent's network architecture
        self.model.set_weights(master_agent.model.get_weights())
        self.environment, self.possible_actions = environment.create_environment()
        self.episode_info = {'states': [], 'rewards': []}
        self.frame_skip = 4

    def step(self, state):
        state_value, action_logits = self.model.predict(state)
        actions_distribution = tf.nn.softmax(action_logits).numpy()[0]  # get a softmax distribution over action logits
        actions = [0, 1, 2]
        action_to_take = self.possible_actions[np.random.choice(actions, p=actions_distribution)]  # select an action
        # from the possible actions according to the action distribution
        reward = self.environment.make_action(action_to_take, self.frame_skip)

        # store what happened in agent's memory
        self.episode_info['states'].append(state.reshape(84, 84, 4))
        self.episode_info['rewards'].append(reward)

    def play(self, turns):
        for turn in range(turns):
            self.environment.new_episode()
            game_start = True
            if game_start:
                state = np.zeros((84, 84, 4))
                state = stack_frames.stack_frames(state, self.environment.get_state().screen_buffer, True)
                self.step(state.reshape(1, 84, 84, 4))
                game_start = False
            while not self.environment.is_episode_finished():
                state = stack_frames.stack_frames(state, self.environment.get_state().screen_buffer, False)
                self.step(state.reshape(1, 84, 84, 4))


def main(train_or_test):
    agent1 = MasterAgent((84, 84, 4), [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         {'learning_rate': 0.001, 'discount_factor': 0.95, 'slaves_num': 10},
                         tf.keras.optimizers.Adam(1e-4))
    agent1.model.load_weights('Master-50.h5')
    print('loading model')

    if train_or_test == 'train':
        for i in range(10):
            agent1.train()

        agent1.model.save('Master-50.h5')
        print('model saved')
    else:
        agent1.test()