import tensorflow as tf
import os
import tensorflow_probability as tfp
import threading
import random
import numpy as np
import gym
import copy
from tensorflow.keras import backend as K

if not os.path.exists(f'{os.getcwd()}//Agent Models'):
    os.makedirs(f'{os.getcwd()}//Agent Models')
    print(os.getcwd())

class ActorNetwork(tf.keras.Model):
    def __init__(self, output_dims,id):
        super(ActorNetwork, self).__init__()
        self.output_dims = output_dims
        # Create a checkpoint directory in case we want to save our model
        name = 'Actor'
        self.model_name = name +f' {id}'

        checkpoint_directory = f'{os.getcwd()}//Agent Models'
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=2048, activation='relu', name='Dense_Layer_1',
                                                   dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2',
                                                   dtype=tf.float32)
        self.action_probs = tf.keras.layers.Dense(units=self.output_dims, activation=None, name='Action_Logits',
                                                  dtype=tf.float32)

    def call(self, state):
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)

        action_probs = self.action_probs(x)
        return action_probs


class CriticNetwork(tf.keras.Model):
    def __init__(self, output_dims, id):
        super(CriticNetwork, self).__init__()
        self.output_dims = output_dims
        # Create a checkpoint directory in case we want to save our model
        name = 'Critic'
        self.model_name = name + f' {id}'

        checkpoint_directory = f'{os.getcwd()}//Agent Models'
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=2048, activation='relu', name='Dense_Layer_1',
                                                   dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2',
                                                   dtype=tf.float32)
        self.state_value = tf.keras.layers.Dense(units=1, activation=None, name='State_Value',
                                                 dtype=tf.float32)

    def call(self, state):
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)

        state_value = self.state_value(x)
        return state_value
class Agent:
    def __init__(self, environment, initial_hyper_parameters, id, log_file_name):
        # Agent's parameters needed for logging
        self.id = id
        self.log_file_name = log_file_name
        self.cum_sum = 0
        self.episode_num = 0

        # Agent's initial hyper-parameters
        self.hyper_parameters = initial_hyper_parameters

        # Agent's environment creation and initialization
        self.environment = gym.make('{}'.format(environment))
        self.episode_finished = False
        self.state = self.environment.reset()
        self.environment_name = environment

        # These are the parameters we want to use with population based training
        self.actor_learning_rate = self.hyper_parameters['actor_learning_rate']
        self.critic_learning_rate = self.hyper_parameters['critic_learning_rate']

        # We're going to use one network for all of our minions
        self.actor_network = ActorNetwork(output_dims=self.environment.action_space.n, id=self.id)
        self.critic_network = CriticNetwork(output_dims=1, id=self.id)

        self.actor_network.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.actor_learning_rate))
        self.critic_network.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.critic_learning_rate))

        # Since Actor-Critic is an on-policy method, we will not use a replay buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = []
        self.scores = []
        self.losses = []

    def save_models(self):
        # print('... saving models ...')
        self.actor_network.save_weights(self.actor_network.checkpoint_file)
        self.critic_network.save_weights(self.critic_network.checkpoint_file)

    def load_models(self):
        # print('... loading models ...')
        self.actor_network.load_weights(self.actor_network.checkpoint_file)
        self.critic_network.load_weights(self.critic_network.checkpoint_file)

    def choose_action(self, state):
        action_logits = self.actor_network(tf.convert_to_tensor([state]))
        action_probabilities = tf.nn.softmax(action_logits)
        action_distribution = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        action = action_distribution.sample()

        return int(action.numpy()[0])

    def play(self):
        done = False
        steps = 1

        while not done:
            action_to_take = self.choose_action(self.state)
            next_state, reward, done, _ = self.environment.step(action_to_take)
            self.cum_sum += reward
            self.states.append(self.state)
            self.actions.append(action_to_take)
            self.rewards.append(reward)
            self.state = next_state
            steps += 1

            if steps % self.hyper_parameters['unroll_length'] == 0 and not done:
                self.learn(self.states, self.rewards, self.actions)
                self.states.clear()
                self.actions.clear()
                self.rewards.clear()

            if done:
                self.state = self.environment.reset()
                self.episode_num += 1
                self.episode_rewards.append(self.cum_sum)
                self.scores.append(self.cum_sum)

                f = open(f'{self.environment_name}-{self.log_file_name}.csv', 'a')
                f.write(f'{self.id},'
                        f'{self.episode_num},'
                        f'{self.cum_sum},'
                        f'{self.actor_network.optimizer.learning_rate.numpy()},'
                        f'{self.critic_network.optimizer.learning_rate.numpy()},'
                        f'{self.hyper_parameters["discount_factor"]},'
                        f'{self.hyper_parameters["unroll_length"]}\n')
                f.close()

                if self.episode_num % 50 == 0:
                    print(self.id, ' -> ', self.episode_num, ' -> ', np.mean(self.episode_rewards))
                    self.episode_rewards.clear()

                self.learn(self.states, self.rewards, self.actions)
                self.states.clear()
                self.actions.clear()
                self.rewards.clear()
                self.cum_sum = 0

    def learn(self, states, rewards, actions):

        discounted_rewards = []
        sum_reward = 0
        rewards.reverse()
        for r in rewards:
            sum_reward = r + agent.hyper_parameters['discount_factor'] * sum_reward
            discounted_rewards.append(sum_reward)
        discounted_rewards.reverse()

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # Start calculating the Actor and Critic losses for each minion's experience
            action_logits = self.actor_network(tf.convert_to_tensor(states))
            state_values = self.critic_network(tf.convert_to_tensor(states))
            action_probabilities = tf.nn.softmax(action_logits)
            # We'll be using an advantage function
            action_distributions = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
            log_probs = action_distributions.log_prob(actions)
            advantage = tf.math.subtract(discounted_rewards, state_values)
            # entropy = -1 * tf.math.reduce_sum(action_probabilities * tf.math.log(action_probabilities))
            actor_loss = tf.math.reduce_sum(-1 * log_probs * advantage)
            critic_loss = tf.math.reduce_sum(advantage ** 2)

            # Optimize master's network with the mean of all the losses
        actor_grads = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic_network.trainable_variables)
        self.actor_network.optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))
        self.critic_network.optimizer.apply_gradients(zip(critic_grads, self.critic_network.trainable_variables))
        self.losses.append(actor_loss.numpy())
def exploit(population):
    sorted_population = sorted(population, key=lambda i: np.mean(i.scores), reverse=True)
    best_agents = sorted_population[:3]
    worst_agents = sorted_population[-3:]

    # for agent in worst_agents:
        # print(f'Agent -> {agent.id} is gonna go through exploitation and exploration')

    # agent_to_exploit = best_agents[:3]
    # print(f'Agent -> {agent_to_exploit.id} is selected for reproduction')

    # for each other agent, load their models here
    for agent in worst_agents:
        worst_agent_id = agent.id
        worst_agent_episode = agent.episode_num
        new_agent = copy.deepcopy(random.choice(best_agents))
        print(f'Agent -> {new_agent.id} will replace {worst_agent_id}')
        new_agent.id = worst_agent_id
        new_agent.episode_num = worst_agent_episode
        explore(agent)
        # print(f'Agent -> {agent.id} has gone through exploitation and exploration')

    for agent in population:
        agent.scores.clear()


def explore(agent):
    new_actor_learning_rate = round(agent.hyper_parameters['actor_learning_rate'] * random.uniform(0.8, 1.2), 6)
    new_critic_learning_rate = round(agent.hyper_parameters['critic_learning_rate'] * random.uniform(0.8, 1.2), 6)
    new_unroll_length = round(agent.hyper_parameters['unroll_length'] * random.uniform(0.8, 1.2), 0)

    # new_discount_factor = round(best_agent.hyper_parameters['discount_factor'] * random.uniform(0.8, 1.2), 2)
    # if new_discount_factor > 1:
    #     new_discount_factor = 1

    agent.actor_network.optimizer.learning_rate.assign(new_actor_learning_rate)
    agent.critic_network.optimizer.learning_rate.assign(new_critic_learning_rate)
    agent.hyper_parameters['actor_learning_rate'] = new_actor_learning_rate
    agent.hyper_parameters['critic_learning_rate'] = new_critic_learning_rate
    agent.hyper_parameters['unroll_length'] = new_unroll_length
    # agent.hyper_parameters['discount_factor'] = new_discount_factor

environment_name = 'LunarLander-v2'
log_file_name = 'PBT-NSteps-(2048-512)'
agents = []
for i in range(16):
    agents.append(Agent('{}'.format(environment_name),
                        {'actor_learning_rate': round(random.uniform(0.00001,0.005),4),
                         'critic_learning_rate': round(random.uniform(0.00001,0.005),4),
                         'entropy_coefficient': 0.00001,
                         'critic_coefficient': 0.00005,
                         'discount_factor': 0.99,
                         'unroll_length': round(random.uniform(10,500))},
                        i + 1,
                        log_file_name))

f = open(f'{environment_name}-{log_file_name}.csv', 'a')
f.write(
    'Agent ID,Episode Number,Episode Score,Actor Learning Rate,Critic Learning Rate,Discount Factor,Unroll Length\n')
f.close()
threads = []
for j in range(1, 4000):
    for agent in agents:
        try:
            agent.play()
        except:
            new_agent_id = agent.id
            new_agent_episode = agent.episode_num
            agents.remove(agent)
            new_agent = copy.deepcopy(random.choice(sorted(agents, key=lambda i: np.mean(i.scores), reverse=True)[:3]))
            new_agent.id = new_agent_id
            new_agent.episode_num = new_agent_episode
            agents.append(new_agent)
    if j % 200 == 0:
        exploit(agents)
