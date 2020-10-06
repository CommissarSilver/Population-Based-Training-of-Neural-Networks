import gym
import ma_gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from network import ActorCriticNetwork


class Agent:
    def __init__(self, observation_dims, output_dims, initial_hyper_parameters):
        self.hyper_parameters = initial_hyper_parameters
        # these are the parameters we want to use with population based training
        self.learning_rate = self.hyper_parameters['learning_rate']
        self.discount_factor = self.hyper_parameters['discount_factor']
        self.unroll_length = self.hyper_parameters['unroll_length']
        # We're going to use one network for all of our minions
        self.network = ActorCriticNetwork(observation_dims, output_dims)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        # Since Actor-Critic is an on-policy method, we will not use a replay buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.losses = []

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        if len(rewards) == 0:
            print('we have a problem')

        for t in range(len(rewards)):
            discounted_sum = 0
            discount = 1

            for k in range(t, len(rewards)):
                discounted_sum += rewards[k] * discount
                discount *= self.discount_factor
            discounted_rewards[t] = discounted_sum
        # Normalize discounter rewards
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards) if np.std(discounted_rewards) > 0 else 1
        discounted_rewards = (discounted_rewards - mean) / std

        return discounted_rewards

    def choose_action(self, state):
        action_logits = self.network(tf.convert_to_tensor([state]))[1]
        action_probabilities = tf.nn.softmax(action_logits)
        action_distribution = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        action = action_distribution.sample()
        self.actions.append(int(action.numpy()[0]))

        return int(action.numpy()[0])

    def learn(self):
        # x = self.rewards.pop()
        discounted_rewards = self.discount_rewards(self.rewards)

        with tf.GradientTape() as tape:
            for exp_num in range(len(self.states)):
                # Start calculating the Actor and Critic losses for each minion's experience
                state_values, action_logits = self.network(tf.convert_to_tensor([self.states[exp_num]]))
                action_probabilities = tf.nn.softmax(action_logits)
                # We'll be using an advantage function
                advantage = discounted_rewards[exp_num] - state_values
                action_distributions = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
                log_probs = action_distributions.log_prob(self.actions[exp_num])
                entropy = -1 * tf.math.reduce_sum(action_probabilities * tf.math.log(action_probabilities))
                actor_loss = tf.math.reduce_sum(-1 * discounted_rewards[exp_num] * log_probs) - 0.0001 * entropy
                critic_loss = tf.math.reduce_sum(advantage ** 2)
                total_loss = actor_loss + 0.1 * critic_loss

                self.losses.append(total_loss)
            # Optimize master's network with the mean of all the losses
            entire_loss = tf.reduce_mean(self.losses)
            grads = tape.gradient(entire_loss, self.network.trainable_variables)
            self.network.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        self.states.clear()
        self.actions.clear()
        self.losses.clear()
        self.rewards.clear()


class Coordinator:
    def __init__(self, environment_name, initial_hyper_parameters):
        self.environment = gym.make(environment_name)
        self.observation = self.environment.reset()
        self.number_of_agents = len(self.environment.get_action_meanings())
        self.output_dims = len(self.environment.get_action_meanings()[0])
        self.observation_dims = self.environment.observation_space[0].shape[0]
        self.agents = [Agent(self.observation_dims, self.output_dims, initial_hyper_parameters) for i in
                       range(self.number_of_agents)]
        self.episode_finished = False

    def play(self):
        j = 0
        scores=[]
        while True:

            self.observation = self.environment.reset()
            # self.observation, _, _, _ = self.environment.step([4, 4])
            dones = [False for i in range(len(self.agents))]
            self.episode_finished = False
            actions = [4 for i in range(self.number_of_agents)]
            # for i, observation in enumerate(self.observation):
            #     self.agents[i].states.append(observation)
            #     actions.append(self.agents[i].choose_action(observation))

            while not self.episode_finished:
                self.observation, rewards, dones, info = self.environment.step(actions)
                actions = []

                for i, observation in enumerate(self.observation):
                    if not dones[i]:
                        self.agents[i].states.append(observation)
                        actions.append(self.agents[i].choose_action(observation))
                    else:
                        actions.append(4)

                # self.environment.render()

                for i, reward in enumerate(rewards):
                    if not dones[i]:
                        self.agents[i].rewards.append(reward)
                # self.observation = next_observations

                if dones == [True for i in range(len(self.agents))]:
                    self.episode_finished = True
            j += 1
            x = 0
            for agent in self.agents:
                x += np.sum(agent.rewards)
                agent.learn()
            scores.append(x)
            if j % 10 == 0:
                f = open('PredatorPrey5x5-v0 - Multi.csv', 'a')
                f.write('{},{}\n'.format(j, np.mean(scores[-10:])))
                f.close()
            if j % 50 == 0:
                print(j, '   ', np.mean(scores))
                scores.clear()

        # else:
        #     actions = []
        #     for i, observation in self.observation:
        #         self.agents[i].states.append(observation)
        #         actions.append(self.agents[i].choose_action(observation))
        #     next_observations, rewards, done, info = self.environment.step(actions)


env1 = Coordinator('PredatorPrey5x5-v0',
                   {'learning_rate': 0.000025, 'discount_factor': 0.99, 'unroll_length': 50, 'minions_num': 5})
env1.play()
