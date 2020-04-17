import numpy as np
import AgentModel
import tensorflow as tf
import experience_replay



# noinspection PyShadowingNames
class DQN:
    # Here we create our agent's DQN
    def __init__(self, state_shape, h_parameters, pretrain_length, stack_size, possible_actions):
        # Arguments:
        #   state_shape: the shape of each input which we're going to feed our agent
        #   num_actions: number of actions that our agent is capable of doing in the environment
        #   gamma: our Q-Learning discount factor
        #   batch_size: batch size
        #   max_buffer_size: the maximum limit of experiences which we can store in our replay buffer
        #   learning_rate: learning rate
        #   pretrain_length: number of instances to put inside our replay buffer before beginning training the agent
        #   stack_size: how many frames to stack together to create a single state for feeding the agent
        # Implements:
        #   Creates an instance of our agent
        self.possible_actions = possible_actions
        self.state_shape = state_shape
        self.num_actions = len(possible_actions)
        self.learning_rate = h_parameters['learning_rate']
        self.optimizer = tf.optimizers.RMSprop(
            learning_rate=self.learning_rate)  # We're gonna use RMSProp as out optimizer
        self.gamma = h_parameters['gamma']
        self.model = AgentModel.AgentModel(self.state_shape, self.num_actions)  # Instantiate an agent
        self.memory = experience_replay.create_and_fill_memory(stack_size,
                                                               pretrain_length)  # Fill agent's replay buffer before training
        self.batch_size = h_parameters['batch_size']
        self.max_buffer_size = h_parameters['max_buffer_size']
        self.losses = []

    def predict(self, inputs):
        # Arguments:
        #   inputs: a single or a batch of states to determine the next action from
        # Returns:
        #   agent network's output. Q values for each possible action
        # Implements:
        #   A single forward pass through agent's network with the given inputs
        return self.model(inputs)

    def get_batch(self):
        # Arguments:
        #   -
        # Returns:
        #   -
        # Implements:
        #   Training the agent on the experiences in its replay buffer

        # Sampling random experiences from agent's replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # predict the next state's Q-values
        next_values = self.predict(next_states)
        # Get the indexes of the action with hte highest Q-value
        indexes = tf.math.argmax(next_values, axis=1)
        # Create an empty list. This is going to contain our target-Qs values
        targetQs = []

        # If we're in a terminal state, return the reward, if not, return rewards+gamma*(next action's Q-value)
        for i, done in enumerate(dones):
            if done:
                targetQs.append(rewards[i])
            else:
                targetQs.append(rewards[i] + self.gamma * next_values[i][indexes[i]])
        return states, actions, targetQs

    def train(self):
        states, actions, targetQs = self.get_batch()
        # start keeping tab on agent network's variables for backpropagation
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(states) * actions, axis=1)
            loss = tf.math.reduce_sum(tf.math.square(tf.math.subtract(targetQs, selected_action_values)))

        self.losses.append(loss)
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        # Arguments:
        #   states: states which we want to predict corresponding actions for
        #   epsilon: exploration rate
        # Returns:
        #   action: the action to be taken
        # Implements:
        #   Passes the states through the agent's network and calculates the action that must be taken

        if np.random.random() < epsilon:
            # Explore
            return self.possible_actions[np.random.choice(self.num_actions)]
        else:
            # Exploit
            action = self.predict(states)
            action_index = tf.math.argmax(action, axis=1).numpy()[0]
            return self.possible_actions[action_index]

    def add_experience(self, experience):
        # Arguments:
        #   experience: an experience instance. it contain the current state, action taken, reward received, next state and whether we have reached the terminal state or not
        # Returns:
        # -
        # Implements:
        #   filling the agent's replay buffer with new experiences

        # if the replay buffer is at it's limit delete the oldest instance and append the latest experience
        if self.max_buffer_size > self.memory.buffer_size:
            for key in self.memory.buffer.keys():
                self.memory.buffer[key].pop(0)
            self.memory.add(experience[0], experience[1], experience[2], experience[3],
                            experience[4])




# def step(game, agent, epsilon):
#     play_game(game, agent, epsilon)
#     agent.train()
