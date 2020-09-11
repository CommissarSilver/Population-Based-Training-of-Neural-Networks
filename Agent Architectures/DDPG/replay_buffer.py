import numpy as np


class ReplayBuffer:
    def __init__(self, max_buffer_size, observation_shape, number_of_actions):
        self.max_buffer_size = max_buffer_size
        self.observation_shape = observation_shape
        self.number_of_actions = number_of_actions
        self.memory_counter = 0

        self.state_memory = np.zeros((self.max_buffer_size, *self.observation_shape))
        self.next_state_memory = np.zeros((self.max_buffer_size, *self.observation_shape))
        self.action_memory = np.zeros((self.max_buffer_size, self.number_of_actions))
        self.reward_memory = np.zeros(self.max_buffer_size)
        self.terminal_memory = np.zeros(self.max_buffer_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.max_buffer_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.max_buffer_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones
