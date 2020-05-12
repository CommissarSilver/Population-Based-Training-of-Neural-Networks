import random
import numpy as np
import sumtree


class Memory:
    def __init__(self, input_shape, stack_size, num_actions, a, max_len=10000):
        self.input_shape = input_shape
        self.stack_size = stack_size
        self.num_actions = num_actions

        self.memory = {i: {'state': np.zeros((self.input_shape[0], self.input_shape[1], self.stack_size)),
                           'action': np.zeros(self.num_actions),
                           'reward': None,
                           'next_state': np.zeros((self.input_shape[0], self.input_shape[1], self.stack_size)),
                           'terminal': False,
                           'error': 0.001}
                       for i in range(max_len)}

        self.memory, self.leaf_nodes = sumtree.create_tree(self.memory)
        self.counter = 0
        self.capacity = max_len
        self.a = a

    def add_experience(self, state, action, reward, next_state, terminal, error):

        if self.counter != self.capacity - 1:
            self.counter += 1
        else:
            self.counter = 0

        node_to_be_updated = [node for node in self.leaf_nodes if node.index == self.counter][0]
        node_to_be_updated.content['state'] = state
        node_to_be_updated.content['action'] = action
        node_to_be_updated.content['reward'] = reward
        node_to_be_updated.content['next_state'] = next_state
        node_to_be_updated.content['terminal'] = terminal

        sumtree.update(node=node_to_be_updated, new_value=(error ** self.a) / self.memory.value)

    def sample(self, batch_size):
        batches_states = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.stack_size))
        batches_actions = np.zeros((batch_size, self.num_actions))
        batches_rewards = np.zeros(batch_size)
        batches_next_states = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.stack_size))
        batches_terminals = np.zeros(batch_size)
        indices = [random.uniform(0, self.memory.value) for _ in range(batch_size)]

        for index, indice in enumerate(indices):
            sample = sumtree.retrieve(indice, self.memory).content
            batches_states[index] = sample['state']
            batches_actions[index] = sample['action']
            batches_rewards[index] = sample['reward']
            batches_next_states[index] = sample['next_state']
            batches_terminals[index] = sample['terminal']

        return batches_states, batches_actions, batches_rewards, batches_next_states, batches_terminals
