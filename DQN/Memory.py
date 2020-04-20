import random
import numpy as np


# creates a replay buffer with prioritized experience replay
class Node:
    # This class implements a SumTree which we'll use for PER
    def __init__(self, left, right, is_leaf=False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.id = idx

        if not self.is_leaf:
            self.value = self.right.value + self.left.value
            self.left.parent = self
            self.right.parent = self

        self.parent = None

    @classmethod
    def create_leaf(cls, value, idx):
        # Arguments:
        #   cls: the class instance
        #   value: node's value
        #   idx: node's id
        # Returns:
        #   leaf: a leaf node
        # Implements:
        #   creates leaf nodes from the given value and id
        leaf = cls(None, None, True, idx)
        leaf.value = value

        return leaf


def create_sumtree(buffer, a, sum_probs):
    # Arguments:
    #   buffer: the replay buffer
    #   a: a hyperparameter which is used to introduce some randomness in the experience selection
    #   sum_probs: sum of all the calculated probabilities of experiences inside the buffer
    # Returns:
    #   nodes[0]: which is the root of our SumTree and contains the entire tree
    # Implements:
    #   creates a SumTree out of the leaf nodes given to it
    nodes = []

    for i in list(buffer.keys()):
        nodes.append(Node.create_leaf(value=(buffer[i].td_error ** a) / sum_probs, idx=i))

    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0]


def retrieve(tree: Node, probs):
    # Arguments:
    #   tree: an entire SumTree
    #   probs: probabilities for selecting nodes
    # Returns:
    #   batch: a batch of selected nodes according to the probabilities in probs
    # Implements:
    #   selects and returns samples from the SumTree according to the prioritized experience replay method
    batch = []

    for prob in probs:
        node = tree

        while not node.is_leaf:
            if prob < node.left.value and not node.is_leaf:
                node = node.left
            elif prob > node.left.value and not node.is_leaf:
                prob = abs(prob - node.right.value)
                node = node.right
            elif node.is_leaf:
                node = node
        batch.append(node.id)

    return batch


class Experience:
    # here we create an experience object
    def __init__(self, state, action, reward, next_state, done, id, td_error):
        self.state = state  # current state
        self.action = action  # action taken at current state
        self.reward = reward  # reward received after taking action
        self.next_state = next_state  # the state we;ve transitioned to by taking action
        self.done = done  # determines if we have reached terminal state or not
        self.td_error = td_error  # td error of this experience
        self.id = id  # experience's id


class Memory:
    # here we create our agent's memory
    def __init__(self, a, maximum_length=1000000):
        self.buffer = {}  # replay buffer
        self.counter = 0  # keeps count of how many experiences we've had
        self.sum_probs = 0  # sum of all experiences' probabilities
        self.a = a  # a hyperparameter which is used to introduce some randomness in the experience selection
        self.max_len = maximum_length  # maximum capacity of agent's replay buffer

    def add(self, state, action, reward, next_state, done, td_error):
        # Arguments:
        #   all the arguments of an experience explained above
        # Returns:
        #   -
        # Implements:
        #   adds an experience to the replay buffer. if the replay buffer is at maximum capacity,
        #   deletes a random memory and adds the new experience in its stead.
        self.counter += 1
        if len(self.buffer.keys()) + 1 < self.max_len:
            self.buffer[self.counter] = Experience(state=np.asarray(state), action=np.asarray(action),
                                                   reward=np.asarray(reward), next_state=np.asarray(next_state),
                                                   done=np.asarray(done), id=self.counter, td_error=td_error)
            self.sum_probs += td_error ** self.a
        else:
            rand_idx = random.choice(list(self.buffer.keys()))
            self.sum_probs -= (self.buffer[rand_idx].td_error ** self.a)
            del self.buffer[rand_idx]
            self.buffer[self.counter] = Experience(state=np.asarray(state), action=np.asarray(action),
                                                   reward=np.asarray(reward),
                                                   next_state=np.asarray(next_state), done=np.asarray(done),
                                                   id=self.counter,
                                                   td_error=td_error)
            self.sum_probs += td_error ** self.a

    def sample(self, batch_size):
        # Arguments:
        #   batch_size: batch size
        # Returns:
        #   sample: a number of experiences selected from our replay buffer according to PER
        # Implements:
        #   selecting random prioritized experiences from the buffer
        tree = create_sumtree(self.buffer, self.a, self.sum_probs)
        batch_probs = [random.uniform(0, tree.value) for _ in range(batch_size)]
        batch = retrieve(tree, batch_probs)
        sample = [self.buffer[i] for i in batch]
        states = np.zeros((batch_size, 84, 84, 4))
        next_states = np.zeros((batch_size, 84, 84, 4))
        actions = np.zeros((batch_size, 3))
        rewards = np.zeros((batch_size, 1))
        terminals = np.zeros((batch_size, 1),dtype=np.bool)
        for index,i in enumerate(batch):
            states[index] = self.buffer[i].state
            next_states[index] = self.buffer[i].next_state
            actions[index] = self.buffer[i].action
            rewards[index] = self.buffer[i].reward
            terminals[index] = self.buffer[i].done
        return states, actions, rewards, next_states, terminals
