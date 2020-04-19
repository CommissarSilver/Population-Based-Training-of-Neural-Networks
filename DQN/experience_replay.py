from collections import deque
import numpy as np
import environment
import random
import stack_frames
import copy


class Memory:
    # This class creates our experience replay buffer. each experience is defined as a combination of state, action,
    # reward, next state and if we're finished or not

    def __init__(self):
        # Define buffer
        self.buffer = {'state': [],
                       'action': [],
                       'reward': [],
                       'next_state': [],
                       'terminal': []}
        self.buffer_size = 0

    def add(self, state, action, reward, next_state, terminal):
        # To add new experiences to buffer
        self.buffer['state'].append(copy.deepcopy(state))
        self.buffer['action'].append(action)
        self.buffer['reward'].append(reward)
        self.buffer['next_state'].append(next_state)
        self.buffer['terminal'].append(terminal)

    def sample(self, batch_size):
        # To return a random batch from the buffer
        rand_indexes = np.random.randint(0, len(self.buffer['state']), batch_size)
        states = np.asarray([self.buffer['state'][i] for i in rand_indexes])
        actions = np.asarray([self.buffer['action'][i] for i in rand_indexes])
        rewards = np.asarray([self.buffer['reward'][i] for i in rand_indexes], dtype=np.float32)
        next_states = np.asarray([self.buffer['next_state'][i] for i in rand_indexes])
        terminals = np.asarray([self.buffer['terminal'][i] for i in rand_indexes])
        batch = (states, actions, rewards, next_states, terminals)
        self.buffer_size += 1
        if self.buffer_size>1000:
            self.buffer['state'].pop()
            self.buffer['action'].pop()
            self.buffer['reward'].pop()
            self.buffer['next_state'].pop()
            self.buffer['terminal'].pop()
            self.buffer_size-=1
        return batch


def create_and_fill_memory(stack_size=4, pretrain_length=64):
    # Arguments:
    #   stack_size: The size of stacks (how many frames are we going to stack together as a state)
    #   pretrain_length: How many instances we're going to fill the memory with before starting training
    # Returns:
    #   memory: The memory object that's going to be used for experience replay
    # Implements:
    #   It creates and fills a memory object which we're going to use to train our agent.

    # Create an empty deque, instantiate a memory object and create the environment
    stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    memory = Memory()
    game, possible_actions = environment.create_environment()

    for i in range(pretrain_length):
        # Start a new episode of the game after it ends
        game.new_episode()
        # Fill the deque with the first frame. this is going to be filled iteratively as the game is played with
        # different consecutive frames
        stacked_frames = stack_frames.stack_frames(stacked_frames, game.get_state().screen_buffer, True, stack_size)
        while not game.is_episode_finished():
            # Until we finish the game (Kill the monster) stack current frame in the state, choose a random action
            # and calculate its reward Add all of them to the memory
            state = game.get_state()
            stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
            action = random.choice(possible_actions)
            reward = game.make_action(action)
            terminal = game.is_episode_finished()
            next_stacked_frames = stack_frames.stack_frames(stacked_frames, state.screen_buffer, False)
            memory.add(np.asarray(stacked_frames).T, action, reward, np.asarray(next_stacked_frames).T, terminal)
    game.close()

    return memory
