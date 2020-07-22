import gym
from collections import deque
import numpy as np
import stack_frames

# We've moved on from VizDOOM to gym's atari environment


def create_environment(env_name='Breakout-v0'):
    environment = gym.make(env_name)
    observation_space = environment.observation_space
    action_space = environment.action_space
    return environment, observation_space, action_space


def test_environment():
    environment = gym.make('Breakout-v0')
    observation_space = environment.observation_space
    action_space = environment.action_space
    print(environment.unwrapped.get_action_meanings())
    print('Observation space: ', observation_space)
    print('Action space: ', action_space)
    observation = environment.reset()
    state = deque([np.zeros((84, 84)) for _ in range(4)], maxlen=4)
    stacked_frames = stack_frames.stack_frames(state, observation, True)
    for _ in range(1000):
        environment.render()
        observation, reward, done, info = environment.step(environment.action_space.sample())
        stacked_frames = stack_frames.stack_frames(stacked_frames, observation, False)
    environment.close()


# test_environment()
