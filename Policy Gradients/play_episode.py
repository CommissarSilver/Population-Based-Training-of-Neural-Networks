import environment
import numpy as np
import stack_frames
from collections import deque

# create a new environment for the agent to play in
game, possible_actions = environment.create_environment()


def discount_and_normalize_rewards(gamma, episode_rewards):
    # Arguments:
    #   gamma: discount rate
    #   episode_rewards: entire rewards collected during playing
    # Returns:
    #   discounted_episode_rewards: episode's rewards discounted with gamma and normalized
    # Implements:
    #   gets the episode's rewards and then discounts and normalizes them for training the agent
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0

    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std

    return discounted_episode_rewards


def play_episode(agent):
    # Arguments:
    #   agent: agent!
    # Returns: states, actions and discounted rewards for an episode. An episode ends with either shooting the monster
    #                  or timeout
    # Implements:
    #   plays an episode until it either shoots the monster or runs out of time. during playing it collects and stores
    #   all the states visited by the agent actions taken in those states and rewards for the actions taken on those
    #   states. after that it calculates the discounted rewards and returns them.
    states = []
    actions = []
    rewards_of_episode = []

    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        stacked_states = deque([np.zeros((84, 84), dtype=np.int) for _ in range(4)], maxlen=4)
        stacked_states = stack_frames.stack_frames(stacked_states, state.screen_buffer, True)
        action_distribution = agent.predict_action_probs(np.asarray(stacked_states).reshape(1, 84, 84, 4),
                                                         False).numpy()

        action = np.random.choice(range(3), p=action_distribution.ravel())
        action = possible_actions[action]
        reward = game.make_action(action)
        states.append(np.asarray(stacked_states).reshape(84, 84, 4))
        actions.append(action)
        rewards_of_episode.append(reward)

    discounted_rewards = discount_and_normalize_rewards(agent.discount_rate, rewards_of_episode)

    return states, actions, discounted_rewards
