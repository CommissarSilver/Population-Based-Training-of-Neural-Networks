import tensorflow.keras.backend as K
from agent import Agent
import random
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
import time

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
TRAIN_ITERS = 500


def train_agent(agent: Agent):
    # print("Agent {} Starting".format(agent.id))
    # trains for the given number of iterations
    for i in range(TRAIN_ITERS):
        agent.learn()
    returns = agent.episode_rewards
    return np.mean(returns)


def explore(agent: Agent):
    hyper_parameter_to_change = random.choice(['actor_learning_rate', 'critic_learning_rate'])
    j = random.uniform(0.7, 1.5)  # perturbs the agents learning rate
    if 'learning_rate' not in hyper_parameter_to_change:
        agent.hyper_parameters[hyper_parameter_to_change] *= j
        print(agent.hyper_parameters[hyper_parameter_to_change])
    else:
        if 'actor' in hyper_parameter_to_change:
            K.set_value(agent.actor.optimizer.learning_rate, agent.hyper_parameters[hyper_parameter_to_change])
            K.set_value(agent.target_actor.optimizer.learning_rate, agent.hyper_parameters[hyper_parameter_to_change])
        elif 'critic' in hyper_parameter_to_change:
            K.set_value(agent.critic.optimizer.learning_rate, agent.hyper_parameters[hyper_parameter_to_change])
            K.set_value(agent.target_critic.optimizer.learning_rate, agent.hyper_parameters[hyper_parameter_to_change])
            print(agent.hyper_parameters)


def exploit(training_results, agents):
    best_agent_id = sorted(training_results.items(), key=lambda x: x[1], reverse=True)[0][
        0]  # sort the agents according
    # their mean return
    for agent in agents:
        if agent.id == best_agent_id:
            best_agent = agent
    for agent in agents:
        # agent.hyper_parameters = best_agent.hyper_parameters
        # agent.actor.set_weights(best_agent.actor.get_weights())
        # agent.target_actor.set_weights(best_agent.target_actor.get_weights())
        # agent.critic.set_weights(best_agent.critic.get_weights())
        # agent.target_critic.set_weights(best_agent.target_critic.get_weights())
        agent.episode_rewards.clear()
    print('explore')
    explore(agents[best_agent_id])


if __name__ == '__main__':
    hyper_parameters = {'tau': 0.005, 'gamma': 0.99, 'actor_learning_rate': 0.0005, 'critic_learning_rate': 0.001}
    agents = [Agent(hyper_parameters, input_dims=8, env_name='LunarLanderContinuous-v2', n_actions=2, id=i) for i in
              range(4)]
    returns = {}
    i = 0
    while True:
        for agent in agents:
            agent_mean_returns = train_agent(agent)
            returns[agent.id] = agent_mean_returns
            i += 1
        if i % 30 == 0:
            exploit(returns, agents)
            print(agent.hyper_parameters)
