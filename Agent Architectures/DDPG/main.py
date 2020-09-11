import gym
from agent import Agent

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    n_games = 1000
    load_checkpoint = False
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    scores = open('scores-1e6.csv', 'a')
    scores.write('Episode,Score\n')
    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state, evaluate)
            next_state, reward, done, info = env.step(action)
            # env.render()
            score += reward
            agent.remember(state, action, reward, next_state, done)
            if not load_checkpoint:
                agent.learn()
            state = next_state
        print(i, '->', score)
        scores = open('scores1e6.csv', 'a')
        scores.write('{0},{1}\n'.format(i, score))
        scores.close()
