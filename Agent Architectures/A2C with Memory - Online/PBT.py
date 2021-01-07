def exploit(population):
    sorted_population = sorted(population, key=lambda i: np.mean(i.scores), reverse=True)
    best_agents = sorted_population[:2]
    worst_agents = sorted_population[-3:]

    for agent in worst_agents:
        print(f'Agent -> {agent.id} is gonna go through exploitation and exploration')
    agent_to_exploit = random.choice(best_agents)
    print(f'Agent -> {agent_to_exploit.id} is selected for reproduction')
    # save agent's model
    agent_to_exploit.save_models()
    exploited_hyper_parameters = agent_to_exploit.hyper_parameters
    # for each other agent, load their models here
    for agent in worst_agents:
        agent.actor_network.set_weights(agent_to_exploit.actor_network.get_weights())
        agent.critic_network.set_weights(agent_to_exploit.critic_network.get_weights())
        agent.hyper_parameters = exploited_hyper_parameters

        K.set_value(agent.actor_network.optimizer.learning_rate,
                    agent_to_exploit.actor_network.optimizer.learning_rate)
        K.set_value(agent.critic_network.optimizer.learning_rate,
                    agent_to_exploit.critic_network.optimizer.learning_rate)

    for agent in worst_agents:
        explore(agent, agent_to_exploit)

    for agent in population:
        agent.scores.clear()


def explore(agent, best_agent):
    agent.hyper_parameters = best_agent.hyper_parameters
    new_actor_learning_rate = round(best_agent.hyper_parameters['actor_learning_rate'] * random.uniform(0.8, 1.2), 6)
    new_critic_learning_rate = round(best_agent.hyper_parameters['critic_learning_rate'] * random.uniform(0.8, 1.2), 6)
    agent.actor_network.optimizer.learning_rate.assign(new_actor_learning_rate)
    agent.critic_network.optimizer.learning_rate.assign(new_critic_learning_rate)
    agent.hyper_parameters['actor_learning_rate'] = new_actor_learning_rate
    agent.hyper_parameters['critic_learning_rate'] = new_critic_learning_rate


environment_name = 'CartPole-v0'
log_file_name = 'PBT-2nd Run'
agents = []
for i in range(10):
    agents.append(Agent('{}'.format(environment_name),
                        {'actor_learning_rate': round(random.uniform(0.00007, 0.005), 6),
                         'critic_learning_rate': round(random.uniform(0.00007, 0.005), 6),
                         'entropy_coefficient':  0, #round(random.uniform(0.00001, 0.001), 4),
                         'critic_coefficient': round(random.uniform(0.1, 0.5), 4),
                         'discount_factor': 0.99},
                        i + 1,
                        log_file_name))

f = open(f'{environment_name}-{log_file_name}.csv', 'a')
f.write('Agent ID,Episode Number,Episode Score,Actor Learning Rate,Critic Learning Rate\n')
f.close()

for j in range(1, 500):
    for agent in agents:
        agent.play()
    print(f'Agents Finished Episode {j}')
    if j % 50 == 0:
        exploit(agents)
