import DQN


class Worker:
    def __init__(self, state_shape, possible_actions, agent_hyper_parameters):
        self.agent = DQN.DQN(state_shape=state_shape, hyper_parameters=agent_hyper_parameters,
                             possible_actions=possible_actions)
