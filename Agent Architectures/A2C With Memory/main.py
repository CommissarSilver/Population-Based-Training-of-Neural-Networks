from agent import Agent

if __name__ == '__main__':
    master = Agent('CartPole-v0',
                   {'learning_rate': 0.0001, 'discount_factor': 0.99, 'unroll_length': 50, 'minions_num': 5})
    master.learn()
