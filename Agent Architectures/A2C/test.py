import A2C
import multiprocessing
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
if __name__ == '__main__':
    processs = []

    for i in range(4):

        _p = multiprocessing.Process(target=A2C.main, args=('train',))
        processs.append(_p)
    for process in processs:
        print('starting process')
        process.start()

agent1 = A2C.MasterAgent((84, 84, 4), [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     {'learning_rate': 0.9, 'discount_factor': 0.95, 'minions_num': 10})
for i in range(50):
    print('iteration: ', i)
    A2C.main('train')
