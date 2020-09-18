import tensorflow as tf
import os


class CriticNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_1', dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2', dtype=tf.float32)
        self.q = tf.keras.layers.Dense(units=1, activation=None, name='Qs', dtype=tf.float32)

    def call(self, state, action):
        action_value = self.dense_layer_1(tf.concat([state, action], axis=1))
        action_value = self.dense_layer_2(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='actor', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_1', dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2', dtype=tf.float32)
        self.mu = tf.keras.layers.Dense(units=self.output_dims, activation='tanh', name='Mu', dtype=tf.float32)

    def call(self, state):
        prob = self.dense_layer_1(state)
        prob = self.dense_layer_2(prob)

        mu = self.mu(prob)

        return mu
