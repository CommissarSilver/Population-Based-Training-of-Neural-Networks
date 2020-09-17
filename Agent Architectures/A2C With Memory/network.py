import tensorflow as tf
import os


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='ActorCritic', checkpoint_directory='tmp/AC'):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims

        self.model_name = name
        self.checkPoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkPoint_dir, self.model_name + 'AC.h5')

        # self.input_layer = tf.keras.layers.Input(shape=self.input_dims, name='Input_Layer', dtype=tf.float32)
        self.dense_layer_1 = tf.keras.layers.Dense(units=1024, activation='relu', name='Dense_Layer_1',
                                                   dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2', dtype=tf.float32)
        self.dense_layer_3 = tf.keras.layers.Dense(units=16, activation='relu', name='Dense_Layer_3', dtype=tf.float32)
        self.action_logits = tf.keras.layers.Dense(units=self.output_dims, activation=None, name='Action_Logits',
                                                   dtype=tf.float32)
        self.state_value = tf.keras.layers.Dense(units=1, activation=None, name='State_Value', dtype=tf.float32)

    def call(self, state):
        # x = self.input_layer(state)
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)
        y = self.dense_layer_3(x)

        action_logits = self.action_logits(x)
        state_value = self.state_value(y)

        return state_value, action_logits
