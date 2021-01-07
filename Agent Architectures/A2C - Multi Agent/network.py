import tensorflow as tf
import os


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='ActorCritic', checkpoint_directory='\Agent-Models'):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims
        # Create a checkpoint directory in case we want to save our model
        self.model_name = name
        self.checkpoint_dir = checkpoint_directory
        current_dir = os.curdir
        self.checkpoint_file = os.path.join(os.getcwd()+self.checkpoint_dir, self.model_name + '.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=2048, activation='relu', name='Dense_Layer_1', dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2', dtype=tf.float32)
        self.dense_layer_3 = tf.keras.layers.Dense(units=256, activation='relu', name='Dense_Layer_3', dtype=tf.float32)
        # self.dense_layer_4 = tf.keras.layers.Dense(units=128, activation='relu', name='Dense_Layer_4', dtype=tf.float32)

        self.dense_layer_5 = tf.keras.layers.Dense(units=16, activation='relu', name='Dense_Layer_5', dtype=tf.float32)

        self.action_logits = tf.keras.layers.Dense(units=self.output_dims, activation=None, name='Action_Logits',
                                                   dtype=tf.float32)
        self.state_value = tf.keras.layers.Dense(units=1, activation='tanh', name='State_Value', dtype=tf.float32)

    def call(self, state):
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)
        x = self.dense_layer_3(x)  # use dense layer 3 as an extra layer for critic
        # x = self.dense_layer_4(x)
        y = self.dense_layer_5(x)

        action_logits = self.action_logits(x)
        state_value = self.state_value(y)

        return state_value, action_logits
