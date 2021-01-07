import tensorflow as tf
import os


class ActorNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='Actor', checkpoint_directory=f'{os.getcwd()}\\Agent Models'):
        super(ActorNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims
        # Create a checkpoint directory in case we want to save our model
        self.model_name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=128, activation='relu', name='Dense_Layer_1',
                                                   dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=128, activation='relu', name='Dense_Layer_2',
                                                   dtype=tf.float32)
        self.action_logits = tf.keras.layers.Dense(units=self.output_dims, activation=None, name='Action_Logits',
                                                   dtype=tf.float32)

    def call(self, state):
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)

        action_logits = self.action_logits(x)
        return action_logits


class CriticNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='Critic', checkpoint_directory=f'{os.getcwd()}\\Agent Models'):
        super(CriticNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims
        # Create a checkpoint directory in case we want to save our model
        self.model_name = name
        self.checkpoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=128, activation='relu', name='Dense_Layer_1',
                                                   dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=128, activation='relu', name='Dense_Layer_2',
                                                   dtype=tf.float32)
        self.state_value = tf.keras.layers.Dense(units=1, activation=None, name='State_Value',
                                                 dtype=tf.float32)

    def call(self, state):
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)

        state_value = self.state_value(x)
        return state_value
