import tensorflow as tf
import os


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, observation_dims, output_dims, name='PolicyGradient', checkpoint_directory='tmp/PG'):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = observation_dims
        self.output_dims = output_dims
        # Create a checkpoint directory in case we want to save our model
        self.model_name = name
        self.checkPoint_dir = checkpoint_directory
        self.checkpoint_file = os.path.join(self.checkPoint_dir, self.model_name + 'PG.h5')

        self.dense_layer_1 = tf.keras.layers.Dense(units=1024, activation='relu', name='Dense_Layer_1',
                                                   dtype=tf.float32)
        self.dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2', dtype=tf.float32)
        self.dense_layer_3 = tf.keras.layers.Dense(units=16, activation='relu', name='Dense_Layer_3', dtype=tf.float32)
        self.action_logits = tf.keras.layers.Dense(units=self.output_dims, activation=None, name='Action_Logits',
                                                   dtype=tf.float32)

    def call(self, state):
        x = self.dense_layer_1(state)
        x = self.dense_layer_2(x)
        x = self.dense_layer_3(x)

        action_logits = self.action_logits(x)

        return action_logits
