import tensorflow as tf


def build_network(observation_dims, output_dims):
    input_layer = tf.keras.layers.Input(shape=observation_dims, name='Input_Layer', dtype=tf.float32)

    dense_layer_1 = tf.keras.layers.Dense(units=1024, activation='relu', name='Dense_Layer_1', dtype=tf.float32)(
        input_layer)
    dense_layer_2 = tf.keras.layers.Dense(units=512, activation='relu', name='Dense_Layer_2', dtype=tf.float32)(
        dense_layer_1)
    dense_layer_3 = tf.keras.layers.Dense(units=16, activation='relu', name='Dense_Layer_3', dtype=tf.float32)(
        dense_layer_2)

    action_logits = tf.keras.layers.Dense(units=output_dims, activation=None, name='Action_Probabilities',
                                          dtype=tf.float32)(dense_layer_2)
    state_value = tf.keras.layers.Dense(units=1, activation=None, name='State_Value', dtype=tf.float32)(dense_layer_3)

    network = tf.keras.models.Model(input_layer, [state_value, action_logits])

    return network
