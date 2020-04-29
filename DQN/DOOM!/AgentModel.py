import tensorflow as tf
import sumtree
import prioritized_experience_replay


def model_architecture(inputs_shape, output_shape):
    input_layer = tf.keras.layers.Input(shape=inputs_shape, name='Input_Layer')
    conv_layer1 = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=[8, 8],
                                         strides=[4, 4],
                                         padding='VALID',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                         use_bias=False,
                                         name='Conv_Layer_1')(input_layer)
    # conv_layer1_batchnorm: batchnormalization on conv_layer1's outputs
    conv_layer1_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                               trainable=True,
                                                               name='Conv_Layer_1_Batchnorm')(conv_layer1)
    # conv_layer2: second convolutional layer
    conv_layer2 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=[4, 4],
                                         strides=[2, 2],
                                         padding='VALID',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                         use_bias=False,
                                         name='Conv_Layer_2')(conv_layer1_batchnorm)
    # conv_layer2_batchnorm: batchnormalization on conv_layer2's outputs
    conv_layer2_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                               trainable=True,
                                                               name='Conv_Layer_2_Batchnorm')(conv_layer2)
    # conv_layer3: third convolutional layer
    conv_layer3 = tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=[4, 4],
                                         strides=[2, 2],
                                         padding='VALID',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                         use_bias=False,
                                         name='Conv_Layer_3')(conv_layer2_batchnorm)
    # conv_layer3_batchnorm: batchnormalization on conv_layer3's outputs
    conv_layer3_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                               trainable=True,
                                                               name='Conv_Layer_3_Batchnorm')(conv_layer3)

    # flattening the last layers outputs. We do this so we can use them as inputs to a dense layer
    value_stream, advantage_stream = tf.keras.layers.Lambda(lambda w: tf.split(w, 2, 3))(conv_layer3_batchnorm)
    value_stream = tf.keras.layers.Flatten()(value_stream)
    value_layer = tf.keras.layers.Dense(units=1,
                                        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(
        value_stream)

    advantage_stream = tf.keras.layers.Flatten()(advantage_stream)
    advantage_layer = tf.keras.layers.Dense(units=output_shape,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(
        advantage_stream)
    reduce_mean = tf.keras.layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))
    q_values = tf.keras.layers.Add()(
        [value_layer, tf.keras.layers.Subtract()([advantage_layer, reduce_mean(advantage_layer)])])

    return tf.keras.Model(input_layer, q_values)


class Agent:
    # Here we implement the agent's neural model
    def __init__(self, input_shape, output_shape, actions, learning_rate, a, memory_capacity, target_network=False):
        self.model = model_architecture(inputs_shape=input_shape, output_shape=output_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # hyperparameter
        self.actions = actions

        if not target_network:
            self.replay_buffer = prioritized_experience_replay.Memory(input_shape=(input_shape[0], input_shape[1]),
                                                                      stack_size=input_shape[2],
                                                                      num_actions=output_shape,
                                                                      a=a,
                                                                      max_len=memory_capacity)