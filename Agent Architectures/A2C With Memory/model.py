import tensorflow as tf


def build_model(input_shape, output_shape):
    # Arguments:
    #   input_shape: shape of inputs to the network
    #   output_shape: number of actions that an agent can take in the given environment
    # Returns:
    #   model: :|
    # Implements:
    #   Builds a model with the given input and output shapes

    # Input layer
    input_layer = tf.keras.layers.Input(shape=input_shape, name='Input_Layer', dtype=tf.float32)
    # Convolutional layers for processing the image input
    conv_layer1 = tf.keras.layers.Conv2D(filters=32,
                                         kernel_size=[8, 8],
                                         strides=[4, 4],
                                         padding='valid',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                         use_bias=True,
                                         name='Conv_Layer_1')(input_layer)
    conv_layer1_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                               trainable=True,
                                                               name='Conv_Layer_1_Batchnorm')(conv_layer1)

    conv_layer2 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=[4, 4],
                                         strides=[2, 2],
                                         padding='valid',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                         use_bias=True,
                                         name='Conv_Layer_2')(conv_layer1_batchnorm)
    conv_layer2_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                               trainable=True,
                                                               name='Conv_Layer_2_Batchnorm')(conv_layer2)

    conv_layer3 = tf.keras.layers.Conv2D(filters=128,
                                         kernel_size=[4, 4],
                                         strides=[2, 2],
                                         padding='valid',
                                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.),
                                         use_bias=True,
                                         name='Conv_Layer_3')(conv_layer2_batchnorm)
    conv_layer3_batchnorm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                               trainable=True,
                                                               name='Conv_Layer_3_Batchnorm')(conv_layer3)

    # Flattening the outputs of the convolutional layers and feeding them to a dense layer
    flatten_layer1 = tf.keras.layers.Flatten()(conv_layer3_batchnorm)
    dense_layer1 = tf.keras.layers.Dense(256, activation='relu')(flatten_layer1)

    # sate_value tells us how good is the state we're in right now
    state_value = tf.keras.layers.Dense(1, activation=None, dtype=tf.float32)(dense_layer1)
    # action_logits returns the value of each action in teh given state
    actions_logits = tf.keras.layers.Dense(units=output_shape, activation='softmax', dtype=tf.float32)(dense_layer1)

    # the model gives us two outputs. action values and state value
    model = tf.keras.models.Model(input_layer, [state_value, actions_logits])

    return model

# We're not using this anymore. delete it entirley? 
# class A2C:
#     def __init__(self, input_shape, possible_actions, hyper_params, optimizer):
#         self.input_shape = input_shape  # (frame width, frame height, stack size)
#         self.possible_actions = possible_actions  # one-hot possible actions
#         self.model = build_model(self.input_shape, len(self.possible_actions))
#         self.learning_rate = hyper_params['learning_rate']
#         self.optimizer = optimizer
