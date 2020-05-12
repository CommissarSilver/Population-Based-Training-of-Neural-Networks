import tensorflow as tf
import numpy as np


class NeuralModel(tf.keras.Model):
    # Agent's neural model
    def __init__(self, input_shape, output_shape):
        super(NeuralModel, self).__init__()
        # Input layer for our NN
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        # first convolutional layer
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=32,
                                                  kernel_size=[8, 8],
                                                  strides=[4, 4],
                                                  padding='VALID',
                                                  kernel_initializer=tf.keras.initializers.he_normal())
        # batchnormalization for conv_layer1
        self.conv_layer1_batchnorm = tf.keras.layers.BatchNormalization(trainable=True,
                                                                        epsilon=1e-5, )
        # second convolutional layer
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=[4, 4],
                                                  strides=[2, 2],
                                                  padding='VALID',
                                                  kernel_initializer=tf.keras.initializers.he_normal())
        # batchnorm for conv_layer2
        self.conv_layer2_batchnorm = tf.keras.layers.BatchNormalization(trainable=True,
                                                                        epsilon=1e-5)

        # third convolutional layer
        self.conv_layer3 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=[4, 4],
                                                  strides=[2, 2],
                                                  padding='VALID',
                                                  kernel_initializer=tf.keras.initializers.he_normal())
        # batchnormalization for conv_layer3
        self.conv_layer3_batchnorm = tf.keras.layers.BatchNormalization(trainable=True,
                                                                        epsilon=1e-5)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.fully_connected1 = tf.keras.layers.Dense(units=512,
                                                      activation='relu',
                                                      kernel_initializer=tf.keras.initializers.he_normal())
        self.logits = tf.keras.layers.Dense(units=output_shape,
                                            activation=None,
                                            kernel_initializer=tf.keras.initializers.he_normal())

    @tf.function
    def call(self, inputs, in_training):
        out = self.input_layer(inputs)
        out = self.conv_layer1(out)
        out = self.conv_layer1_batchnorm(out)
        out = tf.nn.elu(out)
        out = self.conv_layer2(out)
        out = self.conv_layer2_batchnorm(out)
        out = tf.nn.elu(out)
        out = self.conv_layer3(out)
        out = self.conv_layer3_batchnorm(out)
        out = self.flatten_layer(out)
        out = self.fully_connected1(out)
        logits = self.logits(out)
        action_distribution = tf.nn.softmax(logits)
        if in_training:
            return logits
        else:
            return action_distribution


# model = NeuralModel([84, 84, 4], 3)
# x = np.random.randn(1, 84, 84, 4)
# out = model(x)
# print('hi')
