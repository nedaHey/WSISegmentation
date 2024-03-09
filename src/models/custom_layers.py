import tensorflow.keras.layers as tfkl

from ..config import UNetTrainingConfig


def _conv_block(x, n_filters, kernel_size, activation, padding):
    x = tfkl.Conv2D(n_filters,
                    kernel_size,
                    activation=None,
                    kernel_initializer="he_normal",
                    padding=padding,
                    kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(activation)(x)

    x = tfkl.Conv2D(n_filters,
                    kernel_size,
                    activation=None,
                    kernel_initializer="he_normal",
                    padding=padding,
                    kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(activation)(x)
    return x


def _transconv_block(x, n_filters, kernel_size, activation, padding):
    x = tfkl.Conv2DTranspose(n_filters,
                             kernel_size,
                             activation=None,
                             kernel_initializer="he_normal",
                             padding=padding,
                             kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation(activation)(x)

    x = tfkl.Conv2DTranspose(n_filters,
                             kernel_size,
                             activation=None,
                             kernel_initializer="he_normal",
                             padding=padding,
                             kernel_regularizer=UNetTrainingConfig.KERNEL_REGULARIZER)(x)
    x = tfkl.Activation(activation)(x)
    x = tfkl.BatchNormalization()(x)
    return x