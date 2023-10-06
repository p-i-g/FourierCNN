from keras_core import ops
from keras_core import layers
from keras_core import activations
from keras_core import initializers
import tensorflow as tf


class FourierConv2D(layers.Layer):
    """
    Implements a 2D convolution using a Fourier Transform
    """
    def __init__(
            self,
            data_format=None,
            activation=None,
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.data_format = data_format or 'channels_last'
        self.use_bias = use_bias
        self.real_kernel_initializer = initializers.get(kernel_initializer)
        self.imag_kernel_initializer = initializers.get(kernel_initializer)

        if self.use_bias:
            self.bias_initializer = initializers.get(bias_initializer)

        self.real_kernel = None
        self.imag_kernel = None

        if self.use_bias:
            self.bias = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            n, c, h, w = input_shape
        else:
            n, h, w, c = input_shape

        kernel_shape = (c, h, w // 2 + 1)
        bias_shape = (c, h, w)

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            dtype=self.dtype,
            initializer=self.real_kernel_initializer,
            name=f'{self.name}/real_kernel'
        )

        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            dtype=self.dtype,
            initializer=self.real_kernel_initializer,
            name=f'{self.name}/real_kernel'
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=bias_shape,
                dtype=self.dtype,
                initializer=self.bias_initializer,
                name=f'{self.name}/bias'
            )

    def call(self, x, *args, **kwargs):
        kernel = tf.complex(real=self.real_kernel, imag=self.imag_kernel)

        if self.data_format == 'channels_last':
            x = ops.transpose(x, (0, 3, 1, 2))

        x = tf.signal.rfft2d(x)
        x = x * kernel

        x = tf.signal.irfft2d(x)
        if self.use_bias:
            x += self.bias

        if self.data_format == 'channels_last':
            x = ops.transpose(x, (0, 2, 3, 1))

        return self.activation(x)
