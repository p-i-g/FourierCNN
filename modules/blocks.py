from keras_core import ops
from keras_core import layers
from keras_core import models
from keras_core import activations
from keras_core import initializers
import tensorflow as tf
from modules import FourierConv2D


class FourierBlock2D(layers.Layer):
    """
    Largely follows ConvNeXt
    """
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.dw_conv = FourierConv2D(
            data_format='channels_last',
            activation=None,
            use_bias=False,
            name=f'{self.name}/dw_conv'
        )

        self.ln = layers.LayerNormalization(epsilon=1e-6)

        self.dense_1 = layers.Dense(filters * 4, activation='gelu', name=f'{self.name}/dense_1')
        self.dense_2 = layers.Dense(filters, activation=None, name=f'{self.name}/dense_2')

    def call(self, x, *args, **kwargs):
        x = self.dw_conv(x)
        x = self.ln(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x


class FourierCNN(models.Model):
    def __init__(
            self,
            num_classes=1000,
            blocks=(3, 3, 9, 3),
            filters=(96, 192, 384, 768),
            **kwargs
    ):
        super().__init__(**kwargs)

        self.blocks = blocks
        self.filters = filters
        self.num_classes = num_classes

        self.downsampling = []

        self.downsampling.append(models.Sequential([
            layers.Conv2D(filters[0], kernel_size=4, strides=(4, 4), use_bias=False, name=f'{self.name}/stem/conv2d'),
            layers.LayerNormalization(epsilon=1e-6, name=f'{self.name}/stem/layer_norm')
        ], name=f'{self.name}/stem'))

        for i in range(len(blocks) - 1):
            self.downsampling.append(models.Sequential([
                layers.LayerNormalization(epsilon=1e-6, name=f'{self.name}/downsampling_{i}/layer_norm'),
                layers.Conv2D(filters[0], kernel_size=2, strides=(2, 2), name=f'{self.name}/downsampling_{i}/conv2d')
            ], name=f'{self.name}/downsampling_{i}'))

        self.stages = []
        for i in range(len(blocks)):
            stage = []
            for j in range(blocks[i]):
                stage.append(FourierBlock2D(filters[i], name=f'{self.name}/stage_{i}/block{j}'))
            self.stages.append(models.Sequential(stage, name=f'{self.name}/stage_{i}'))

        self.pool = layers.GlobalAvgPool2D(name=f'{self.name}/final_pool')
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name=f'{self.name}/final_layer_norm')
        self.head = layers.Dense(num_classes, name=f'{self.name}/classification_head')

    def call(self, x, training=False):
        for i in range(len(self.blocks)):
            x = self.downsampling[i](x)
            x = self.stages[i](x)

        x = self.final_norm(self.pool(x))
        return self.head(x)
