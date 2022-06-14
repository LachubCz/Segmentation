import tensorflow as tf
from functools import partial


class SimpleNet(tf.keras.Model):
    def __init__(self, data_format="channels_last"):
        super(SimpleNet, self).__init__()

        self.model = tf.keras.Sequential()
        self.data_format = data_format
        # first block
        self.model.add(tf.keras.layers.Conv2D(32, (21, 21), padding="same",
                                              kernel_initializer="he_normal", data_format=data_format))
        self.model.add(tf.keras.layers.ReLU())
        # second block
        self.model.add(tf.keras.layers.Conv2D(1, (5, 5), padding="same",
                                              kernel_initializer="he_normal", data_format=data_format))

    def call(self, x, training=True):
        x = tf.cast(x, dtype=tf.float32) * (1. / 255)
        if self.data_format == "channels_first":
            x = tf.transpose(x, [0, 3, 1, 2])
        x = self.model(x, training=training)
        return x  # tf.math.softplus(x)


class OriginalNet(tf.keras.Model):
    def __init__(self, data_format="channels_last"):
        super(OriginalNet, self).__init__()

        self.first_model = tf.keras.Sequential()
        self.first_model.add(tf.keras.layers.Conv2D(1, (3, 3), padding="same", data_format=data_format))
        self.first_model.add(tf.keras.layers.ReLU())

        self.second_model = tf.keras.Sequential()
        self.second_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", data_format=data_format))
        self.second_model.add(tf.keras.layers.ReLU())
        self.second_model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same'))
        self.second_model.add(tf.keras.layers.ReLU())

        self.third_model = tf.keras.Sequential()
        self.third_model.add(tf.keras.layers.Conv2D(1, (3, 3), padding="same", data_format=data_format))
        self.third_model.add(tf.keras.layers.ReLU())

    def call(self, x, training=True):
        x = tf.cast(x, dtype=tf.float32) * (1. / 255)
        first = self.first_model(x, training=training)
        second = self.second_model(first, training=training)
        merge = tf.keras.layers.concatenate([second, first], axis=3)
        return self.third_model(merge, training=training)


class InferenceModel(tf.keras.Model):
    def __init__(self, original_model, data_format):
        super(InferenceModel, self).__init__()
        self.model = original_model
        self.data_format = data_format

    def call(self, x, training=True):
        return self.model(x, training=training)


NET_CONFIGS = {
    'OriginalNet': partial(OriginalNet),
    'SimpleNet': partial(SimpleNet)
}
