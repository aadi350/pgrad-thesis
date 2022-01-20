import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *


# ---------------CUSTOM LAYERS------------------ #
class SomeThingMeaninful(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(
            mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(
            mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def MyModel(output_channels: int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    outputs = inputs

    return tf.keras.Model(inputs=inputs, outputs=outputs)
