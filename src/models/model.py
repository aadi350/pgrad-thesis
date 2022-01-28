import tensorflow as tf


def build_model():
    # data shapes are 256x256x3
    INPUT_SHAPE = (256, 256, 1)
    IMAGE_H_W = (256, 256)

    # also works
    # equivalent to code in src/autoen.py
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    flatten = tf.keras.layers.Flatten()(inputs)
    dense_1 = tf.keras.layers.Dense(64, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(65536, activation='sigmoid')(dense_1)
    outputs = tf.keras.layers.Reshape((256, 256))(dense_2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
