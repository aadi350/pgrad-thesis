from rasterio import pad
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import layers


INPUT_SHAPE = (256, 256, 3)
IMAGE_H_W = (256, 256)


def build_resnet():
    # data shapes are 256x256x3

    # also works
    # equivalent to code in src/autoen.py
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="img")
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(65536, activation="relu")(x)

    outputs = tf.keras.layers.Reshape(IMAGE_H_W)(x)
    model = tf.keras.Model(inputs, outputs, name="toy_resnet")
    model.summary()

    return model

# ---------------------SIAMESE MODEL -------------------- #


def build_in_channel(name):
    inputs = Input(shape=INPUT_SHAPE, name=name)
    out_0 = Conv2D(16, (2, 2), padding='same')(inputs)
    out_0 = Conv2D(16, (2, 2), padding='same')(out_0)

    # shape = 256, 256, 16

    out_1 = MaxPooling2D((2, 2))(out_0)
    # shape = 128, 128, 16
    # 2x2 MaxPool HALVES w and h

    out_1 = Conv2D(16, (2, 2), padding='same')(out_1)
    out_1 = Conv2D(32, (2, 2), padding='same')(out_1)
    out_1 = Conv2D(32, (2, 2), padding='same')(out_1)
    # shape = 128, 128, 32

    out_2 = MaxPooling2D((2, 2))(out_1)
    # shape = 64, 64, 32

    out_2 = Conv2D(32, (2, 2), padding='same')(out_2)
    out_2 = Conv2D(64, (2, 2), padding='same')(out_2)
    out_2 = Conv2D(64, (2, 2), padding='same')(out_2)
    out_2 = Conv2D(64, (2, 2), padding='same')(out_2)
    # shape = 64, 64, 64

    out_3 = MaxPooling2D((2, 2))(out_2)
    # shape = 32, 32, 64

    out_3 = Conv2D(64, (2, 2),   padding='same')(out_3)
    out_3 = Conv2D(128, (2, 2),  padding='same')(out_3)
    out_3 = Conv2D(128, (2, 2),  padding='same')(out_3)
    out_3 = Conv2D(128, (2, 2),  padding='same')(out_3)
    # shape = 32, 32, 128

    return inputs, (out_0, out_1, out_2, out_3)


def build_siamese_autoencoder():

    left_in, (l_out_0, l_out_1, l_out_2, l_out_3) = build_in_channel('left')
    right_in, (r_out_0, r_out_1, r_out_2, r_out_3) = build_in_channel('right')

    output = subtract([l_out_3, r_out_3])
    # shape = 32, 32, 128

    l_out_3 = MaxPooling2D((2, 2), padding='same')(l_out_3)
    # shape = 16, 16, 128
    l_out_3 = Conv2DTranspose(128, 2, padding='same')(l_out_3)
    # shape = 16, 16, 128
    l_out_3 = Conv2DTranspose(128, 2, padding='same')(l_out_3)
    # shape = 16, 16, 128
    l_out_3 = UpSampling2D((2, 2))(l_out_3)
    # shape = 32, 32, 128

    diff_3 = subtract([l_out_3, r_out_3])
    output = concatenate([output, diff_3])
    # shape = 32, 32, 256

    output = Conv2DTranspose(256, 2, padding='same')(output)
    # shape = 32, 32, 256
    output = Conv2DTranspose(128, 2, padding='same')(output)
    # shape = 32, 32, 128
    output = Conv2DTranspose(128, 2, padding='same')(output)
    # shape = 32, 32, 128
    output = Conv2DTranspose(64, 2, padding='same')(output)
    # shape = 32, 32, 64
    output = Conv2DTranspose(64, 2, padding='same')(output)
    # shape = 32, 32, 64
    output = Conv2DTranspose(64, 2, padding='same')(output)
    # shape = 32, 32, 64
    output = UpSampling2D((2, 2))(output)
    # shape = 64, 64, 64

    diff_2 = subtract([l_out_2, r_out_2])
    output = concatenate([output, diff_2])
    # shape = 64, 64, 128

    output = Conv2DTranspose(128, (2, 2), padding='same')(output)
    output = Conv2DTranspose(64, (2, 2), padding='same')(output)
    output = Conv2DTranspose(64, (2, 2), padding='same')(output)
    output = Conv2DTranspose(32, (2, 2), padding='same')(output)
    output = Conv2DTranspose(32, (2, 2), padding='same')(output)
    output = Conv2DTranspose(32, (2, 2), padding='same')(output)
    # shape = 64, 64, 32

    output = UpSampling2D((2, 2))(output)
    # shape = 128, 128, 32

    diff_1 = subtract([l_out_1, r_out_1])
    output = concatenate([output, diff_1])
    # shape = 128, 128, 64

    output = Conv2DTranspose(64, (2, 2), padding='same')(output)
    output = Conv2DTranspose(32, (2, 2), padding='same')(output)
    output = Conv2DTranspose(16, (2, 2), padding='same')(output)
    output = Conv2DTranspose(16, (2, 2), padding='same')(output)
    output = Conv2DTranspose(16, (2, 2), padding='same')(output)
    # shape = 128, 128, 16

    output = UpSampling2D((2, 2))(output)
    # shape = 256, 256, 16

    diff_0 = subtract([l_out_0, r_out_0])
    output = concatenate([output, diff_0])
    # shape = 256, 256, 32
    output = Conv2DTranspose(32, (2, 2), padding='same')(output)
    output = Conv2DTranspose(16, (2, 2), padding='same')(output)
    output = Conv2DTranspose(1, (2, 2), padding='same')(output)
    # shape =  256, 256, 1

    output = tf.keras.activations.sigmoid(output)

    model = tf.keras.models.Model(inputs=[right_in, left_in], outputs=output)

    return model
