from configparser import Interpolation
from logging import info, debug
from re import I
from tkinter import W
import cupy as cp
import numpy as np
import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
# from osgeo import gdal
import rasterio
from utils import from_data
from PIL import Image
from skimage import io
from skimage.io import imshow
from rasterio.plot import show
from cucim import CuImage
import json
import pprint
import numpy as np
import cucim
from logging import log, info, debug
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from models.model_dummy import MyModel as Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Model
io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)

DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrag-thesis'

logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)


def do_differencing(before, after) -> np.array:
    return([b-a for b, a in zip(before, after)])


def build_single_stream(input_shape, embedding_dim=64):
    inputs = Input(input_shape)

    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    # to do cool stuff here
    outputs = Dense(embedding_dim)(x)

    return Model(inputs, outputs)


def difference_function(before, after):
    return np.subtract(before, after)


latent_dim = 16


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, input_shape=(
                256, 256, 3), activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(1024, activation='sigmoid'),
            layers.Reshape((32, 32)),
            layers.Resizing(256, 256, interpolation='bilinear')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def show_progress(it, milestones=1):
    for i, x in enumerate(it):
        yield x
        processed = i + 1
        if processed % milestones == 0:
            logging.info('Processed %s elements' % processed)


if __name__ == '__main__':
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    TEST_DIR = 'data/test'
    N = 240

    train_before_list = os.listdir(TRAIN_DIR + '/time1')[:N]
    train_after_list = os.listdir(TRAIN_DIR + '/time2')[:N]
    train_label_list = os.listdir(TRAIN_DIR + '/label')[:N]

    before_train = np.array([np.array(Image.open(TRAIN_DIR + '/time1/' + fname))
                            for fname in show_progress(train_before_list)])
    after_train = np.array([np.array(Image.open(TRAIN_DIR + '/time2/' + fname))
                           for fname in show_progress(train_after_list)])
    label_train = np.array([np.array(Image.open(TRAIN_DIR + '/label/' + fname))
                           for fname in show_progress(train_label_list)])

    train_diff = np.array([difference_function(b, a)
                          for (b, a) in zip(before_train, after_train)])

    single_r = [i[:, :, 0] for i in train_diff]
    single = train_diff[0]
    train_diff = train_diff[..., tf.newaxis]
    #label_train = label_train[..., tf.newaxis]
    val_before_list = os.listdir(VAL_DIR + '/time1')[:N]
    val_after_list = os.listdir(VAL_DIR + '/time2')[:N]
    val_label_list = os.listdir(VAL_DIR + '/label')[:N]
    before_val = np.array([np.array(Image.open(VAL_DIR + '/time1/' + fname))
                           for fname in show_progress(val_before_list)])
    after_val = np.array([np.array(Image.open(VAL_DIR + '/time2/' + fname))
                          for fname in show_progress(val_after_list)])
    label_val = np.array([np.array(Image.open(VAL_DIR + '/label/' + fname))
                          for fname in show_progress(val_label_list)])

    val_diff = np.array([difference_function(b, a)
                        for (b, a) in zip(before_train, after_train)])
    val_diff = val_diff[..., tf.newaxis]
    label_val = label_val[..., tf.newaxis]

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
    # logging.debug(model(train_diff))

#     model = Autoencoder()

    def to_int32(x): return (x/255.0).astype(np.float32)

    single_r = np.array([to_int32(i) for i in single_r])
    label_train = to_int32(label_train)
    out = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=single_r, labels=label_train)

    # works (apparently)
    # also in src/metrics.py
    def compute_loss(model, input, expected):
        actual = model(input)
        info(actual)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=actual, labels=expected)
        log_loss = -tf.reduce_sum(cross_ent, axis=[0, 1, 2])
        return -tf.reduce_mean(log_loss)

    out = model(single_r[..., tf.newaxis])
    info(out)

    loss = compute_loss(model, single_r[..., tf.newaxis], label_train)
    info(loss)

    # works
    @tf.function
    def train_step(model, input, expected, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, input, expected)

        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_step(model, single_r[..., tf.newaxis], label_train, optimizer)
