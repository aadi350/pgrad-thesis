import datetime
import pdb
from plotly import express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from logging import info, debug
from tkinter import W
import numpy as np
import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib
from PIL import Image
from skimage import io
import pprint
from utils import show_progress
import numpy as np
from logging import log, info, debug
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import Lambda
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Model

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)
#--------------------------------------DATA PATHS AND LOGGING----------------------------------------#
DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrag-thesis'

logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)

#----------------------------------------------------------------------------------------------------#


def difference_function(before, after):
    return np.subtract(before, after)


latent_dim = 16


if __name__ == '__main__':
    #-----------------------------------DATA SETUP AND PREPROCESSING------------------------------------#
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

    single_grey = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in train_diff]
    single = train_diff[0]
    train_diff = train_diff[..., tf.newaxis]

    fig = make_subplots(rows=1, cols=4)
    fig.add_trace(px.imshow(before_train[3]).data[0], row=1, col=1)
    fig.add_trace(px.imshow(after_train[3]).data[0], row=1, col=2)
    fig.add_trace(px.imshow(single_grey[3],
                  binary_string=True, aspect='equal').data[0], row=1, col=3)
    fig.add_trace(
        px.imshow(label_train[3], binary_string=True, aspect='equal').data[0], row=1, col=4)

    fig.show()
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

    single_grey_val = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in val_diff]
    val_diff = val_diff[..., tf.newaxis]

    def float_32(x): return (x/255.0).astype(np.float32)

    #------------------------------------------LOSS FUNCTIONS/METRICS---------------------------------------------#

    @tf.function
    def dice_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    class DiceLoss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)

            return 1 - numerator / denominator

    class DiceMetric(tf.keras.metrics.Metric):

        def __init__(self, name='dice_loss_fn_metric', **kwargs):
            super(DiceMetric, self).__init__(name=name, **kwargs)
            self.dice = self.add_weight(name='numerator', initializer='zeros')

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)

            self.dice.assign(1 - numerator / denominator)

        def result(self):
            return self.dice

    #-----------------------------------------------MODEL DEFINITION----------------------------------------#
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

    single_grey = np.array([float_32(i) for i in single_grey])
    single_grey_val = np.array([float_32(i) for i in single_grey_val])
    label_train = float_32(label_train)
    label_val = float_32(label_val)

    # works (apparently)
    # also in src/metrics.py

    @tf.function
    def cross_entropy_loss_fn(logits,  labels):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)
        log_loss = -tf.reduce_sum(cross_ent, axis=[0, 1, 2])
        return -tf.reduce_mean(log_loss)

    train_loss = tf.keras.metrics.Mean(
        'train_loss', dtype=tf.float32)  # mean of Dice Coefficient
    train_acc = tf.keras.metrics.BinaryCrossentropy(
        'train_accuracy')

    # initialize Dice for training step
    train_dice = DiceMetric()
    val_dice = DiceMetric()

    # works
    @ tf.function
    def train_step(model, input, labels, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            logits = model(input, training=True)
            # Instantiating DiceLoss() is for passing into model constructor
            loss = dice_loss_fn(logits, labels)

        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        train_dice(labels, logits)
        return loss

    @tf.function
    def val_step(model, input, labels):
        with tf.GradientTape() as tape:
            logits = model(input, training=False)
            loss = dice_loss_fn(labels, logits)

        val_dice(labels, logits)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    # TensorBoard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    #--------------------------------------------TRAINING LOOP---------------------------------------------#
    EPOCHS = 5000
    for epoch in range(EPOCHS):
        info(f'\nStart of epoch {epoch}')

        # TODO use batches instead of entire dataset
        train_step(
            model, single_grey[..., tf.newaxis], label_train, optimizer)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('dice', train_dice.result(), step=epoch)

        # TODO fix dimensions for single_grey_val and label_val to match
        val_step(
            model, single_grey_val[..., tf.newaxis], label_val
        )

        with val_summary_writer.as_default():
            tf.summary.scalar('val_dice', val_dice.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_dice.result(),
                              None,  # test_loss.result(),
                              val_dice.result()))  # test_accuracy.result()*100))
