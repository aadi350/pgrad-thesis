from re import I
import wandb
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
# LOGGING/CONFIG FOR TF
wandb.init(project="pgrad-thesis", entity="aadi350", tags=['test'])

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)


logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)

#----------------------------------------------------------------------------------------------------#


def difference_function(before, after):
    return np.subtract(before, after)


if __name__ == '__main__':
    train = True
    test = False
    #-----------------------------------------------MODEL CONFIG/PARAMS----------------------------------------#
    # MODEL CONFIG/PARAMS
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    BATCH_SIZE = 100
    N = 300
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": None,
        "n_samples": N,
        "color": 'greyscale'
    }
    #-----------------------------------DATA SETUP AND PREPROCESSING------------------------------------#
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    TEST_DIR = 'data/test'
    #----------------------------------------TRAINING DATTA---------------------------------------------#
    # TRAINING DATA
    if train:

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

        # DISPLAY SOME IMAGES
        fig = make_subplots(rows=1, cols=4)
        fig.add_trace(px.imshow(before_train[3]).data[0], row=1, col=1)
        fig.add_trace(px.imshow(after_train[3]).data[0], row=1, col=2)
        fig.add_trace(px.imshow(single_grey[3],
                                binary_string=True, aspect='equal').data[0], row=1, col=3)
        fig.add_trace(
            px.imshow(label_train[3], binary_string=True, aspect='equal').data[0], row=1, col=4)

    # fig.show()
    if train:

        #------------------------------------------VALIDATION DATA---------------------------------------------#
        # VALIDATION DATA
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
                            for (b, a) in zip(before_val, after_val)])

        single_grey_val = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in val_diff]
        val_data = tf.data.Dataset.from_tensor_slices(
            (single_grey_val, label_val)).batch(batch_size=BATCH_SIZE)
        train_data = tf.data.Dataset.from_tensor_slices(
            (single_grey, label_train)).batch(batch_size=BATCH_SIZE)
        #------------------------------------------TESTDATA---------------------------------------------#
    # TEST DATA
    test_before_list = os.listdir(TEST_DIR + '/time1')[:N]
    test_after_list = os.listdir(TEST_DIR + '/time2')[:N]
    test_label_list = os.listdir(TEST_DIR + '/label')[:N]

    before_test = np.array([np.array(Image.open(TEST_DIR + '/time1/' + fname))
                            for fname in show_progress(test_before_list)])
    after_test = np.array([np.array(Image.open(TEST_DIR + '/time2/' + fname))
                           for fname in show_progress(test_after_list)])
    label_test = np.array([np.array(Image.open(TEST_DIR + '/label/' + fname))
                           for fname in show_progress(test_label_list)])

    test_diff = np.array([difference_function(b, a)
                          for (b, a) in zip(before_test, after_test)])

    single_grey_test = [cv.cvtColor(i, cv.COLOR_RGB2GRAY)
                        for i in test_diff]

    # TF DATA API

    x_test_data = tf.data.Dataset.from_tensor_slices(
        single_grey_test).batch(BATCH_SIZE)
    y_test_data = tf.data.Dataset.from_tensor_slices(label_test)
    test_data = tf.data.Dataset.from_tensor_slices(
        (single_grey_test, label_test)).batch(BATCH_SIZE)

    #------------------------------------------LOSS FUNCTIONS/METRICS---------------------------------------------#
    # LOSS FUNCTIONS

    class DiceLoss(tf.keras.losses.Loss):
        def __init__(self, smooth=1e-6, gama=2):
            super(DiceLoss, self).__init__()
            self.name = 'NDL'
            self.smooth = smooth
            self.gama = gama

        def call(self, y_true, y_pred):
            y_true, y_pred = tf.cast(
                y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
            nominator = 2 * \
                tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
            denominator = tf.reduce_sum(
                y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
            result = 1 - tf.divide(nominator, denominator)
            return result

    class DiceMetric(tf.keras.metrics.Metric):

        def __init__(self, name='dice_loss_metric', smooth=1e-6, gama=2, **kwargs):
            super(DiceMetric, self).__init__(name=name, **kwargs)
            self.dice = self.add_weight(name='dice', initializer='zeros')
            self.smooth = smooth
            self.gama = gama

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            nominator = 2 * \
                tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
            denominator = tf.reduce_sum(
                y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
            result = 1 - tf.divide(nominator, denominator)

            self.dice.assign(tf.divide(nominator, denominator))

        def result(self):
            return self.dice

    #-----------------------------------------------MODEL DEFINITION----------------------------------------#
    # MODEL DEF

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

    # initialize Dice merttric for training step
    train_dice = DiceMetric()
    test_dice = DiceMetric()
    val_dice = DiceMetric()

    # initialize Dice LOSS for training step
    dice_loss = DiceLoss()

    # works
    @tf.function
    def train_step(model, input, labels, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            logits = model(input, training=True)
            # Instantiating DiceLoss() is for passing into model constructor
            loss = dice_loss(labels, logits)
            info(loss)

            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_dice.update_state(labels, logits)
        return loss

    @tf.function
    def val_step(model, input, labels):
        logits = model(input, training=False)
        val_dice.update_state(labels, logits)

    @tf.function
    def test_step(model, input, labels):
        test_logits = model(input, training=False)
        test_dice.update_state(labels, test_logits)

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # TensorBoard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    #--------------------------------------------TRAINING LOOP---------------------------------------------#

    MODEL_PATH = './init_test_run.h5'
    if train:
        checkpoint_directory = "./tmp/training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        # TRAINING LOOP
        for epoch in range(EPOCHS):
            info(f'\nStart of epoch {epoch}')

            for step, (x_train, y_train) in enumerate(train_data):
                train_loss = train_step(
                    model, x_train, y_train, optimizer)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss, step=epoch)
                    tf.summary.scalar(
                        'train_dice', train_dice.result(), step=epoch)

            for step, (x_val, y_val) in enumerate(val_data):
                val_step(
                    model, x_val, y_val
                )
                with val_summary_writer.as_default():
                    tf.summary.scalar(
                        'val_dice', val_dice.result(), step=epoch)

            wandb.log({
                'train_dice': train_dice.result(),
                'train_loss': train_loss,
                'val_dice': val_dice.result()
            })
            if (epoch % 100 == 0):
                # save every 100 epochs
                checkpoint = tf.train.Checkpoint(model)
                checkpoint.save(file_prefix=checkpoint_prefix)

        model.save(MODEL_PATH)

        #model = tf.keras.models.load_model(MODEL_PATH)
        # TEST
        model.compile(optimizer=optimizer, loss=DiceLoss(),
                      metrics=[DiceMetric()])
        # hist = model.fit(train_data, epochs=EPOCHS, batch_size=BATCH_SIZE)

        loss, acc = model.evaluate(test_data)

        # output is numpy array
        output = model.predict(x_test_data)
        # output is tf object if instead output = model(x_test_data)
        # show output against expected using Plotly
        fig = make_subplots(rows=2, cols=2)
        fig.add_trace(px.imshow(output[0], binary_string=True,
                                aspect='equal').data[0], row=1, col=2)

        fig.add_trace(px.imshow(label_test[0], binary_string=True,
                                aspect='equal').data[0], row=1, col=1)
        fig.show()

        wandb.log({
            'test_dice': acc,
        })
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        features, labels = next(iter(test_data))
        pred = model.predict(features)
        pred.get_single_element()
