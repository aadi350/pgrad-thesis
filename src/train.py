import time
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
from tqdm import tqdm, trange
import matplotlib
from PIL import Image
from skimage import io
import pprint
from models.utils import show_progress
import numpy as np
from logging import log, info, debug
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv
import sys
from build_data import build_data_horizontal_separate
from models.models import build_resnet, build_siamese_autoencoder
from losses import DiceLoss
from metrics import DiceMetric
from matplotlib import pyplot as plt
from params import *
import json
from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

sys.path.append('/home/aadi/projects/pgrad-thesis/src/models')

# LOGGING/CONFIG FOR TF
wandb.init(mode='disabled', project="pgrad-thesis",
           entity="aadi350", tags=['run'])


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Before Image', 'After Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)


if __name__ == '__main__':

    # ensure clean divides
    if TRAIN_SIZE != -1:
        assert TRAIN_SIZE % BATCH_SIZE == 0
        assert TEST_SIZE % BATCH_SIZE == 0

    model = build_siamese_autoencoder()

    train_data, val_data = build_data_horizontal_separate(
        BATCH_SIZE, take=TRAIN_SIZE)
    logging.info('Data generators loaded')

    # initialize METRICS for Tracking progress
    train_dice = DiceMetric()
    test_dice = DiceMetric()
    val_dice = DiceMetric()

    # initialize Dice LOSS for training step
    loss_obj = DiceLoss()
    optimizer = tf.keras.optimizers.Adam()

    # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    config = {
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "n_samples": TRAIN_SIZE,
        "loss_fn": 'Dice Loss',
        "color": 'full_rgb',
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "other_params": {}
        }
    }
    #--------------------------------------TENSORFLOW TRAINING LOOP STEPS---------------------------------#

    @tf.function
    def train_step(model, input, labels, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        print('train')
        with tf.GradientTape() as tape:
            logits = model(input, training=True)
            # Instantiating DiceLoss() is for passing into model constructor
            print(logits.shape)
            # loss = dice_loss(labels, logits)
            loss = loss_obj(labels, logits)
        info(loss)

        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        train_dice(labels, logits)
        return loss

    @tf.function
    def val_step(model, input, labels):
        logits = model(input, training=False)
        loss = loss_obj(labels, logits)

        val_loss(loss)
        val_dice(labels, logits)
        return loss

    # ! Not used
    @tf.function
    def test_step(model, input, labels):
        test_logits = model(input, training=False)
        test_dice.update_state(labels, test_logits)

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # TensorBoard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    ckpt_dir = 'ckpts/model/' + current_time

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    with train_summary_writer.as_default():
        tf.summary.text('run_params', pretty_json(config), step=0)
    #--------------------------------------------TRAINING LOOP---------------------------------------------#

    MODEL_PATH = './siamese_autoencoder.h5'

    import ctypes

    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128

    for epoch in range(EPOCHS):
        log_dict = {}
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in tqdm(train_data):
            loss_value = train_step(
                model, x_batch_train, y_batch_train, optimizer)

        with train_summary_writer.as_default():
            tf.summary.scalar('train_dice', train_dice.result(), step=epoch)
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)

        train_dice.reset_states()
        # Run a validation loop at the end of each epoch.
        tqdm_val = tqdm(val_data)
        for x_batch_val, y_batch_val in tqdm_val:
            val_step(model, x_batch_val, y_batch_val)

            tqdm_val.set_description(
                f'Processing epoch: {epoch}, val_loss: {str(val_loss.result())}')

        with train_summary_writer.as_default():
            tf.summary.scalar('val_dice', val_dice.result(), step=epoch)
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch)

        train_loss.reset_states()
        val_loss.reset_states()

        train_dice.reset_states()
        val_dice.reset_states()

        model.save_weights(ckpt_dir + f'/{epoch}/model')
