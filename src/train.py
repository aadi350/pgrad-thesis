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
from tqdm import tqdm
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

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

sys.path.append('/home/aadi/projects/pgrad-thesis/src/models')

# LOGGING/CONFIG FOR TF
wandb.init(project="pgrad-thesis",
           entity="aadi350", tags=['run'])


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

    #-----------------------------------------------MODEL CONFIG/PARAMS----------------------------------------#
    # MODEL CONFIG/PARAMS
    LEARNING_RATE = 1e-3
    EPOCHS = 1000
    BATCH_SIZE = 25
    STEPS_PER_EPOCH = 12000 // BATCH_SIZE
    TRAIN_SIZE = -1
    TEST_SIZE = 200
    # ensure clean divides
    if TRAIN_SIZE != -1:
        assert TRAIN_SIZE % BATCH_SIZE == 0
        assert TEST_SIZE % BATCH_SIZE == 0

    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "n_samples": TRAIN_SIZE,
        "color": 'full_rgb'
    }

    model = build_siamese_autoencoder()

    train_data, val_data = build_data_horizontal_separate(BATCH_SIZE)
    logging.info('Data generators loaded')

    # initialize METRICS for Tracking progress
    train_dice = DiceMetric()
    test_dice = DiceMetric()
    val_dice = DiceMetric()

    # initialize Dice LOSS for training step
    dice_loss = DiceLoss()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #--------------------------------------TENSORFLOW TRAINING LOOP STEPS---------------------------------#

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
            # loss = bce_loss(labels, logits)
            info(loss)

            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_dice.update_state(labels, logits)
        return loss

    @tf.function
    def val_step(model, input, labels):
        logits = model(input, training=False)
        val_dice.update_state(labels, logits)

        loss = dice_loss(labels, logits)
        return loss

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

    MODEL_PATH = './siamese_autoencoder.h5'
    checkpoint_directory = "./tmp/training_checkpoints/siamese_autoen"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    for epoch in range(EPOCHS):
        log_dict = {}
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            loss_value = train_step(
                model, x_batch_train, y_batch_train, optimizer)

            wandb.log({
                'train_loss': loss_value,
                'epoch': epoch,
                'train_dice': train_dice.result()
            })
        with train_summary_writer.as_default():
            tf.summary.scalar('train_dice', train_dice.result(), step=epoch)
            tf.summary.scalar('train_loss', loss_value, step=epoch)

        train_dice.reset_states()
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_data:
            val_loss = val_step(x_batch_val, y_batch_val)

            wandb.log({
                'val_dice': val_dice.result()
            })
        with train_summary_writer.as_default():
            tf.summary.scalar('val_dice', val_dice.result(), step=epoch)
            tf.summary.scalar('val_loss', val_loss, step=epoch)

        train_dice.reset_states()
        val_dice.reset_states()
