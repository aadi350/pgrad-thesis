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

from build_data import build_data_grey, build_test_data
from models.model import build_model
from losses import DiceLoss
from metrics import DiceMetric

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


if __name__ == '__main__':

    #-----------------------------------------------MODEL CONFIG/PARAMS----------------------------------------#
    # MODEL CONFIG/PARAMS
    LEARNING_RATE = 1e-3
    EPOCHS = 1000
    BATCH_SIZE = 1000
    N = 300
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": None,
        "n_samples": N,
        "color": 'greyscale'
    }

    model = build_model()
    train_data, val_data = build_data_grey(BATCH_SIZE, N)

    X_test, label_test = build_test_data()
    # initialize METRICS for Tracking progress
    train_dice = DiceMetric()
    test_dice = DiceMetric()
    val_dice = DiceMetric()

    # initialize Dice LOSS for training step
    dice_loss = DiceLoss()

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
    checkpoint_directory = "./tmp/training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    quit()
    # TRAINING LOOP
    for epoch in range(EPOCHS):
        info(f'\nStart of epoch {epoch}')

        for step, (x_train, y_train) in enumerate(train_data):
            train_loss = train_step(model, x_train, y_train, optimizer)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar(
                    'train_dice', train_dice.result(), step=epoch)

        for step, (x_val, y_val) in enumerate(val_data):
            val_step(model, x_val, y_val)
            with val_summary_writer.as_default():
                tf.summary.scalar('val_dice', val_dice.result(), step=epoch)

        wandb.log({
            'train_dice': train_dice.result(),
            'train_loss': train_loss,
            'val_dice': val_dice.result()
        })
        # save checkpoint every 100 epochs
        if (epoch % 100 == 0):
            # save every 100 epochs
            checkpoint = tf.train.Checkpoint(model)
            checkpoint.save(file_prefix=checkpoint_prefix)
        # # output is numpy array
        # output = model.predict(x_test_data)
        # # output is tf object if instead output = model(x_test_data)
        output = model.predict(X_test)
        fig = make_subplots(rows=2, cols=2)
        fig.add_trace(px.imshow(output[0], binary_string=True,
                                aspect='equal').data[0], row=1, col=2)
        fig.add_trace(px.imshow(label_test[0], binary_string=True,
                                aspect='equal').data[0], row=1, col=1)
        fig.show()
        break
        # save model
    model.save(MODEL_PATH)
