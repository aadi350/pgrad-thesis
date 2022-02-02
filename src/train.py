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
from build_data import build_data_grey, build_data_rgb, build_differenced_data, build_test_data
from models.model import build_model
from models.segnet import INPUT_SHAPE, build_segnet
from losses import DiceLoss
from metrics import DiceMetric


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

sys.path.append('/home/aadi/projects/pgrad-thesis/src/models')

# LOGGING/CONFIG FOR TF
wandb.init(mode='disabled', project="pgrad-thesis",
           entity="aadi350", tags=['test'])


io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)


logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)


if __name__ == '__main__':

    #-----------------------------------------------MODEL CONFIG/PARAMS----------------------------------------#
    # MODEL CONFIG/PARAMS
    LEARNING_RATE = 1e-3
    STEPS_PER_EPOCH = 100
    EPOCHS = 1000
    BATCH_SIZE = 25
    TRAIN_SIZE = 300
    TEST_SIZE = 200
    # ensure clean divides
    assert TRAIN_SIZE % BATCH_SIZE == 0
    assert TEST_SIZE % BATCH_SIZE == 0
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": None,
        "n_samples": TRAIN_SIZE,
        "color": 'greyscale'
    }

    # model = build_model()
    model = build_segnet()
    # train_data, val_data = build_data_grey(BATCH_SIZE, N)
    #train_data, val_data = build_data_rgb(BATCH_SIZE, TRAIN_SIZE, (256, 256))
    train_data = build_differenced_data(N=500, batch_size=100, train=True)
    val_data = build_differenced_data(N=500, batch_size=100, train=False, val=True, test=False)

    X_test, label_test = build_test_data(BATCH_SIZE, TEST_SIZE)
    # initialize METRICS for Tracking progress
    train_dice = DiceMetric()
    test_dice = DiceMetric()
    val_dice = DiceMetric()

    # initialize Dice LOSS for training step
    # dice_loss = DiceLoss()
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
            # loss = dice_loss(labels, logits)
            loss = bce_loss(labels, logits)
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

    model.compile(
        loss=bce_loss, 
        optimizer=optimizer,
        metrics=[DiceMetric()])
    
    info(model.summary())

    callbacks = [] # TODO set this up for logging
    model.fit(train_data,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=callbacks,)

    
    model.save(MODEL_PATH)
