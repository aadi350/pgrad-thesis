import os
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import show_progress
import cv2 as cv
from difference_functions import basic_subtract

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'


def build_data(batch_size, N=-1):
    train_before_list = os.listdir(TRAIN_DIR + '/time1')[:N]
    train_after_list = os.listdir(TRAIN_DIR + '/time2')[:N]
    train_label_list = os.listdir(TRAIN_DIR + '/label')[:N]
    before_train = np.array([np.array(Image.open(TRAIN_DIR + '/time1/' + fname))
                            for fname in show_progress(train_before_list)])
    after_train = np.array([np.array(Image.open(TRAIN_DIR + '/time2/' + fname))
                            for fname in show_progress(train_after_list)])
    label_train = np.array([np.array(Image.open(TRAIN_DIR + '/label/' + fname))
                            for fname in show_progress(train_label_list)])
    train_diff = np.array([basic_subtract(b, a)
                           for (b, a) in zip(before_train, after_train)])
    single_grey = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in train_diff]

    val_before_list = os.listdir(VAL_DIR + '/time1')[:N]
    val_after_list = os.listdir(VAL_DIR + '/time2')[:N]
    val_label_list = os.listdir(VAL_DIR + '/label')[:N]
    before_val = np.array([np.array(Image.open(VAL_DIR + '/time1/' + fname))
                           for fname in show_progress(val_before_list)])
    after_val = np.array([np.array(Image.open(VAL_DIR + '/time2/' + fname))
                          for fname in show_progress(val_after_list)])
    label_val = np.array([np.array(Image.open(VAL_DIR + '/label/' + fname))
                          for fname in show_progress(val_label_list)])

    val_diff = np.array([basic_subtract(b, a)
                         for (b, a) in zip(before_val, after_val)])

    single_grey_val = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in val_diff]
    val_data = tf.data.Dataset.from_tensor_slices(
        (single_grey_val, label_val)).batch(batch_size=batch_size)
    train_data = tf.data.Dataset.from_tensor_slices(
        (single_grey, label_train)).batch(batch_size=batch_size)

    return (train_data, val_data)


def build_test_data(batch_size=1000, N=-1, return_labels_as_array=True):
    test_before_list = os.listdir(TEST_DIR + '/time1')[:N]
    test_after_list = os.listdir(TEST_DIR + '/time2')[:N]
    test_label_list = os.listdir(TEST_DIR + '/label')[:N]
    before_test = np.array([np.array(Image.open(TEST_DIR + '/time1/' + fname))
                            for fname in show_progress(test_before_list)])
    after_test = np.array([np.array(Image.open(TEST_DIR + '/time2/' + fname))
                           for fname in show_progress(test_after_list)])
    label_test = np.array([np.array(Image.open(TEST_DIR + '/label/' + fname))
                           for fname in show_progress(test_label_list)])
    test_diff = np.array([basic_subtract(b, a)
                          for (b, a) in zip(before_test, after_test)])
    single_grey_test = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in test_diff]
    x_test_data = tf.data.Dataset.from_tensor_slices(
        single_grey_test).batch(batch_size)
    y_test_data = tf.data.Dataset.from_tensor_slices(label_test)

    if return_labels_as_array:
        # don't load into TF dataset if flag is set
        return (x_test_data, label_test)
    return (x_test_data, y_test_data)
