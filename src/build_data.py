import sys
from cucim.skimage.transform import resize
from distutils.debug import DEBUG
import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from models.utils import show_progress
import cv2 as cv
import skimage
from tqdm import tqdm
from difference_functions import basic_subtract

logging.basicConfig(level=DEBUG)
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'


def _normalize(images, input_mask):
    before_image, after_image = images
    before_image = tf.cast(before_image, tf.float32) / 255.0
    after_image = tf.cast(after_image, tf.float32) / 255.0
    input_mask -= 1
    return (before_image, after_image), input_mask


def difference_data(data_path, difference_function=None):
    '''Assume three folders exist in data_path: time1, time2, label'''

    if not difference_function:
        difference_function = basic_subtract

    assert 'differenced' not in os.listdir(
        data_path), 'Differenced folder exists'
    assert set(os.listdir(data_path)) == set(
        ['time1', 'time2', 'label']), "Folders not present OR too many folders"

    before_path = os.path.join(data_path, 'time1')
    after_path = os.path.join(data_path, 'time2')
    label_path = os.path.join(data_path, 'label')

    assert len(os.listdir(before_path)) == len(os.listdir(after_path)
                                               ), "Length of before images different from length of after images"

    os.mkdir(os.path.join(data_path, 'differenced'))

    before_image_list = os.listdir(before_path)
    after_image_list = os.listdir(after_path)
    label_image_list = os.listdir(label_path)

    for i in tqdm(range(len(os.listdir(before_path)))):
        before_image_path = os.path.join(
            data_path, 'time1', before_image_list[i])
        after_image_path = os.path.join(
            data_path, 'time2', after_image_list[i])
        label_image_path = os.path.join(
            data_path, 'label', label_image_list[i])
        before_img = skimage.io.imread(before_image_path)
        after_img = skimage.io.imread(after_image_path)

        difference = difference_function(before_img, after_img)
        skimage.io.imsave(os.path.join(
            *[data_path, 'differenced', f'{i}.png']), difference)


def build_differenced_data(N=-1, batch_size=100, train=True, val=False, test=False):
    '''Assumes differenced images are in data/train/differenced, data/val/differenced, data/test/differenced and loads into single tf.data.Dataset with labels,
        returns tf.data.Dataset of (differenced, labels)
    '''
    autotune = tf.data.AUTOTUNE

    if train:
        main_dir = TRAIN_DIR
    elif val:
        main_dir = VAL_DIR
    else:
        main_dir = TEST_DIR

    diff_list = os.listdir(main_dir + '/differenced')[:N]
    label_list = os.listdir(main_dir + '/label')[:N]

    diff_list = [np.array(Image.open(main_dir + '/differenced/' + fname))
                 for fname in diff_list]
    label_list = [np.array(Image.open(main_dir + '/label/' + fname))
                  for fname in label_list]

    diff_list = [tf.convert_to_tensor(
        image, dtype=tf.float32) for image in diff_list]
    label_list = [tf.convert_to_tensor(
        image, dtype=tf.float32) for image in label_list]

    ds = tf.data.Dataset.from_tensor_slices((diff_list, label_list))

    del diff_list
    del label_list

    return ds.batch(batch_size).prefetch(buffer_size=autotune)

# Uses too much memory


def _build_train_arrays(N=-1):
    raise MemoryError(
        f'{_build_train_arrays.__name__} exceeds Protobuf memory usage')
    train_before_list = os.listdir(TRAIN_DIR + '/time1')[:N]
    train_after_list = os.listdir(TRAIN_DIR + '/time2')[:N]
    train_label_list = os.listdir(TRAIN_DIR + '/label')[:N]
    before_train = np.array([np.array(Image.open(TRAIN_DIR + '/time1/' + fname))
                            for fname in tqdm(train_before_list)])
    after_train = np.array([np.array(Image.open(TRAIN_DIR + '/time2/' + fname))
                            for fname in tqdm(train_after_list)])
    label_train = np.array([np.array(Image.open(TRAIN_DIR + '/label/' + fname))
                            for fname in tqdm(train_label_list)])

    return before_train, after_train, label_train


def _build_val_arrays(N=-1):
    raise MemoryError(
        f'{_build_val_arrays} exceeds Protobuf memory usage')
    val_before_list = os.listdir(VAL_DIR + '/time1')[:N]
    val_after_list = os.listdir(VAL_DIR + '/time2')[:N]
    val_label_list = os.listdir(VAL_DIR + '/label')[:N]
    before_val = np.array([np.array(Image.open(VAL_DIR + '/time1/' + fname))
                           for fname in show_progress(val_before_list)])
    after_val = np.array([np.array(Image.open(VAL_DIR + '/time2/' + fname))
                          for fname in show_progress(val_after_list)])
    label_val = np.array([np.array(Image.open(VAL_DIR + '/label/' + fname))
                          for fname in show_progress(val_label_list)])

    return before_val, after_val, label_val


def build_diff_data_rgb(batch_size, N=-1, image_shape=None):
    before_train, after_train, train_label = _build_train_arrays(N=N)
    before_val, after_val, val_label = _build_val_arrays(N=N)

    train_diff = np.array([basic_subtract(b, a)
                           for (b, a) in zip(before_train, after_train)])
    val_diff = np.array([basic_subtract(b, a)
                         for (b, a) in zip(before_val, after_val)])

    if image_shape:
        # reshape if pretrained weights expect different input sizes
        train_diff = [cv.resize(img, image_shape) for img in train_diff]
        val_diff = [cv.resize(img, image_shape) for img in val_diff]
        train_label = [cv.resize(img, image_shape) for img in train_label]
        val_label = [cv.resize(img, image_shape) for img in val_label]

    train_data = tf.data.Dataset.from_tensor_slices(
        (train_diff, train_label)).batch(batch_size=batch_size)
    val_data = tf.data.Dataset.from_tensor_slices(
        (val_diff, val_label)).batch(batch_size=batch_size)

    return train_data, val_data


def _normalize(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img


def decode_grey(img):
    img = tf.io.decode_png(img, channels=1)
    return img


def decode_rgb(img):
    img = tf.io.decode_png(img, channels=3)
    return img


def process_path_rgb(fp):
    img = tf.io.read_file(fp)
    img = decode_rgb(img)
    img = _normalize(img)
    return img


def process_path_grey(fp):
    img = tf.io.read_file(fp)
    img = decode_grey(img)
    return img


def build_data_horizontal_separate(batch_size, buffer_size=1000):

    def train_gen(split='train', data_path='data/'):
        path = data_path + split
        for t1, t2, l in zip(sorted(os.listdir(path+'/time1')), sorted(os.listdir(path+'/time2')), sorted(os.listdir(path+'/label'))):
            # get full paths

            t1 = process_path_rgb(f'data/{split}/time1/' + t1)
            t2 = process_path_rgb(f'data/{split}/time2/' + t2)
            l = process_path_grey(f'data/{split}/label/' + l)

            yield (t1, t2), l

    def val_gen(split='val', data_path='data/'):
        path = data_path + split
        for t1, t2, l in zip(sorted(os.listdir(path+'/time1')), sorted(os.listdir(path+'/time2')), sorted(os.listdir(path+'/label'))):
            # get full paths

            t1 = process_path_rgb(f'data/{split}/time1/' + t1)
            t2 = process_path_rgb(f'data/{split}/time2/' + t2)
            l = process_path_grey(f'data/{split}/label/' + l)

            yield (t1, t2), l

    train_ds = tf.data.Dataset.from_generator(
        train_gen, output_types=((tf.float32, tf.float32), tf.uint8))
    val_ds = tf.data.Dataset.from_generator(
        val_gen, output_types=((tf.float32, tf.float32), tf.uint8))

    train_batches = (
        train_ds
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_batches = (
        val_ds
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    return train_batches, val_batches


def build_data_grey(batch_size, N=-1):

    before_train, after_train, label_train = _build_train_arrays(N=N)
    before_val, after_val, label_val = _build_val_arrays(N=N)

    train_diff = np.array([basic_subtract(b, a)
                           for (b, a) in zip(before_train, after_train)])
    single_grey = [cv.cvtColor(i, cv.COLOR_RGB2GRAY) for i in train_diff]

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


if __name__ == '__main__':
    raise NotImplementedError
