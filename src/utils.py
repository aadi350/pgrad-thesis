import os
import logging
import numpy as np

DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrad-thesis'

logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)

# create wrapper to use @ to set and unset data path


def from_data(func):
    def wrapper(*args, **kwargs):
        os.chdir(DATA_PATH)
        ret = func(*args, **kwargs)
        os.chdir(PROJ_PATH)
        return ret
    return wrapper


def from_res(func):
    def wrapper(*args, **kwargs):
        os.chdir(RES_PATH)
        ret = func(*args, **kwargs)
        os.chdir(PROJ_PATH)
        return ret
    return wrapper


def to_rgb(img: np.array):
    '''Converts given image to RHB and normalized to [0,1]'''
    raise NotImplementedError


if __name__ == '__main__':
    raise NotImplementedError('This file is not meant be run standalone')
