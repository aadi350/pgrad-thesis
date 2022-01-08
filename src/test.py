import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
import georaster

DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrag-thesis'

logging.basicConfig(format='%(asctime)s : Line: %(lineno)d - %(message)s', level = logging.INFO)

# create wrapper to use @ to set and unset data path

def read_data(func):
    def wrapper(*args, **kwargs):
        os.chdir(DATA_PATH)
        func(*args, **kwargs)
        oc.chdir(PROJ_PATH)

    return wrapper

@read_data
def read_geotiff(fp):
    logging.info(os.listdir())
    img = georaster.MultiBandRaster(fp)
    plt.imshow(img.r[:,:,2])

if __name__ == '__main__':
    read_geotiff('after/1050010027972300.tif')
