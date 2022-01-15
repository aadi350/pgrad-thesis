from re import I
import cupy as cp
import numpy as np
import os
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
# from osgeo import gdal
import rasterio
from utils import from_data
from PIL import Image
from skimage import io
from skimage.io import imshow
from rasterio.plot import show
from cucim import CuImage
import json
import pprint
import cucim
from logging import log, info, debug
from PIL import Image

io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)

DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrag-thesis'

logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)


@from_data
def read_geotiff(fp: str) -> cp.array:
    """Reads geotiff from a file-path and returns cupy array"""
    logging.info(fp)

    img = rasterio.open(fp)
    return cp.array(img.read())


img = read_geotiff(
    'before/earthquake/2021_pakistan/10300100BB64AB00.tif')
