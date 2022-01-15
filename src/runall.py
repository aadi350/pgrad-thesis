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

io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)

DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrag-thesis'

logging.basicConfig(
    format='%(asctime)s : Line: %(lineno)d - %(message)s', level=logging.INFO)


@from_data
def read_geotiff(fp):
    logging.info(os.listdir())
    img = rasterio.open(fp)
    # show(img)
    arr = img.read()
    logging.info(img)
    return img


img = read_geotiff(
    'before/earthquake/2021_pakistan/10300100BB64AB00.tif')
logging.info(img.is_loaded)        # True if image data is loaded & available.
logging.info(img.device)           # A device type.
logging.info(img.ndim)             # The number of dimensions.
# A string containing a list of dimensions being requested.
logging.info(img.dims)
# A tuple of dimension sizes (in the order of `dims`).
logging.info(img.shape)
# Returns size as a tuple for the given dimension order.
logging.info(img.size('XYC'))
logging.info(img.dtype)            # The data type of the image.
logging.info(img.channel_names)    # A channel name list.
logging.info(img.spacing())        # Returns physical size in tuple.
# Units for each spacing element (size is same with `ndim`).
logging.info(img.spacing_units())
# Physical location of (0, 0, 0) (size is always 3).
logging.info(img.origin)
logging.info(img.direction)        # Direction cosines (size is always 3x3).
# Coordinate frame in which the direction cosines are
logging.info(img.coord_sys)
# measured. Available Coordinate frame is not finalized yet.

# Returns a set of associated image names.
logging.info(img.associated_images)
# Returns a dict that includes resolution information.
logging.info(json.dumps(img.resolutions, indent=2))
# A metadata object as `dict`
logging.info(json.dumps(img.metadata, indent=2))
# A raw metadata string.
pp.pprint(img.raw_metadata)
