import numpy as np
import pandas as pd
import rasterio as rs
import cuspatial as cx
import os
import logging
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from osgeo import gdal
from utils import from_data
from PIL import Image
from skimage import io
from cucim import CuImage
import json
import pprint

io.use_plugin('pil')
matplotlib.use('TkAgg')

pp = pprint.PrettyPrinter(indent=4)

DATA_PATH = '/home/aadi/projects/pgrad-thesis/data'
RES_PATH = '/home/aadi/projects/pgrad-thesis/results'
PROJ_PATH = '/home/aadi/projects/pgrag-thesis'

logging.basicConfig(format='%(asctime)s : Line: %(lineno)d - %(message)s', level = logging.INFO)

@from_data
def read_geotiff(fp):
    #img = rasterio.open(fp)
    img = CuImage(fp)
    logging.info(img)
    return img

img = CuImage('./data/before/pakistan_earthquake_2021/10300100BB64AB00.tif')
logging.info(img.is_loaded)        # True if image data is loaded & available.
logging.info(img.device)           # A device type.
logging.info(img.ndim)             # The number of dimensions.
logging.info(img.dims)             # A string containing a list of dimensions being requested.
logging.info(img.shape)            # A tuple of dimension sizes (in the order of `dims`).
logging.info(img.size('XYC'))      # Returns size as a tuple for the given dimension order.
logging.info(img.dtype)            # The data type of the image.
logging.info(img.channel_names)    # A channel name list.
logging.info(img.spacing())        # Returns physical size in tuple.
logging.info(img.spacing_units())  # Units for each spacing element (size is same with `ndim`).
logging.info(img.origin)           # Physical location of (0, 0, 0) (size is always 3).
logging.info(img.direction)        # Direction cosines (size is always 3x3).
logging.info(img.coord_sys)        # Coordinate frame in which the direction cosines are
                            # measured. Available Coordinate frame is not finalized yet.

# Returns a set of associated image names.
logging.info(img.associated_images)
# Returns a dict that includes resolution information.
logging.info(json.dumps(img.resolutions, indent=2))
# A metadata object as `dict`
logging.info(json.dumps(img.metadata, indent=2))
# A raw metadata string.
pp.pprint(img.raw_metadata)
