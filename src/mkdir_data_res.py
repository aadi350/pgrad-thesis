"""
    This file contains the functions to make matching folders for before and after events in the data folders

    It accepts a single input argument as the new folder name and returns nothing, it contains the following functions:

        * make_data_folder - creates pair of matching folders
"""
# /usr/bin/env python3

import os
import sys
from src.models.utils import from_data, from_res


@from_data
def make_data_folder(name: str):
    """Creates matching folders in before and after data folders using @from_data decorator"""
    os.mkdir(f'before/{name}')
    os.mkdir(f'after/{name}')


if __name__ == '__main__':
    folder_name = sys.argv[-1]
    make_data_folder(folder_name)
