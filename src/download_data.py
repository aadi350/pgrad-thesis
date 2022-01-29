"""
    Functions for downloading files from list of URLs defined in event-specific events at the same time in before and after data folders

    Functions defined are:
        * download_from_list
"""
# /usr/bin/env python3
import sys
import os
from src.models.utils import from_data
from tqdm import tqdm
import urllib.request
import progressbar
import urllib.request


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


@from_data
def download_from_list(event_name: str) -> None:
    """Downloads from inline file list"""
    with open(f'after/{event_name}/filelist.txt', 'r') as f:
        for url in tqdm(f.readlines()):
            url = url.split()[0]
            filename = url.split('/')[-1]
            print(filename)
            urllib.request.urlretrieve(
                url, f'after/{event_name}/{filename}', show_progress)

    with open(f'before/{event_name}/filelist.txt', 'r') as f:
        for url in tqdm(f.readlines()):
            url = url.split()[0]
            filename = url.split('/')[-1]
            print(filename)
            urllib.request.urlretrieve(
                url, f'after/{event_name}/{filename}', show_progress)


if __name__ == '__main__':
    download_from_list(sys.argv[-1])
