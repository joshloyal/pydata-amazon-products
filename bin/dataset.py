#!/usr/bin/env python

import os
import sys
import subprocess
import shutil
import tarfile
import urllib.request


URL = ("https://s3.amazonaws.com/amazon-products-talk/"
       "amazon_products_data.tar.gz")
ARCHIVE_NAME = "amazon_products_data.tar.gz"
DATA_DIR = "amazon_products_data"


def clean_path(dir_path):
    if os.path.exists(dir_path):
        print('Removing {}'.format(dir_path))
        shutil.rmtree(dir_path)


def fetch_amazon_products(target_dir='.'):
    """Amazon Products Dataset."""
    opener = urllib.request.urlopen(URL)

    print('Downloading {}'.format(URL))
    with open(ARCHIVE_NAME, 'wb') as f:
        f.write(opener.read())

    print('Extracting {}'.format(ARCHIVE_NAME))
    tarfile.open(ARCHIVE_NAME, "r:gz").extractall(path=target_dir)
    os.remove(ARCHIVE_NAME)


if __name__ == '__main__':
    file_path = os.path.abspath(os.path.dirname(__file__))

    package_path = os.path.join(file_path, '..', 'notebooks')
    data_path = os.path.join(package_path, DATA_DIR)

    if len(sys.argv[1]) > 1 and sys.argv[1] == 'clean':
        clean_path(data_path)

    # download the dataset only if it doesn't exist
    if not os.path.exists(data_path):
        fetch_amazon_products(target_dir=package_path)
