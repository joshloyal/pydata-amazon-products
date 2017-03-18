import os
import tarfile
import urllib.request


URL = ("https://s3.amazonaws.com/amazon-products-talk/"
       "amazon_products_data.tar.gz")
ARCHIVE_NAME = "amazon_products_data.tar.gz"


def fetch_amazon_products(target_dir='.'):
    """Amazon Products Dataset."""
    opener = urllib.request.urlopen(URL)
    with open(ARCHIVE_NAME, 'wb') as f:
        f.write(opener.read())

    tarfile.open(ARCHIVE_NAME, "r:gz").extractall(path=target_dir)
    os.remove(ARCHIVE_NAME)
