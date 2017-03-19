import os
import contextlib
from setuptools import setup
import subprocess
import sys


PACKAGES = [
    'amazon_products'
]


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def download_data(root):
    p = subprocess.call(['./bin/dataset.py'] + [root])
    if p != 0:
        raise Exception('Could not download dataset')


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    with chdir(root):
        download_data(root)

        setup(
            name="pydata-amazon-products",
            version='0.1.0',
            description="Code for PyData Talk on 'Classifying Products Based on Images and Text using Keras'.",
            author='Joshua D. Loyal',
            url='https://github.com/joshloyal/pydata-amazon-products',
            license='MIT',
            install_requires=[
                'scikit-learn>=0.18.1',
                'keras>=2.0.1',
                'scipy>=0.19.0',
                'numpy>=1.12.1'
            ],
            packages=PACKAGES,
        )


if __name__ == '__main__':
    setup_package()
