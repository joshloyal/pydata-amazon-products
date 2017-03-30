import contextlib
import os
import shutil
import tempfile

import pytest
import numpy as np
import PIL.Image as pil_image

from amazon_products import image_generators as image_gen


@contextlib.contextmanager
def tempdir(dir=None):
    """Context manager that creates a temporary directory using tempfile.
    The directory is deleted after exiting the context."""
    dirpath = tempfile.mkdtemp(dir=dir)

    def cleanup():
        shutil.rmtree(dirpath)

    try:
        yield dirpath
    finally:
        cleanup()



def get_images(n_samples=100, img_width=20, img_height=20, random_state=123):
    rng = np.random.RandomState(random_state)

    images = []
    for _ in range(n_samples):
        bias = rng.rand(img_width, img_height, 1) * 64
        variance = rng.rand(img_width, img_height, 1) * (255 - 64)
        img = rng.rand(img_width, img_height, 3) * variance + bias
        img = pil_image.fromarray(img.astype(np.uint8)).convert('RGB')
        images.append(img)

    return images


def get_image_list(image_directory, n_samples=100, dir=None, random_state=123):
    images = get_images(n_samples=n_samples, random_state=random_state)

    image_list = []
    for index, image in enumerate(images):
        image_file = 'image-{}.jpg'.format(index)
        image_path = os.path.join(image_directory, image_file)
        image.save(image_path)

        image_list.append(image_file)

    return image_list


@pytest.fixture(scope='module')
def image_data():
    with tempdir() as image_dir:
        image_list = get_image_list(image_directory=image_dir, n_samples=20)
        yield image_dir, image_list


def test_image_generator(image_data):
    image_dir, image_list = image_data

    generator = image_gen.ImageListDataGenerator(
        rescale=1./255.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    iterator = generator.flow_from_image_list(image_list,
                                              y=None,
                                              image_dir=image_dir)

    next(iterator)
