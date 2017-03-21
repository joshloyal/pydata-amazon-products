from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import glob
import os
import functools
import itertools

import numpy as np
from joblib import Parallel, delayed
from PIL import Image as pil_image


#from .image_features import hsv_features


image_extensions = {'jpg', 'jpeg', 'png'}


def image_path(image_file, image_dir=''):
    return os.path.join(image_dir, image_file)


def sample_images(seq, n_samples, seed=123):
    random_state = np.random.RandomState(seed)
    return random_state.choice(seq, size=n_samples, replace=False)


def load_image(image_file,
               image_dir='',
               target_size=None,
               dtype=np.uint8,
               as_image=False):
    """Loads an image into PIL format."""
    image_loc = image_path(image_file, image_dir=image_dir)
    img = pil_image.open(image_loc).convert('RGB')

    if target_size:
        img = img.resize((target_size[1], target_size[0]), pil_image.ANTIALIAS)

    if as_image:
        return img

    return np.expand_dims(np.asarray(img, dtype), 0)


def load_images(image_files,
                image_dir='',
                n_samples=None,
                target_size=(128, 128),
                dtype=np.uint8,
                as_image=False,
                random_state=123,
                n_jobs=1):
    if n_samples is not None and n_samples < len(image_files):
        image_files = sample_images(image_files, n_samples, seed=random_state)

    # perform this in parallel with joblib
    images = Parallel(n_jobs=n_jobs)(
                delayed(load_image)(img,
                                    image_dir=image_dir,
                                    target_size=target_size,
                                    as_image=as_image,
                                    dtype=dtype)
                for img in image_files)

    if as_image:
        return images

    return np.vstack(images)


def image_glob_pattern(image_directory, ext):
    return os.path.join(image_directory, '*.' + ext)


def image_glob(image_directory, ext):
    return glob.glob(image_glob_pattern(image_directory, ext))


def load_from_directory(image_directory,
                        n_samples=None,
                        dtype=np.uint8,
                        as_image=False,
                        random_state=123,
                        n_jobs=1,):
    image_files = list(itertools.chain.from_iterable(
        [image_glob(image_directory, ext) for ext in image_extensions]))
    return load_images(image_files,
                       n_samples=n_samples,
                       dtype=dtype,
                       as_image=as_image,
                       random_state=random_state,
                       n_jobs=n_jobs)


def images_to_sprite(images):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    images : list
        A List of PIL Image objects.

    Returns
    -------
    A properly shaped NxWx3 PIL Image with any necessary padding.
    """
    n_samples = len(images)

    #features = hsv_features(images, background='white', n_jobs=-1)
    #image_order = np.argsort(features[:, 0])

    if n_samples < 1:
        raise ValueError('Cannot create a sprite image from zero images.')

    image_width, image_height = images[0].size

    # sprite image should be sqrt(n_samples) x sqrt(n_samples). If
    # n_samples is not a perfect square then we pad with white images.
    table_size = int(np.ceil(np.sqrt(n_samples)))

    # create the new image. Hard-code the background color to white
    background_color = (255, 255, 255)
    sprite_size = (table_size * image_width, table_size * image_height)
    sprite_image = pil_image.new('RGB', sprite_size, background_color)

    # loop through the images and add them to the sprite image
    for index, image in enumerate(images):
    #for index, image_index in enumerate(image_order):
    #    image = images[image_index]
        # Determine where we are in the sprite image.
        row_index = int(index / table_size)
        column_index = index % table_size

        # determine the bounding box of the image (where it is)
        left = column_index * image_width
        right = left + image_width
        upper = row_index * image_height
        lower = upper + image_height
        bounding_box = (left, upper, right, lower)

        sprite_image.paste(image, bounding_box)

    return sprite_image


def directory_to_sprites(image_directory,
                         n_samples=None,
                         random_state=123,
                         n_jobs=1):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_directory : str
        Path to the directory holding the images.

    n_samples : int (default=None)
        The number of random sample images to use. If None, then
        all images are loaded. This can be memory expensive.

    as_image : bool (default=False)
        Whether to return a PIL image otherwise return a numpy array.

    random_state : int (default=123)
        The seed to use for the random sampling.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    images = load_from_directory(
        image_directory,
        n_samples=n_samples,
        dtype=np.float32,
        as_image=True,
        random_state=random_state,
        n_jobs=n_jobs)

    return images_to_sprite(images)


def column_to_sprites(image_column,
                      sort_by=None,
                      data=None,
                      image_directory='',
                      n_samples=None,
                      random_state=123,
                      n_jobs=1):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_column : str
        Column name corresponding to the images.

    sort_by : str
        Column to sort by.

    data : pd.DataFrame
        Pandas dataframe holding the dataset.

    image_directory : str (default='')
        The location of the image files on disk.

    n_samples : int (default=None)
        The number of random sample images to use. If None, then
        all images are loaded. This can be memory expensive.

    as_image : bool (default=False)
        Whether to return a PIL image otherwise return a numpy array.

    random_state : int (default=123)
        The seed to use for the random sampling.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n=n_samples,
                           replace=False,
                           random_state=random_state)

    if sort_by is not None:
        data = data.sort_values(by=sort_by, ascending=True)

    images = load_images(
        data[image_column],
        image_dir=image_directory,
        as_image=True,
        n_jobs=n_jobs)

    return images_to_sprite(images)


def image_histogram(image_column,
                    groupby_column,
                    data,
                    image_directory='',
                    n_samples=None,
                    n_jobs=1,
                    random_state=123):
    """Create an image histogram binned by the `groupby_column`.

    Parameters
    ----------
    image_column : str
        Name of the column pointing to the image files

    groupby_column : str
        Name of the column to groupby

    data : pandas.DataFrame
        The dataframe where both columns are present.

    image_directory : str
        Path to the directory holding the images.
    """
    groupby = data.groupby(by=groupby_column)[image_column]
    height = None

    # we need to bin by group then bin within group
    bins = []
    for group_name, group_data in groupby:
        images = load_images(
            group_data,
            image_dir=image_directory,
            n_samples=n_samples,
            as_image=True,
            target_size=(50, 50),
            random_state=random_state,
            n_jobs=n_jobs)
        bins.append((group_name, images))

        # keep track of maximum height as well
        if height is None or len(images) > height:
            height = len(images)

    # get information to actually generate the plot
    n_bins = len(bins)
    bar_padding = 5
    image_size = 50
    hist_height = (image_size + bar_padding) * height
    hist_width = (image_size + bar_padding) * n_bins

    background_color = (0, 0, 0)  # hard code to black for now
    hist_size = (hist_width, hist_height)
    hist_image = pil_image.new('RGB', hist_size, background_color)

    for bin_index, (bin_name, bin_data) in enumerate(bins):
        y_coord = hist_height
        x_coord = bin_index * (image_size + bar_padding)
        for image in bin_data:
            #image.thumbnail((image_size, image_size), pil_image.ANTIALIAS)
            hist_image.paste(image, (x_coord, y_coord))
            y_coord -= (image_size + bar_padding)

    return hist_image

