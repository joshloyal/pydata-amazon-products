import os
import functools
import pickle

import chest
import joblib
from keras.applications import resnet50
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from amazon_products import image_generators


def imagenet_scale(x):
    """ImageNet is trained with the following mean pixels.
    """
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68

    return x


def get_cache(cache_dir):
    """Return the class responsible for feature cacheing. Currently uses
    Chest.
    """
    joblib_dump = functools.partial(joblib.dump,
                                    compress=True,
                                    protocol=pickle.HIGHEST_PROTOCOL)
    return chest.Chest(path=cache_dir,
                       dump=joblib_dump,
                       load=joblib.load)


def extract_filename(file_path):
    """Extracts a file's name without the extension.

    Parameters
    ----------
    file_path: str
        The path to the image file.

    Returns
    -------
    filename: str
        The name of the filename with path and extension information
        stripped, e.g, 'path/to/file.txt' returns 'file'.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def split_cache_streams(image_list, cache):
    cache_indices, cache_files = [], []
    image_indices, image_files = [], []
    for index, image_file in enumerate(image_list):
        image_filename = extract_filename(image_file)
        if image_filename in cache:
            cache_indices.append(index)
            cache_files.append(image_filename)
        else:
            image_indices.append(index)
            image_files.append(image_file)

    return (cache_indices, cache_files), (image_indices, image_files)


def fill_from_cache(output_array, cache_files, cache_indices, cache):
    for index, cache_file in zip(cache_indices, cache_files):
        output_array[index, :] = cache[cache_file]


def write_to_cache(features, image_files, cache):
    for index, image_file in enumerate(image_files):
        image_filename = extract_filename(image_file)
        feature = features[index, :]
        cache[image_filename] = feature


def extract_resnet50_features(image_list,
                              image_dir='',
                              scale_factor=255.,
                              pooling=None,
                              batch_size=500,
                              include_top=False):
    """Vectorize images by passing them through the ResNet50 architecture.

    The weight's of the ResNet model are pre-trained on ImageNet. The underlying
    model use's a Kera's implamentation. See `keras.applications.ResNet50` for
    further details.

    Parameters
    ----------
    image_list : list of str
        A list containing the image files to vectorize. Can either
        be the full path to the file on disk or just the file name.
        If only the filename is provided then use the `image_dir`
        parameter to specify the full path.
    image_dir : str (default='')
        The directory where the images are located.
    scale_factor : float (default=255)
        Value to scale all pixels. For RGB images this should be
        the maximum value (255) so that all values are between zero and one.
    pooling : str (default=None)
        Optional pooling mode for feature extraction.
            - `None` means that the output of the model will be a 4D output
               of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
              output of the last convolutional layer, and thus the output
              of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
    batch_size : int (default=500)
        The size of the mini-batches to pass through the ResNet architecture.
    include_top : bool (default=False)
        Whether to include the top classification layer.
    """
    if pooling not in (None, 'avg', 'max'):
        raise ValueError('Invalid pooling type `{}`.'
                         'Pooling should be `None`, `avg`, or `max`.')

    image_size = 224
    n_channels = 3
    model = resnet50.ResNet50(include_top=include_top,
                              weights='imagenet',
                              pooling=pooling,
                              input_shape=(image_size, image_size, n_channels))

    datagen = image_generators.ImageListDataGenerator(
        preprocessing_function=imagenet_scale)
    generator = datagen.flow_from_image_list(image_list, y=None,
                                             image_dir=image_dir,
                                             target_size=(image_size, image_size),
                                             batch_size=batch_size,
                                             shuffle=False)

    # Keras 2 uses steps.. this makes it difficult to process exactly
    # all the entries so we may predict on a little more and then truncate
    steps = int(np.ceil(len(image_list) / float(batch_size)))
    return model.predict_generator(generator, steps=steps)[:len(image_list)]


class ResNetVectorizer(BaseEstimator, TransformerMixin):
    """Simple scikit-learn style transform that passes images through a
    resnet model. Has an option to cache the images for subsequent calls
    to the transformer.

    Parameters
    ----------
    pooling : str (default=None)
        Optional pooling mode for feature extraction.
            - `None` means that the output of the model will be a 4D output
               of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
              output of the last convolutional layer, and thus the output
              of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
    batch_size : int (default=500)
        The size of the mini-batches to pass through the ResNet architecture.
    image_dir : str (default='')
        The directory where the images are located.
    use_cache : bool (default=False)
        Whether to write the features to a on-disk cache.
    cache_dir : str (default=None)
        The directory to use for the cache.
    """
    def __init__(self,
                 pooling=None,
                 batch_size=500,
                 include_top=False,
                 image_dir='',
                 use_cache=False,
                 cache_dir=None):
        self.pooling = pooling
        self.batch_size = batch_size
        self.include_top = include_top
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache

    def clear_cache(self):
        """Clear the persistent cache."""
        cache = get_cache(self.cache_dir)
        cache.drop()

    def fit(self, image_files):
        """Currently this is a No-Op."""
        if self.cache_dir and os.path.exists(self.cache_dir):
            raise ValueError(
                'Cache {} already exists! '
                'If you mean to overwrite this cache, then call '
                '`clear_cache` before fitting!'.format(self.cache_dir))

        return self

    def transform(self, image_files):
        """Pass images through Resnet50.

        Parameters
        ----------
        image_list : list of str
            A list containing the image files to vectorize. Can either
            be the full path to the file on disk or just the file name.
            If only the filename is provided then use the `image_dir`
            parameter to specify the full path.
        """
        n_samples = len(image_files)

        if self.use_cache:
            cache = get_cache(self.cache_dir)
            (cache_indices, cache_files), (image_indices, image_files) = (
                split_cache_streams(image_files, cache))

            if self.include_top:
                output_shape = (n_samples, 1000)
            else:
                output_shape = (n_samples, 2048)
            output = np.zeros(output_shape, dtype=np.float32)

            if image_files:
                features = extract_resnet50_features(image_files,
                                                     pooling=self.pooling,
                                                     batch_size=self.batch_size,
                                                     include_top=self.include_top,
                                                     image_dir=self.image_dir)
                output[image_indices, :] = np.squeeze(features)

                if self.use_cache:
                    write_to_cache(features, image_files, cache)

            if cache_files:
                fill_from_cache(output, cache_files, cache_indices, cache)
        else:
            output = extract_resnet50_features(image_files,
                                               pooling=self.pooling,
                                               batch_size=self.batch_size,
                                               image_dir=self.image_dir)
            output = np.squeeze(output)

        return output
