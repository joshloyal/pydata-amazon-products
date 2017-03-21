import numpy as np
from joblib import Parallel, delayed
from skimage import color


from . import image_utils


def hsv_features_single(image, agg_func=np.mean, background=None):
    image = np.asarray(image, dtype=np.uint8)
    hsv_image = color.rgb2hsv(image)

    if background is not None:
        h_channel = hsv_image[:, :, 0]
        h_mean = agg_func(
            np.ma.array(h_channel, mask=(h_channel == background[0]))
        )
        h_mean = background[0] if h_mean is np.ma.masked else h_mean

        s_channel = hsv_image[:, :, 1]
        s_mean = agg_func(
            np.ma.array(s_channel, mask=(s_channel == background[1]))
        )
        s_mean = background[1] if s_mean is np.ma.masked else s_mean

        v_channel = hsv_image[:, :, 2]
        v_mean = agg_func(
            np.ma.array(v_channel, mask=(v_channel == background[2]))
        )
        v_mean = background[2] if v_mean is np.ma.masked else v_mean
    else:
        h_mean = agg_func(hsv_image[:, :, 0])
        s_mean = agg_func(hsv_image[:, :, 1])
        v_mean = agg_func(hsv_image[:, :, 2])

    return h_mean, s_mean, v_mean


def array_to_hsv(image_list, mode='mean', background=None, n_jobs=1):
    """A useful ordering tool is the HSV values of an RGB image.
    In particular, H (hue) will order by color.
    """
    if background == 'white':
        background = np.array([0, 0, 1], dtype=np.uint8)
    elif background == 'black':
        background = np.array([0, 0, 0], dtype=np.uint8)


    if mode == 'mean':
        agg_func = np.mean
    elif mode == 'median':
        agg_func = np.median
    else:
        raise ValueError("Unkown mode `{}`.".format(mode))

    result = Parallel(n_jobs=n_jobs)(
        delayed(hsv_features_single)(image, agg_func, background)
        for image in image_list)

    return np.vstack(result)


def mean_hsv(image_column,
             data,
             mode='mean',
             background=None,
             image_directory='',
             n_jobs=-1):
    images = image_utils.load_images(
        data[image_column],
        image_dir=image_directory,
        as_image=True,
        n_jobs=n_jobs)

    return array_to_hsv(
        images, mode=mode, background=background, n_jobs=n_jobs)
