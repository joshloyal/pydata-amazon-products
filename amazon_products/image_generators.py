import os

import numpy as np
import keras.backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


from amazon_products import image_utils


class ImageListIterator(image.Iterator):
    def __init__(self, image_list, y,
                 image_data_generator,
                 image_dir='',
                 target_size=(256, 256), color_mode='rgb',
                 data_format=None,
                 batch_size=32, shuffle=False, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(image_list) != len(y):
            raise ValueError

        if data_format is None:
            data_format = K.image_data_format()

        self.image_list = np.asarray(image_list)
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None

        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.image_dir = image_dir
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImageListIterator, self).__init__(len(image_list), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.image_list[j]
            img = image.load_img(os.path.join(self.image_dir, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for k in range(current_batch_size):
                img = image.array_to_img(batch_x[k], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


class ImageListDataGenerator(image.ImageDataGenerator):
    def flow_from_image_list(self, image_list, y,
                             image_dir='',
                             target_size=(256, 256), color_mode='rgb',
                             data_format=None,
                             batch_size=32, shuffle=False, seed=None,
                             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return ImageListIterator(image_list, y, self,
                                 image_dir=image_dir,
                                 target_size=target_size, color_mode=color_mode,
                                 data_format=data_format,
                                 batch_size=batch_size, shuffle=shuffle, seed=None,
                                 save_to_dir=save_to_dir, save_prefix=save_prefix,
                                 save_format=save_format)
