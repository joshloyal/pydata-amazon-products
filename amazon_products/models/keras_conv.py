import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten

from amazon_products.image_generators import ImageListDataGenerator


def conv_model(n_classes):
    image_features = Input(shape=(150, 150, 3), dtype='float32')

    x = Conv2D(32, (3, 3), activation='relu')(image_features)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu')(image_features)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(image_features)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[image_features], outputs=[predictions])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


data_dir = 'amazon_products_data'
image_dir = os.path.join(data_dir, 'images')

# training
df = pd.read_csv(os.path.join(data_dir, 'amazon_products_train.csv'))
train_image_list = df['image_file'].values
train_categories = df['product_category'].values

# dev
df = pd.read_csv(os.path.join(data_dir, 'amazon_products_dev.csv'))
dev_image_list = df['image_file'].values
dev_categories = df['product_category'].values

# encode labels (binary labels)
encoder = LabelBinarizer()
train_labels = encoder.fit_transform(train_categories)
dev_labels = encoder.transform(dev_categories)


scale_factor = 255.
batch_size = 128


datagen = ImageListDataGenerator(rescale=(1./scale_factor))
train_generator = datagen.flow_from_image_list(train_image_list, y=train_labels,
                                               image_dir=image_dir,
                                               target_size=(150, 150),
                                               batch_size=batch_size,
                                               shuffle=True)

dev_generator = datagen.flow_from_image_list(dev_image_list, y=dev_labels,
                                             image_dir=image_dir,
                                             target_size=(150, 150),
                                             batch_size=batch_size,
                                             shuffle=False)

model = conv_model(n_classes=4)
steps_per_epoch = int(np.ceil(train_image_list.shape[0]/float(batch_size)))
model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=dev_image_list.shape[0] // batch_size,
                    epochs=10,
                    validation_data=dev_generator)
model.save_weights('simple_conv.hdf5')
