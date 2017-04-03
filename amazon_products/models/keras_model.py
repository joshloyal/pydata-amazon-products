import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Input, Dense, Dropout

from amazon_products.resnet import ResNetVectorizer


data_dir = 'amazon_products_data'
image_dir = os.path.join(data_dir, 'images')
cache_dir = 'resnet50'

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

# get features from a pre-trained resnet model
vec = ResNetVectorizer(batch_size=500,
                       image_dir=image_dir,
                       use_cache=True,
                       cache_dir=cache_dir)
train_features = vec.transform(train_image_list)
dev_features = vec.transform(dev_image_list)

# fine-tune the last layer
input_features = Input(shape=train_features.shape[1:], dtype='float32')

x = Dense(256, activation='relu')(input_features)
x = Dropout(0.5)(x)
predictions = Dense(encoder.classes_.shape[0], activation='softmax')(x)

model = Model(inputs=[input_features], outputs=[predictions])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_features, train_labels,
          nb_epoch=50, batch_size=32,
          validation_data=[dev_features, dev_labels])
