import os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from amazon_products.resnet import ResNetVectorizer
from amazon_products.text_generators import sparse_batch_generator


def image_text_model(image_features, text_features, n_classes):
    # fine-tune the last layer
    image_features = Input(shape=image_features.shape[1:], dtype='float32')

    n_text_features = text_features.shape[1]
    text_features = Input(shape=text_features.shape[1:], dtype='float32')

    # text model
    x_text = Dense(256, activation='elu', kernel_regularizer=l2(1e-5))(text_features)
    x_text = Dropout(0.5)(x_text)

    # image model
    x_img = Dense(256, activation='elu')(image_features)
    x_img = Dropout(0.5)(x_img)
    x_img = Dense(256, activation='elu')(x_img)
    x_img = Dropout(0.5)(x_img)

    merged = concatenate([x_img, x_text])
    predictions = Dense(n_classes, activation='softmax')(merged)

    model = Model(inputs=[image_features, text_features], outputs=[predictions])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

data_dir = 'amazon_products_data'
image_dir = os.path.join(data_dir, 'images')
cache_dir = 'resnet50'
n_samples = 5000

# training
df = pd.read_csv(os.path.join(data_dir, 'amazon_products_train.csv'))
train_image_list = df['image_file'].values
train_text = df['title'].values.tolist()
train_categories = df['product_category'].values

# dev
df = pd.read_csv(os.path.join(data_dir, 'amazon_products_dev.csv'))
dev_image_list = df['image_file'].values
dev_text = df['title'].values.tolist()
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
train_image_features = vec.transform(train_image_list)
dev_image_features = vec.transform(dev_image_list)


# get text features
tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000)
train_text_features = tfidf.fit_transform(train_text)
dev_text_features = tfidf.transform(dev_text).toarray()

# fine-tune the last layer
n_classes = encoder.classes_.shape[0]
model = image_text_model(train_image_features, train_text_features, n_classes)

#model.fit([train_image_features, train_text_features], train_labels,
#          epochs=50, batch_size=32,
#          validation_data=[[dev_image_features, dev_text_features], dev_labels])

data_gen = sparse_batch_generator(train_image_features, train_text_features, train_labels, shuffle=True)
steps_per_epoch = int(np.ceil(train_image_features.shape[0]/32.))
model.fit_generator(data_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=50,
                    validation_data=[[dev_image_features, dev_text_features], dev_labels])
