import numpy as np
import scipy.sparse as sparse


def sparse_batch_generator(image_features, text_features, labels, batch_size=32, shuffle=False, random_state=123):
    random_state = np.random.RandomState(random_state)

    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError('Features have different number of samples!')

    n_samples = image_features.shape[0]
    n_batches = np.ceil(n_samples / batch_size)

    counter = 0
    sample_index = np.arange(n_samples)
    if shuffle:
        random_state.shuffle(sample_index)

    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        image_batch = image_features[batch_index, :]
        if sparse.issparse(image_batch):
            image_batch = image_batch.toarray()

        text_batch = text_features[batch_index, :]
        if sparse.issparse(text_batch):
            text_batch = text_batch.toarray()

        labels_batch = labels[batch_index]
        counter += 1
        yield [image_batch, text_batch], labels_batch

        if counter == n_batches:
            if shuffle:
                random_state.shuffle(sample_index)
            counter = 0
