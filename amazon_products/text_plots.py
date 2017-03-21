import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


def sample_array(seq, n_samples, seed=123):
    random_state = np.random.RandomState(seed)
    return random_state.choice(seq, size=n_samples, replace=False)


def text_projection(text_list,
                    n_samples=None,
                    labels=None,
                    n_components=50,
                    perplexity=30,
                    n_iter=1000,
                    random_state=123):
    # downsample the data if necessary
    if n_samples:
        text_list = sample_array(text_list, n_samples, seed=random_state)
        if labels is not None:
            labels = sample_array(labels, n_samples, seed=random_state)

    # make a simple LSA pipeline
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 stop_words='english',
                                 sublinear_tf=True,
                                 use_idf=True,
                                 max_features=10000)
    normalizer = Normalizer(copy=False)
    svd = TruncatedSVD(n_components=n_components, random_state=123)

    lsa = make_pipeline(vectorizer,
                        normalizer,
                        svd)

    # fit lsa
    X = lsa.fit_transform(text_list)

    # project down to two dimensions with TSNE
    X = TSNE(n_components=2,
             init='pca',
             perplexity=perplexity,
             n_iter=n_iter,
             random_state=123).fit_transform(X)
    proj = pd.DataFrame({'component_1': X[:, 0], 'component_2': X[:, 1]})
    proj['text'] = text_list

    if labels is not None:
        proj['labels'] = labels

    return proj
