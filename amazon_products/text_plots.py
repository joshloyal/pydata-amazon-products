import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


def sample_array(seq, n_samples, seed=123):
    random_state = np.random.RandomState(seed)
    return random_state.choice(seq, size=n_samples, replace=False)


def frequency_plot(text_list,
                   ngram_range=(1,2),
                   max_words=500,
                   plot_n_words=10,
                   **kwargs):
    """Generate a horizontal bar chart of words ranked by their global
    tf-idf weights in the corpus.

    Parameters
    ----------
    text_list : array-like of shape [n_samples,]
        The list of documents to generate the word cloud.
    ngram_range : tuple (default=(1,2))
        The ngrams to use. Defaults to uni-gram and bi-grams.
    max_words : int (default=500)
        The maximum vocabulary of the word cloud.
    plot_n_words : int (default=10)
        The number of words to display in the bar plot.
    **kwargs
        Any remaining key word arguments to pass in to `sns.barplot`.

    Returns
    -------
    A seaborn barplot.
    """
    # fit text features
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 stop_words='english',
                                 sublinear_tf=False,
                                 use_idf=True,
                                 max_df=0.95,
                                 min_df=5,
                                 max_features=max_words)
    tfidf = vectorizer.fit_transform(text_list)
    global_tfidf = np.asarray(tfidf.sum(axis=0)).flatten()

    # weight word cloud by global idf weights
    vocab = vectorizer.vocabulary_
    freq = {word: global_tfidf[word_index] for
            word, word_index in vocab.items()}
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    word, weight = zip(*freq)
    data = pd.DataFrame({'word': word, 'tf-idf': weight})

    return sns.barplot(
        'tf-idf', 'word', data=data[:plot_n_words], **kwargs)


def word_cloud(text_list,
               ngram_range=(1,2),
               max_words=500,
               fig_size=(500,500)):
    """Generate a word-cloud weighted by idf weights to a list of
    documents.

    Parameters
    ----------
    text_list : array-like of shape [n_samples,]
        The list of documents to generate the word cloud.
    ngram_range : tuple (default=(1,2))
        The ngrams to use. Defaults to uni-gram and bi-grams.
    max_words : int (default=500)
        The maximum vocabulary of the word cloud.
    fig_size : tuple (default=(500,500))
        The size of the word cloud image.

    Returns
    -------
    The word cloud as a PIL Image.
    """
    # fit text features
    vectorizer = TfidfVectorizer(ngram_range=(1,2),
                                 stop_words='english',
                                 sublinear_tf=False,
                                 use_idf=True,
                                 max_df=0.95,
                                 min_df=5,
                                 max_features=max_words)
    tfidf = vectorizer.fit_transform(text_list)
    global_tfidf = np.asarray(tfidf.sum(axis=0)).flatten()

    width, height = fig_size
    word_cloud = wordcloud.WordCloud(width=width, height=height)

    # weight word cloud by global idf weights
    vocab = vectorizer.vocabulary_
    freq = {word: global_tfidf[word_index] for
            word, word_index in vocab.items()}
    word_cloud.fit_words(freq)

    return word_cloud.to_image()


def text_embedding(text_list,
                   n_samples=None,
                   labels=None,
                   n_components=50,
                   perplexity=30,
                   n_iter=1000,
                   random_state=123):
    """Create an embedding that reflects the semantics of a collection of
    documents using LSA and t-SNE.
    """
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

    # project down to two dimensions with t-SNE
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
