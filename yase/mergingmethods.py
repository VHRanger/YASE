import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from typing import Iterable

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: Input matrix. Each row is an observation
    :param npc: number of principal components to remove
    :return: input matrix with first npc components removed
    """
    if npc == 0:
        return X
    # Get principal components
    pc = TruncatedSVD(n_components=npc, n_iter=7).fit(X).components_
    if npc == 1:
        res = X - X.dot(pc.transpose()) * pc
    else:
        res = X - X.dot(pc.transpose()).dot(pc)
    return res


##################################################################
#                                                                #
#                                                                #
#                INPUT FUNCTIONS for mergeEmbeddings             #
#      Refer to docstring of mergeEmbeddings for more details    #
#                                                                #
#                                                                #
##################################################################


def avgMerge(embeddings: Iterable[Iterable[float]]=[],
             weights: Iterable[float]=None,
             components_to_remove=0,
             normalize=False):
    """
    Merge embeddings using (possibly weighed) averages.

    You can also remove principal components from resulting matrix. 
        Doing this is equivalent to SIF embeddings in Arora et al. (2017)

    Generally INPUT FUNCTION for embedding.mergeEmbeddings method
    Refer to docstring of mergeEmbeddings for more details

    :param embeddings: the embeddings. for instance, an array of word embeddings
    :param weights: weights to do a weighted average.
                    Weights are on the words (eg. rows)
    :param components_to_remove: number of principal components to remove
        set to 1 or higher to remove principal components

    :return: a single array of the merged embeddings
    :rtype: np.array[float]
    
    Reference: Arora et al. (2017) 
        "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
        https://openreview.net/forum?id=SyK00v5xx
    """
    if weights is not None:
        embeddings = embeddings * weights[:, np.newaxis]
    if components_to_remove > 0:
        embeddings = remove_pc(embeddings, components_to_remove)
    res = np.mean(embeddings, axis=0)
    if normalize:
        res = Normalizer('l2', copy=False).fit_transform([res])[0]
    return res


def sumMerge(embeddings: Iterable[Iterable[float]]=[],
             weights: Iterable[float]=None,
             components_to_remove=0,
             normalize=False):
    """
    Merge embeddings using (possibly weighed) sum. 
        Sometimes better on normalized embeddings

    You can also remove principal components from resulting matrix.

    Generally INPUT FUNCTION for embedding.mergeEmbeddings method
    Refer to docstring of mergeEmbeddings for more details

    :param embeddings: the embeddings. for instance, an array of word embeddings
    :param weights: weights to do a weighted average.
                    Weights are on the words (eg. rows)
    :param components_to_remove: number of principal components to remove
        set to 1 or higher to remove principal components

    :return: a single array of the merged embeddings
    :rtype: np.array[float]
    """
    if weights is not None:
        embeddings = embeddings * weights[:, np.newaxis]
    if components_to_remove > 0:
        embeddings = remove_pc(embeddings, components_to_remove)
    res = np.sum(embeddings, axis=0)
    if normalize:
        res = Normalizer('l2', copy=False).fit_transform([res])[0]
    return res


def pcaMerge(embeddings: Iterable[Iterable[float]]=[],
             weights: Iterable[float]=None,
             normalize=False):
    """
    Merge embeddings by getting their first principal component

    Generally INPUT FUNCTION for embedding.mergeEmbeddings method
    Refer to docstring of mergeEmbeddings for more details

    :param embeddings: the embeddings. for instance, an array of word embeddings
    :param weights: weights to do a weighted average.
                    Weights are on the words (eg. rows)

    :return: a single array of the merged embeddings
    :rtype: np.array[float]
    """
    if weights is not None:
        embeddings = embeddings * weights[:, np.newaxis]
    res = TruncatedSVD(n_components=1, n_iter=7).fit(embeddings).components_[0]
    if normalize:
        res = Normalizer('l2', copy=False).fit_transform([res])[0]
    return res
