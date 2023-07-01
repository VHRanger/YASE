import numpy as np
import pandas as pd
from sklearn import cluster, metrics
from typing import Iterable


def evalClusteringOnLabels(
    SentenceEmbeddings: Iterable[Iterable[float]],
    groupLabels: Iterable,
    verbose=True
    ) -> Iterable[float]: # pragma: no cover
    """
    Evaluate a vector of sentence embeddings for clustering around
        pre-defined categories.

    For example, you can test that embeddings in an ad campaign cluster 
        around the campaigns, by passing the ad embeddings and the campaign labels.

    Since cluster evaluation is hard, we use agglomerative hierarchical with euclidean/ward linkage. 
    See below for why this specific algorithm was chosen. We also pass through three metrics:
        adjusted mutual information score,
        adj. Rand index,
        Fowlkes-Mallows score

    The result is a vector of all results in order (
        adj. MI score, adj. Rand score, F-M score
    ) for agglomerative/spectral clusterings

    :param SentenceEmbeddings: 
        A vector of sentence embedding vectors. Co-indexed with groupLabels
    :param groupLabels: 
        Category labels for the sentences. Co-indexed with SentenceEmbeddings
    :param verbose: 
        prints a result table if true.

    :returns: a np.array of all index score results
    -------------------------------
    Algorithm choice:
        We need a clustering algorithm that does not assume cluster shape, that is stable across runs,
        and that is stable across parameter choices. 
        This is for evaluation to be as deterministic as possible.

        This means the following are unacceptable: k-means (unstable across runs), 
        spectral (unstable across parameters), mean shift (assumes globular shape).

        This leaves DBSCAN and agglomerative clustering. 
        
        DBSCAN tends to perform poorly on clusters of word embeddings. 
        It seems they are not clustered by density.

        Agglomerative has an added advantage: on normalized embeddings, 
            the euclidian metric is the same as the well-liked cosine distance 
            (for semantic similarity). Thus agglomerative is our choice.
    """
    n_clusters = len(set(groupLabels))
    agglo = cluster.AgglomerativeClustering(
                                n_clusters=n_clusters, 
                                affinity='euclidean', linkage='ward'
                            ).fit(SentenceEmbeddings).labels_
    results = []
    results.append(metrics.adjusted_mutual_info_score(agglo, groupLabels))
    results.append(metrics.adjusted_rand_score(agglo, groupLabels))
    results.append(metrics.fowlkes_mallows_score(agglo, groupLabels))
    if verbose:
        print("adj. MI score:   {0:.2f}".format(results[0]))
        print("adj. RAND score: {0:.2f}".format(results[1]))
        print("F-M score:       {0:.2f}".format(results[2]))
    return np.array(results)