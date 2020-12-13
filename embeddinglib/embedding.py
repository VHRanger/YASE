import collections
import logging
import gc
import inspect
import numpy as np
import pandas as pd
from types import FunctionType
from typing import Iterable

from embeddinglib import mergingmethods


def needs_tokenization(col):
    """
    TODO: make it
    """
    pass


def oov_words(col, model):
    """
    replaces verbose output in Sentence Embedding
    Gets list of OoV words between model and column
    """
    pass


# def getWordWeights(wordListVec: Iterable[Iterable[str]],
#                    method="tf-idf",
#                    normalize=True
#                    ) -> Iterable[Iterable[float]]:
#     """
#     Produce weights for words from a list of lists of words.
#     The resulting shape is the same as wordListVec (eg. 1 weight per word)
#     The weights are created only taking the wordListVec input into account
#     (for now)

#     :param wordListVec:
#         A iterable of iterables of words. 
#         We assume the documents are already split in word lists
#         eg. [["sale", "ford"], ["cheap", "good"]]
#     :param method:
#         Weighing scheme. Should be in:
#             'tf': Word count in the corpus
#             'df': Document Frequency (# of documents the word appears in)
#             'idf': IDF part of tf-idf. Equal to log(num_documents/df)
#             'tf-idf': Standard tf-idf weighing scheme
#         NOTE: tf-idf != (tf / idf) from the independent methods
#             there is internal normalizing and SMART value differences
#             https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
#     :param normalize:
#         whether to normalize the output in each document
#         normalizing by weights = weights / weights.sum()

#     :return: an vector of vectors of weights matching each word in each document
#     :raises: ValueError
#     """
#     # TODO: Below
#     import gensim
#     # TODO (mranger): refactor!! remove gensim dependencies
#     #                 they complicate most of the logic for little gain
#     #                 Make sure new scaling is good, too
#     valid_methods = ['tf-idf', 'df', 'idf', 'tf']
#     if method not in valid_methods:
#         raise ValueError(
#             "Invalid method '{0}' in getWordWeights."
#              "valid methods are {1}".format(
#                  method, ', '.join(valid_methods)))
#     if method == 'tf':
#         wordCount = collections.Counter(sum(wordListVec, []))
#         res = []
#         if normalize:
#             for sentence in wordListVec:
#                 w = np.array([wordCount[word] for word in sentence])
#                 res.append(w / w.sum())
#         else:
#             for sentence in wordListVec:
#                 res.append(np.array([wordCount[word] for word in sentence]))
#         return res
#     # Cases with gensim dependencies
#     # TODO: refactor this
#     # gensim harms more than it helps to get these stats
#     dct = gensim.corpora.Dictionary(wordListVec)
#     weightTupleList = None
#     if method == 'tf-idf':
#         corpus = [dct.doc2bow(line) for line in wordListVec]
#         tfidfTuples = gensim.models.TfidfModel(corpus)
#         weightTupleList = []
#         for line in wordListVec:
#             wordFreq = collections.Counter(line)
#             wordids = [wordId for wordId in dct.doc2idx(line) if wordId > -1]
#             wordFreqs = [wordFreq[word] for word in line]
#             bagOfWords = list(zip(wordids, wordFreqs))
#             weightTupleList.append(tfidfTuples[bagOfWords])
#     elif method in ['df', 'idf']:
#         weightTupleList = []
#         for line in wordListVec:
#             wordIDs = [wordId
#                        for wordId in dct.doc2idx(line) 
#                        if wordId > -1]
#             if method =='df':
#                 weightTupleList.append([
#                     (wordId, dct.dfs[wordId])
#                     for wordId in wordIDs])
#             elif method =='idf':
#                 weightTupleList.append([
#                     (wordId, 
#                     np.log(dct.num_docs / dct.dfs[wordId]))
#                     for wordId in wordIDs])
#     res = []
#     if normalize:
#         for wordlist in weightTupleList:
#             w = np.array([weightTuple[1] for weightTuple in wordlist])
#             res.append(w / w.sum())
#     else:
#         for wordlist in weightTupleList:
#             res.append(np.array([weightTuple[1]
#                         for weightTuple in wordlist]))
#     return res 


def sentenceEmbedding(
        documentCol: Iterable[Iterable[str]],
        model,
        weights: Iterable[Iterable[float]]=None,
        mergingFunction: FunctionType=mergingmethods.sumMerge,
        mergingFunctionKwargs: dict={},
        return_oov_stats=False,
        verbose=True
        )-> np.array:
    """
    Merge a vector of sentences (which are already split into words)
        into a vector of sentence embedding.

    Methods for each embedding merging methods are input as params. 
    See mergingFunction param for reference

    NOTE: This method expects the document column to be already split 
        and preprocessed (so in series of ['word', 'another', 'word'] format)
        Use our prepackaged string processing methods if you need it.

    :param documentCol: 
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
        gets converted to a pd.series before treating
    :param model: a trained embedding model 
        (usually gensim Keyedvectors object)
        Needs to support ["word"] -> vector style API
    :param weights:
        weights on each words, to be used in the embedding merging method called
    :param mergingFunction:
        a function to merge word embeddings into sentence embeddings
        passed to mergeEmbeddings.
        function is usually from mergingmethods.py
        Should respect the require function signature
            (embeddings, weights, {keyword arguments}) if weights are passed
            (embeddings, {keyword arguments}) if no weights are passed
    :param mergingFunctionKwargs:
        keyword arguments for above function
        With mergingmethods.avgMerge You can pass {"components_to_remove": 1}
            which removes the first principal component from resulting embedding

    :return: Matrix, each row is the sentence embedding of row in documentCol.
    """
    def map_f(row):
        try: return model.__getitem__(row)
        except KeyError: return np.nan
    def merging_f(row):
        return mergingFunction(row, **mergingFunctionKwargs)
    res = documentCol.applymap(map_f)
    res = res.aggregate(func=merging_f, axis='columns', result_type='expand')
    return res
