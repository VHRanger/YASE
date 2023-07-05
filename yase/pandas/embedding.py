import collections
import logging
import gc
import inspect
import numpy as np
import pandas as pd
from types import FunctionType
from typing import Iterable

from yase import mergingmethods


def getWordWeights(wordListVec: Iterable[Iterable[str]],
                   method="tf-idf",
                   normalize=True
                   ) -> Iterable[Iterable[float]]:
    """
    Produce weights for words from a list of lists of words.
    The resulting shape is the same as wordListVec (eg. 1 weight per word)
    The weights are created only taking the wordListVec input into account
    (for now)

    :param wordListVec:
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
    :param method:
        Weighing scheme. Should be in:
            'tf': Word count in the corpus
            'df': Document Frequency (# of documents the word appears in)
            'idf': IDF part of tf-idf. Equal to log(num_documents/df)
            'tf-idf': Standard tf-idf weighing scheme
        NOTE: tf-idf != (tf / idf) from the independent methods
            there is internal normalizing and SMART value differences
            https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
    :param normalize:
        whether to normalize the output in each document
        normalizing by weights = weights / weights.sum()

    :return: an vector of vectors of weights matching each word in each document
    :raises: ValueError
    """
    # TODO: Below
    import gensim
    # TODO (mranger): refactor!! remove gensim dependencies
    #                 they complicate most of the logic for little gain
    #                 Make sure new scaling is good, too
    valid_methods = ['tf-idf', 'df', 'idf', 'tf']
    if method not in valid_methods:
        raise ValueError(
            "Invalid method '{0}' in getWordWeights."
             "valid methods are {1}".format(
                 method, ', '.join(valid_methods)))
    if method == 'tf':
        wordCount = collections.Counter(sum(wordListVec, []))
        res = []
        if normalize:
            for sentence in wordListVec:
                w = np.array([wordCount[word] for word in sentence])
                res.append(w / w.sum())
        else:
            for sentence in wordListVec:
                res.append(np.array([wordCount[word] for word in sentence]))
        return res
    # Cases with gensim dependencies
    # TODO: refactor this
    # gensim harms more than it helps to get these stats
    dct = gensim.corpora.Dictionary(wordListVec)
    weightTupleList = None
    if method == 'tf-idf':
        corpus = [dct.doc2bow(line) for line in wordListVec]
        tfidfTuples = gensim.models.TfidfModel(corpus)
        weightTupleList = []
        for line in wordListVec:
            wordFreq = collections.Counter(line)
            wordids = [wordId for wordId in dct.doc2idx(line) if wordId > -1]
            wordFreqs = [wordFreq[word] for word in line]
            bagOfWords = list(zip(wordids, wordFreqs))
            weightTupleList.append(tfidfTuples[bagOfWords])
    elif method in ['df', 'idf']:
        weightTupleList = []
        for line in wordListVec:
            wordIDs = [wordId
                       for wordId in dct.doc2idx(line) 
                       if wordId > -1]
            if method =='df':
                weightTupleList.append([
                    (wordId, dct.dfs[wordId])
                    for wordId in wordIDs])
            elif method =='idf':
                weightTupleList.append([
                    (wordId, 
                    np.log(dct.num_docs / dct.dfs[wordId]))
                    for wordId in wordIDs])
    res = []
    if normalize:
        for wordlist in weightTupleList:
            w = np.array([weightTuple[1] for weightTuple in wordlist])
            res.append(w / w.sum())
    else:
        for wordlist in weightTupleList:
            res.append(np.array([weightTuple[1]
                        for weightTuple in wordlist]))
    return res

#
# TODO: remove, doesn't do anything
#
def mergeEmbeddings(embeddings: Iterable[Iterable[float]],
                    weights: Iterable[float]=None,
                    method: FunctionType=mergingmethods.sumMerge,
                    **kwargs) -> np.array:
    """
    Takes a list of embeddings and merges them into a single embeddings.
    Typical usecase is to create a sentence embedding from word embeddings.

    :param embeddings:
        Embedding matrix. 
        Each row represents one observation (eg. one word if we're merging a sentence embedding)
        for instance, an array of word embeddings
        note that weird numbers (np.NaN, np.inf, etc.) are undefined behavior
    :param weights: 
        weights on the embeddings, to be used in the method called
        weird numbers are undefined behavior here too
    :param method:
        A method is passed that takes either the format 
            Iterable[Iterable[float]], **kwargs 
        eg. the embeddings and of keyword arguments
        or the format 
            Iterable[Iterable[float]], Iterable[float], **kwargs
        which is (embeddings, weights, {keyword arguments})
    
        Methods include:
            "avg": method=np.sum, axis=0
                elementwise mean of word vectors. 
                This can be quality-degrading due to semantically meaningless 
                components having large values (see ICLR (2017), p.2, par.4)
            "sum": method=np.mean, axis=0
                elementwise sum of word vectors. 
                Also called "word centroid" in literature

    :return: a single array merged by the procedure
    :rtype: np.array[float]
    :raises: ValueError
    ------------------------------------------------------------------
    references:
        "Simple but tough to beat baseline", Arora et al. (ICLR 2017)
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings input empty!")
    if weights is not None:
        if type(weights) != np.array:
            weights = np.array(weights)
        if weights.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "Incorrect shape of weights! weights: {0}, embeddings: {1}".format(
                     weights.shape, embeddings.shape))
    try:
        if weights is None:
            return method(embeddings, **kwargs)
        return method(embeddings, weights, **kwargs)
    except TypeError as te:
        print(("\n\nError calling defined method.\n "
               + "method called: {0}\n").format(method),
            "\n\nNOTE: This can happen if you are passing weights "
            "in a function that doesn't take them as the second argument!\n"
            "Function signature was:\n\t {0}".format(inspect.signature(method)),
            ("\nArgs passed were:"
              + "\n\tembeddings: {0}"
              + "\n\tweights: {1}"
              + "\n\tkwargs: {2}").format(
                  embeddings, weights, kwargs))
        raise(te)


def sentenceEmbedding(documentCol: Iterable[Iterable[str]],
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
    if verbose == 'log':
        logger = logging.getLogger(__name__)
    if type(documentCol) != pd.Series:
        documentCol = pd.Series(documentCol)
    if weights is not None and type(weights) != pd.Series:
        weights = pd.Series(weights)
    SentenceEmbeddings = np.zeros((len(documentCol), model.vector_size))
    oovWords = [] # list of OoV words
    oovRowIndices = set() # rows with OoV Words
    for row in range(len(documentCol)):
        document = documentCol.iloc[row]
        wordWeights = np.array(weights.iloc[row]) if weights is not None else None
        if wordWeights is not None and len(wordWeights) != len(document):
            raise ValueError(
                ("Incorrect # of weights on row {0} Weights:\n{1}\nWords:\n{2}"
                ).format(row, wordWeights, document))
        # pre allocate word embedding matrix for sentence 
        sentenceWords = np.zeros((len(document), model.vector_size))
        oovIndeces = []
        for word_idx in range(len(document)):
            try:
                sentenceWords[word_idx] = model[document[word_idx]]
            except KeyError:
                oovIndeces.append(word_idx)
                oovRowIndices.add(row)
                if verbose or return_oov_stats:
                    oovWords.append(document[word_idx])
        if oovIndeces: # if oov words, filter result
            allIndices = np.indices((len(document),))
            notOOVindices = np.setxor1d(allIndices, oovIndeces)
            sentenceWords = sentenceWords[notOOVindices]
            if weights is not None:
                try:
                    wordWeights = wordWeights[notOOVindices]
                except IndexError:
                    print(("Index error on: \n\t{0}\n with weights: \n\t {1}"
                        "\nDropped Indices: {2}"
                        ).format(document, weights[row], oovIndeces))
                    raise IndexError
        # edge cases (0 or 1 word sentence)
        if len(sentenceWords) == 0:
            continue
        elif len(sentenceWords) == 1:
            SentenceEmbeddings[row] = sentenceWords[0]
        else:
            SentenceEmbeddings[row] = mergeEmbeddings(
                    sentenceWords,
                    weights=wordWeights,
                    method=mergingFunction,
                    **mergingFunctionKwargs)
    if verbose and len(oovWords) > 0:
        unique_oov = sorted(set(oovWords))
        warn_str = "Out of Vocab words (ignored): {0}. Unique: {1}".format(
                                len(oovWords), len(unique_oov))
        if verbose == 'log':
            logger.warn(warn_str)
        else: 
            print(warn_str)
        if len(unique_oov) < 1000:
            print("OoV word list: {}".format([str(oovw) for oovw in unique_oov]))
        else:
            print("First 1000 from OoV word list: {}".format(
                [str(oovw) for oovw in unique_oov][:1000]))
    if not return_oov_stats:
        return SentenceEmbeddings
    unique_oov = sorted(set(oovWords))
    res = {}
    res['unique_oov'] = unique_oov
    res['total_missing'] = len(oovWords)
    res['total_documents'] = len(documentCol)
    res['total_oov_rows'] = len(oovRowIndices)
    res['oov_row_indices'] = oovRowIndices
    res['oov_row_pct']= res['total_oov_rows'] / res['total_documents']
    return SentenceEmbeddings, res
