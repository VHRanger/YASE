from copy import deepcopy
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Normalizer
import unittest
from unittest import TestCase

import embeddinglib
from embeddinglib.stringprocessing import tokenizeCol, tokenize
from embeddinglib import stringprocessing
from embeddinglib import mergingmethods
from embeddinglib import embedding


class TestIsolatedStringProcessing(TestCase):
    """
    basic testing of vectorized document cleaning -> splitting
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = [
            ["157681 Miles. Only $12,849", 
             "Used 2006 Ford F-150"],
            ["\"city, state\" format (quotes)", 
             "different 'quotes'"],
        ]
        for i in range(len(tdata)):
            tdata[i] = pd.Series(tdata[i])
        removeCharDict = deepcopy(stringprocessing.REMOVE_CHAR)
        for i in range(0, 10):
            removeCharDict[str(i)] = "#"
        identityOp = lambda x: tokenizeCol(
            x, split=False, nanValue=np.nan, lower=False, replaceDict={})
        lowerSplit = lambda x: tokenizeCol(
            x, split=True, nanValue='', lower=True, replaceDict={})
        replaceNumerics = lambda x: tokenizeCol(
            x, lower=True, nanValue='', split=True, 
            replaceDict=removeCharDict)

        for i in tdata:
            # The function with everything off returns the same data
            self.assertTrue(identityOp(i).equals(i))
        
        # whitespace splitting and lowercasing works as expected
        self.assertTrue(lowerSplit(tdata[0]).equals(
            pd.DataFrame([["157681", "miles.", "only", "$12,849"], 
                        ["used", "2006", "ford", "f-150"]])
        ), msg=lowerSplit(tdata[0]))
        self.assertTrue(lowerSplit(tdata[1]).equals(
            pd.DataFrame([["\"city,", "state\"", "format", "(quotes)"], 
                        ["different", "'quotes'"]])
        ), msg=lowerSplit(tdata[1]))
        # replacing punctuation & numerics -> #
        self.assertTrue(replaceNumerics(tdata[0]).equals(
            pd.DataFrame([["######", "miles", "only", "#####"], 
                        ["used", "####", "ford", "f", "###"]]),
        ), msg=replaceNumerics(tdata[0]))
        self.assertTrue(replaceNumerics(tdata[1]).equals(
            pd.DataFrame([["city", "state", "format", "quotes"], 
                        ["different", "quotes"]]),
        ), msg=replaceNumerics(tdata[1]))
    
    def test_edge_cases(self):
        tdata = [
            "157681 Miles. Only $12,849", 
            "Used 2006 Ford F-150",
            "\"city, state\" format (quotes)", 
            "different 'quotes'",
            [],
            np.nan,
            ""]
        # passing as list or as pd.Series should be the same
        self.assertTrue(tokenizeCol(tdata).equals(
                        tokenizeCol(pd.Series(tdata))))
        resultLen = list(tokenizeCol(tdata, expand=False).str.len())
        self.assertEqual(resultLen, [4, 5, 4, 2, 0, 0, 0])

    def test_tokenizing(self):
        """Test Inputs in tokenize method"""
        tdata = [
            "157681 Miles. Only $12,849", 
            "Used 2006 Ford F-150",
            "\"city, state\" format (quotes)", 
            "different 'quotes'",
            [],
            np.nan,
            ""]
        # passing a single column to tokenize should do the same as tokenizeCol
        self.assertTrue(tokenizeCol(tdata).equals(
                        tokenize(pd.Series(tdata))))
        self.assertTrue(tokenizeCol(tdata).equals(
                        tokenize(tdata)))
        # result is equal to concatenating several tokenizeCol results
        cat_res = tokenizeCol(tdata) + tokenizeCol(tdata)
        self.assertTrue(cat_res.equals(
                        tokenize([tdata, tdata])))
        # DataFrame input works the same as above
        self.assertTrue(cat_res.equals(
                        tokenize(pd.DataFrame([tdata, tdata]).T)))
# Input data for embedding methods below
def given_test_embeddings():
    return np.array([[1, -2, 3.5, 1], 
                    [0.4, 5e12, 0, 1], 
                    [0.0, 8, 0.000001, 1]])
def given_test_weights():
    return [
        [1.,1.,1.],
        [0.,0.,0.],
        [0.1, 0.2, 0.3],
        [2., 2., 2.]
    ]

def given_tdocs():
    """test docs for groupedEmbedding"""
    return [
        ["157681", "miles.", "only", "$12,849"], 
        ["used", "2006", "ford", "f-150", 'repeatedword'],
        ["\"city,", "state\"", "format", "(quotes)", 'repeatedword'], 
        ["different", "'quotes'"],
        ["######", "miles", "only", "#####"], 
        ["used", "####", "ford", "f-###"],
        ["city", "state", "format", "quotes"], 
        ["different", "quotes"]]

def given_repeated_words():
    """doc with repeated words"""
    return [["test", "test", "test"],
           ["unique", "words", "except", "here"],
           ["test", "again", "here"]]

def given_tgroups():
    """test columns keys, indexed with tdocs"""
    return [1, 1, 1, 1, 2, 2, 3, 3]

def given_normalized_tweights():
    "test weights coindexed to tdocs"
    return [
        [0.5, 0.2, 0.2, 0.1], 
        [0.3, 0.3, 0.3, 0.05, 0.05],
        [0.25, 0.25, 0.25, 0.20, 0.05], 
        [1,  0],
        [0, 0.000001, 0.9, 0.999999], 
        [0.8, 0.05, 0.05, 0.1],
        [0.1, 0.7, 0.01, 0.19], 
        [0.5, 0.5]]

def given_tweights():
    "test weights coindexed to tdocs"
    return [
        [5, 2, 20, 10000], 
        [3, 3, 3, 1],
        [0.0000025, 0.25, 2500, 250], 
        [1,  0],
        [0, 0.000001, 9, 999999], 
        [8, 0.05, 0.05, 1],
        [1, 7, 0.000001, 19], 
        [5, 5.]]

class mockW2V(object):
    """Mock gensim.models.keyedvectors class"""
    def __init__(self, embeddingDict):
        """init based on a word embedding dict[str -> np.array]"""
        self.embeddings = embeddingDict
        self.words = list(embeddingDict.keys())
        self.vector_size = len(self.embeddings[self.words[0]])
        for word in self.words:
            assert len(self.embeddings[word]) == self.vector_size, (
                "vector length not the same fro all mock word2vec model input")

    def __getitem__(self, key):
        return self.embeddings[key]

    def remove(self, key):
        self.embeddings.pop(key)
        self.words = list(self.embeddings.keys())

    def keys(self):
        return self.words

def given_mock_keyedVectors():
    """
    mock a gensim keyedVector object 
    (base class of word2vec)
    """
    # concat list of lists to unique words
    all_words = list(set(sum(given_tdocs(), [])))
    mymock = {}
    for word in range(len(all_words)):
        mymock[all_words[word]] = np.array([0.1, 0.2, 0.3]) * word
    return mockW2V(mymock)

def given_incomplete_keyedVectors():
    """
    mock gensim keyedVector object with missing words
    """
    kv = given_mock_keyedVectors()
    for _ in range(4):
        kv.remove(random.choice(list(kv.keys())))
    return kv


# class TestWordWeights(TestCase):
#     """
#     tests for the getWordWeights function

#     Note: can't test tf-idf method by composing tf and idf methods
#     since there is internal normalizing and SMART value differences
#     https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
#     """
#     def test_given_unique_words_per_doc_tf_and_idf_normalize_equal(self):
#         tdocs = [["157681", "miles.", "only", "$12,849"], 
#                 ["used", "2006", "ford", "f-150"],
#                 ["\"city,", "state\"", "format", "(quotes)"], 
#                 ["different", "'quotes'"],
#                 ["######", "miles", "only", "#####"], 
#                 ["used", "####", "ford", "f-###"],
#                 ["city", "state", "format", "quotes"], 
#                 ["different", "quotes"]]
#         normalizedtf = embedding.getWordWeights(tdocs, "tf", True)
#         normalizeddf = embedding.getWordWeights(tdocs, "df", True)
#         for i in range(len(normalizedtf)):
#             np.testing.assert_array_almost_equal(
#                 normalizedtf[i], normalizeddf[i])

#     def test_given_wordlist_tf_idf_count_normally(self):
#         tdocs= given_repeated_words()
#         tf = embedding.getWordWeights(tdocs, "tf", False)
#         np.testing.assert_array_almost_equal(tf[0], [4, 4, 4])
#         np.testing.assert_array_almost_equal(tf[1], [1, 1, 1, 2])
#         np.testing.assert_array_almost_equal(tf[2], [4, 1, 2])
#         df = embedding.getWordWeights(tdocs, "df", False)
#         np.testing.assert_array_almost_equal(df[0], [2, 2, 2])
#         np.testing.assert_array_almost_equal(df[1], [1, 1, 1, 2])
#         np.testing.assert_array_almost_equal(df[2], [2, 1, 2])

#     def test_given_normalized_sums_1(self):
#         normalizedtf = embedding.getWordWeights(given_tdocs(), "tf", True)
#         normalizeddf = embedding.getWordWeights(given_tdocs(), "df", True)
#         normalizeddf = embedding.getWordWeights(given_tdocs(), "idf", True)
#         normalizedtfidf = embedding.getWordWeights(given_tdocs(), "tf-idf", True)
#         for i in range(len(given_tdocs())):
#             for res in [normalizedtf[i], normalizeddf[i], 
#                         normalizeddf[i], normalizedtfidf[i]]:
#                 self.assertAlmostEqual(res.sum(), 1)

#     def test_given_repeated_words_output_dim_ok(self):
#         """repeated words break gensim pipeline by default
#            check weights are OK here"""
#         tdocs = given_repeated_words()
#         for method in ["tf", "idf", "df", "tf-idf"]:
#             ww = embedding.getWordWeights(tdocs, method)
#             for line in range(len(tdocs)):
#                 self.assertEqual(len(ww[line]), len(tdocs[line]),
#                     msg="weights output dim should be equal on repeated words")


class TestSentenceEmbedding(TestCase):
    def test_given_oov_applies_correctly(self):
        """
        Test that OOV words apply equally and don't break pipeline
        """
        tdocs = pd.DataFrame(given_tdocs())
        w2vec = given_incomplete_keyedVectors()
        meanEmbeddingsManual = embedding.sentenceEmbedding(
            tdocs, model=w2vec, weights=None,
            mergingFunction=np.mean,
            mergingFunctionKwargs={'axis':0},
            verbose=False)
        meanEmbeddings = embedding.sentenceEmbedding(
            tdocs, model=w2vec, weights=None,
            mergingFunction=mergingmethods.avgMerge,
            verbose=False)
        for key in range(len(meanEmbeddings)):
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual.iloc[key],
                meanEmbeddings.iloc[key])

class TestMergingMethods(TestCase):
    """
    Sanity checks for merging methods
    """
    def test_avg(self):
        tt = np.array([
            [1, 1, 1],
            [3, 3, 3],
        ])
        np.testing.assert_array_almost_equal(
            mergingmethods.avgMerge(tt),
            np.array([2,2,2])
        )

    def test_sum(self):
        tt = np.array([
            [1, 1, 1],
            [3, 3, 3],
        ])
        np.testing.assert_array_almost_equal(
            mergingmethods.sumMerge(tt),
            np.array([4,4,4])
        )

    def test_pca(self):
        tt = np.array([
            [1, 1, 1],
            [3, 3, 3],
        ])
        np.testing.assert_array_almost_equal(
            mergingmethods.pcaMerge(tt),
            np.array([0.577, 0.577, 0.577]),
            decimal=3
        )