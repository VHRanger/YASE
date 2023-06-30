from copy import deepcopy
import gensim
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Normalizer
import unittest
from unittest import TestCase

import embeddinglib
from embeddinglib import mergeEmbeddings
from embeddinglib.stringprocessing import tokenizeCol, tokenize
from embeddinglib import stringprocessing, mergingmethods, embedding, encoders

import common_test_setups as cts

# TODO: Test encoders.RobustOrdinalEncoder

#################################
#                               #
#          Test Cases           #
#                               #
#################################

class TestTopLevelFunctions(TestCase):
    """
    Testing the top level user functions
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = deepcopy(cts.testdata)
        tdata2 = pd.Series(cts.testdata, name="testcol")
        tmodel = cts.mockW2V(cts.testmodel)
        res = [
            encoders.embed_column(tdata, tmodel, verbose=False, replace_dict=None, col_name="test"),
            encoders.embed_column(tdata, tmodel, verbose=False, replace_dict=None),
            encoders.embed_column(tdata2, tmodel, verbose=False, replace_dict=None, col_name="test"),
            encoders.embed_column(tdata2, tmodel, verbose=False, replace_dict=None),
        ]
        for r in res:
            self.assertTrue(r.iloc[0][0] > -9999999)


class TestSKLAPIModels(TestCase):
    """
    Testing the top level user functions
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = pd.DataFrame({
            "first": pd.Series(cts.testdata), 
            "second": pd.Series(cts.testdata)
        })
        tmodel = cts.mockW2V(cts.testmodel)
        m1 = encoders.ColumnEmbedder(
            tmodel, verbose=False, threads=1, replace_dict=None, min_rows=1e10)
        res = [
            m1.transform(tdata),
        ]
        for r in res:
            self.assertTrue(r.iloc[0][0] > -9999999)


class TestIsolatedStringProcessing(TestCase):
    """
    basic testing of vectorized document cleaning -> splitting
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = [
            [cts.testdata[0], cts.testdata[1]],
            [cts.testdata[2], cts.testdata[3]],
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
            pd.Series([["157681", "miles.", "only", "$12,849"], 
                        ["used", "2006", "ford", "f-150"]])
        ), msg=lowerSplit(tdata[0]))
        self.assertTrue(lowerSplit(tdata[1]).equals(
            pd.Series([["\"city,", "state\"", "format", "(quotes)"], 
                        ["different", "'quotes'"]])
        ), msg=lowerSplit(tdata[1]))
        # replacing punctuation & numerics -> #
        self.assertTrue(replaceNumerics(tdata[0]).equals(
            pd.Series([["######", "miles", "only", "#####"], 
                        ["used", "####", "ford", "f", "###"]]),
        ), msg=replaceNumerics(tdata[0]))
        self.assertTrue(replaceNumerics(tdata[1]).equals(
            pd.Series([["city", "state", "format", "quotes"], 
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

        resultLen = list(tokenizeCol(tdata).str.len())
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

class testEmbeddingMerging(TestCase):
    """
    basic testing of the mergeEmbeddings function
    """
    def test_given_empty_input_raises(self):
        with self.assertRaises(ValueError):
            mergeEmbeddings([])

    def test_given_normal_input_MergesEmbeddings_normally(self):
        tembeddings = given_test_embeddings()
        tweights = given_test_weights()

        # default weight behavior is multiplication on np.sum
        # and multiplying by weights of 1 is same as not having weights
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings),
            mergeEmbeddings(tembeddings, tweights[0]))

        # passing weights in default behavior works
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(2 * tembeddings),
            mergeEmbeddings(tembeddings, tweights[3]))

        # Assure that passing weights on default behavior doesn't delete them
        # as the weights = None line
        self.assertIsNotNone(tweights[0])

        # passing empty weight vector raises error
        with self.assertRaises(ValueError):
            mergeEmbeddings(tembeddings, [])
        
        # tautological check on numpy mean function
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, method=np.mean, axis=0),
            np.mean(tembeddings, axis=0))

        # Same with different func signature (for kwargs messiness)
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, None, np.mean, axis=0),
            np.mean(tembeddings, axis=0))                                        

         # PCA Merging method
        pcae = mergeEmbeddings(tembeddings, tweights[2], 
                            method=mergingmethods.pcaMerge)
        avge = mergeEmbeddings(tembeddings, tweights[0], 
                            method=mergingmethods.avgMerge)
        np.testing.assert_array_almost_equal(pcae.shape, avge.shape)

    
    def test_given_normal_input_predefined_funcs_merge_correctly(self):
        tembeddings = given_test_embeddings()
        tweights = given_test_weights()
        # avgMerge function works as expected
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, None, mergingmethods.avgMerge),
            np.mean(tembeddings, axis=0))
        
        # weighed average
        # note you need to multiply each row (word) by the weights
        np.testing.assert_array_almost_equal(
            mergeEmbeddings(tembeddings, tweights[2], 
                            method=mergingmethods.avgMerge),
            np.mean(tembeddings * np.array(tweights[2])[:, np.newaxis], axis=0))


class TestWordWeights(TestCase):
    """
    tests for the getWordWeights function

    Note: can't test tf-idf method by composing tf and idf methods
    since there is internal normalizing and SMART value differences
    https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System
    """
    def test_given_unique_words_per_doc_tf_and_idf_normalize_equal(self):
        tdocs = [["157681", "miles.", "only", "$12,849"], 
                ["used", "2006", "ford", "f-150"],
                ["\"city,", "state\"", "format", "(quotes)"], 
                ["different", "'quotes'"],
                ["######", "miles", "only", "#####"], 
                ["used", "####", "ford", "f-###"],
                ["city", "state", "format", "quotes"], 
                ["different", "quotes"]]
        normalizedtf = embedding.getWordWeights(tdocs, "tf", True)
        normalizeddf = embedding.getWordWeights(tdocs, "df", True)
        for i in range(len(normalizedtf)):
            np.testing.assert_array_almost_equal(
                normalizedtf[i], normalizeddf[i])

    def test_given_wordlist_tf_idf_count_normally(self):
        tdocs= cts.given_repeated_words()
        tf = embedding.getWordWeights(tdocs, "tf", False)
        np.testing.assert_array_almost_equal(tf[0], [4, 4, 4])
        np.testing.assert_array_almost_equal(tf[1], [1, 1, 1, 2])
        np.testing.assert_array_almost_equal(tf[2], [4, 1, 2])
        df = embedding.getWordWeights(tdocs, "df", False)
        np.testing.assert_array_almost_equal(df[0], [2, 2, 2])
        np.testing.assert_array_almost_equal(df[1], [1, 1, 1, 2])
        np.testing.assert_array_almost_equal(df[2], [2, 1, 2])

    def test_given_normalized_sums_1(self):
        normalizedtf = embedding.getWordWeights(cts.given_tdocs(), "tf", True)
        normalizeddf = embedding.getWordWeights(cts.given_tdocs(), "df", True)
        normalizeddf = embedding.getWordWeights(cts.given_tdocs(), "idf", True)
        normalizedtfidf = embedding.getWordWeights(cts.given_tdocs(), "tf-idf", True)
        for i in range(len(cts.given_tdocs())):
            for res in [normalizedtf[i], normalizeddf[i], 
                        normalizeddf[i], normalizedtfidf[i]]:
                self.assertAlmostEqual(res.sum(), 1)

    def test_given_repeated_words_output_dim_ok(self):
        """repeated words break gensim pipeline by default
           check weights are OK here"""
        tdocs = cts.given_repeated_words()
        for method in ["tf", "idf", "df", "tf-idf"]:
            ww = embedding.getWordWeights(tdocs, method)
            for line in range(len(tdocs)):
                self.assertEqual(len(ww[line]), len(tdocs[line]),
                    msg="weights output dim should be equal on repeated words")


class TestSentenceEmbedding(TestCase):
    def test_given_oov_applies_correctly(self):
        """
        Test that OOV words apply equally and don't break pipeline
        """
        tdocs = cts.given_tdocs()
        w2vec = cts.given_incomplete_keyedVectors()
        meanEmbeddingsManual, oov_stats = embedding.sentenceEmbedding(
            tdocs, model=w2vec, weights=None,
            mergingFunction=np.mean,
            mergingFunctionKwargs={"axis": 0},
            return_oov_stats=True,
            verbose=False)
        meanEmbeddings = embedding.sentenceEmbedding(
            tdocs, model=w2vec, weights=None,
            mergingFunction=mergingmethods.avgMerge,
            verbose=False)
        SIFEmbeddings = embedding.sentenceEmbedding(
            tdocs, model=w2vec, weights=None,
            mergingFunction=mergingmethods.avgMerge,
            verbose=False)
        self.assertGreater(oov_stats['total_oov_rows'], 0)
        for key in range(len(meanEmbeddings)):
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key],
                meanEmbeddings[key])

class TestGroupedEmbedding(TestCase):
    """
    basic checks for groupedEmbedding function
    """    
    def test_given_list_input_input_untouched(self):
        tdocs = cts.given_tdocs()
        tgroups = cts.given_tgroups()
        w2vec = cts.given_mock_keyedVectors()
        gembeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, verbose=False)
        self.assertTrue(type(tdocs) == list and type(tgroups) == list,
                        msg="input should be unchanged")
        self.assertEqual(len(gembeddings), len(set(tgroups)),
                        msg="groupe dict should be # of categories")

    def test_give_input_functions_applies_correctly(self):
        """
        test function from mergingmethods.py works with tautological equivalent
        """
        tdocs = cts.given_tdocs()
        tgroups = cts.given_tgroups()
        w2vec = cts.given_mock_keyedVectors()
        meanEmbeddingsManual = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=np.mean,
            word2SentenceKwargs={"axis": 0},
            sentence2GroupMerge=np.mean,
            sentence2GroupKwargs={"axis": 0},
            verbose=False)
        meanEmbeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=mergingmethods.avgMerge,
            sentence2GroupMerge=mergingmethods.avgMerge,
            verbose=False)
        for key in meanEmbeddings:
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key], 
                meanEmbeddings[key])

    def test_given_pooling_applies_correctly(self):
        """
        test function from mergingmethods.py works with tautological equivalent
        """
        tdocs = cts.given_tdocs()
        tgroups = cts.given_tgroups()
        w2vec = cts.given_mock_keyedVectors()
        meanEmbeddingsManual = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec,
            word2SentenceMerge='pooled',
            sentence2GroupMerge=np.mean,
            sentence2GroupKwargs={"axis": 0},
            verbose=False)
        meanEmbeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec,
            word2SentenceMerge='pooled',
            sentence2GroupMerge=mergingmethods.avgMerge,
            verbose=False)
        for key in meanEmbeddings:
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key],
                meanEmbeddings[key])

    def test_given_unique_pooling_applies_correctly(self):
        """
        test function from mergingmethods.py works with tautological equivalent
        """
        tdocs = cts.given_tdocs()
        tgroups = cts.given_tgroups()
        w2vec = cts.given_mock_keyedVectors()
        meanEmbeddingsManual = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec,
            word2SentenceMerge='unique',
            sentence2GroupMerge=np.mean,
            sentence2GroupKwargs={"axis": 0},
            verbose=False)
        meanEmbeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec,
            word2SentenceMerge='unique',
            sentence2GroupMerge=mergingmethods.avgMerge,
            verbose=False)
        for key in meanEmbeddings:
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key],
                meanEmbeddings[key])

    def test_given_oov_applies_correctly(self):
        """
        Test that OOV words apply equally and don't break pipeline
        """
        tdocs = cts.given_tdocs()
        tgroups = cts.given_tgroups()
        w2vec = cts.given_incomplete_keyedVectors()
        meanEmbeddingsManual = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=np.mean,
            word2SentenceKwargs={"axis": 0},
            sentence2GroupMerge=np.mean,
            sentence2GroupKwargs={"axis": 0},
            verbose=False)
        meanEmbeddings = embedding.groupedEmbedding(
            tdocs, tgroups, model=w2vec, weights=None,
            word2SentenceMerge=mergingmethods.avgMerge,
            sentence2GroupMerge=mergingmethods.avgMerge,
            verbose=False)
        for key in meanEmbeddings:
            np.testing.assert_array_almost_equal(
                meanEmbeddingsManual[key],
                meanEmbeddings[key])

    def test_weights_tautological_works(self):
        """ 
        test that weights work normally in a tautological check
        """
        meanEmbeddings = embedding.groupedEmbedding(
            cts.given_tdocs(), cts.given_tgroups(), 
            model=cts.given_incomplete_keyedVectors(), 
            weights=cts.given_normalized_tweights(),
            word2SentenceMerge=mergingmethods.sumMerge,
            sentence2GroupMerge=mergingmethods.sumMerge,
            verbose=False)
        w2vec = cts.given_incomplete_keyedVectors()
        for key in meanEmbeddings:
            self.assertEqual(len(meanEmbeddings[key]),
                             w2vec.vector_size)

    def test_SIF_works(self):
        """ 
        test that components_to_remove works in grouped embedding merging
        """
        # Just there to integration test normalization
        meanEmbeddings_n = embedding.groupedEmbedding(
            cts.given_tdocs(), cts.given_tgroups(), 
            model=cts.given_incomplete_keyedVectors(), 
            weights=cts.given_normalized_tweights(),
            word2SentenceMerge=mergingmethods.sumMerge,
            sentence2GroupMerge=mergingmethods.avgMerge,
            sentence2GroupKwargs={"components_to_remove": 1, 
                                  "normalize": True},
            verbose=False)
        meanEmbeddings = embedding.groupedEmbedding(
            cts.given_tdocs(), cts.given_tgroups(), 
            model=cts.given_incomplete_keyedVectors(), 
            weights=cts.given_normalized_tweights(),
            word2SentenceMerge=mergingmethods.sumMerge,
            sentence2GroupMerge=mergingmethods.avgMerge,
            sentence2GroupKwargs={"components_to_remove": 1, 
                                  "normalize": False},
            verbose=False)
        w2vec = cts.given_incomplete_keyedVectors()
        for key in meanEmbeddings:
            self.assertEqual(len(meanEmbeddings[key]),
                             w2vec.vector_size)
            self.assertEqual(len(meanEmbeddings_n[key]),
                    w2vec.vector_size)

    def test_given_weight_embedding_mismatch_raises(self):
        """weight dimension != embedding dimension should raise"""
        bad_weights = cts.given_normalized_tweights()
        bad_weights[0] = bad_weights[0][:-1]
        bad_weights[1] = bad_weights[0][:-2]
        with self.assertRaises(ValueError):
            try:
                meanEmbeddings = embedding.groupedEmbedding(
                    cts.given_tdocs(), cts.given_tgroups(), 
                    model=cts.given_incomplete_keyedVectors(), 
                    weights=bad_weights,
                    word2SentenceMerge=mergingmethods.sumMerge,
                    sentence2GroupMerge=mergingmethods.sumMerge,
                    verbose=False)
                del meanEmbeddings # avoid pylint warning
            except IndexError: # can happen on edgecase of dropped word
                raise ValueError


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