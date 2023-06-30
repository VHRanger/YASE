"""
Tests for fse library:

https://github.com/oborchers/Fast_Sentence_Embeddings
"""
from copy import deepcopy
import gensim
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Normalizer
import unittest
from unittest import TestCase
import tempfile

import embeddinglib
from embeddinglib import mergeEmbeddings
from embeddinglib.stringprocessing import tokenizeCol, tokenize
from embeddinglib import stringprocessing, mergingmethods, embedding, encoders, utils

from fse import Average, SplitIndexedList

testdata = [
    "157681 Miles. Only $12,849", 
    "Used 2006 Ford F-150",
    "\"city, state\" format (quotes)", 
    "different 'quotes'",
]

testmodel = {
    "only": np.array([1.0, -1.0]),
    "Miles": np.array([1.5, -1.5]),
    "Used": np.array([0.4, -1.0]),
    "city": np.array([0.2, -0.2]),
    "format": np.array([10.9, -3.99]),
    "quotes": np.array([3.0, -1.2]),
}

m = gensim.models.keyedvectors.Word2VecKeyedVectors(
    vector_size=len(testmodel[list(testmodel.keys())[0]])
)
m.key_to_index = testmodel
m.vectors = np.array(list(testmodel.values()))

with tempfile.NamedTemporaryFile() as tmp:
    utils.my_save_word2vec_format(
        binary=True, fname=tmp.name, total_vec=len(testmodel), 
        vocab=m.key_to_index, vectors=m.vectors
    )
    m2 = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(tmp.name, binary=True)

class TestFSECompat(TestCase):
    """
    Testing the top level user functions
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        # Check that FSE Works by itself
        model = Average(m2)
        model.train(SplitIndexedList(testdata))
        orig_res = model.sv.vectors
        self.assertTrue(model.sv.vector_size > 0)
        self.assertTrue(orig_res.sum() > -1e10)
        # Check that FSE works in embed_column
        res1 = encoders.embed_column(
            testdata, m2, verbose=False, 
            replace_dict=None, col_name=None
        )
        res2 = encoders.embed_column(
            testdata, Average(m2), verbose=False, 
            replace_dict=None, col_name=None
        )
        res3 = encoders.embed_column(
            SplitIndexedList(testdata), Average(m2), verbose=False, 
            replace_dict=None, col_name=None
        )
        self.assertTrue(res1.sum().sum() > -1e10)
        self.assertTrue((res2 - orig_res).sum().sum() < 10)
        self.assertTrue((res2 - res3).sum().sum() < 10)


    def test_columnembedder_fse(self):
        tdata = pd.DataFrame({
            "first": pd.Series(testdata), 
            "second": pd.Series(testdata)
        })
        ce = encoders.ColumnEmbedder(
            m2, verbose=False, threads=1, 
            replace_dict=None, min_rows=1e10,
        )
        res = ce.transform(tdata)
        self.assertTrue(res.sum().sum() > -1e10)

    def test_model_routing(self):
        # Model to be routed
        model = {"en": Average(m2)}
        model_router = lambda x : "en"
        # Check that FSE works in embed_column
        #    with model_routing
        res1 = encoders.embed_column(
            testdata, model, model_router=model_router,
            verbose=False, replace_dict=None, col_name=None
        )
        self.assertTrue(res1.sum().sum() > -1e10)