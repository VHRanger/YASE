"""
Tests for sbert library:

https://www.sbert.net/

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
from embeddinglib import (
    stringprocessing, mergingmethods, 
    embedding, encoders, utils
)

from sentence_transformers import SentenceTransformer

testdata = [
    "157681 Miles. Only $12,849", 
    "Used 2006 Ford F-150",
    "\"city, state\" format (quotes)", 
    "different 'quotes'",
]

testmodel = SentenceTransformer('all-MiniLM-L6-v2')


class TestSBertCompat(TestCase):
    """
    Testing the top level user functions
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        orig_res = testmodel.encode(testdata)
        self.assertTrue(orig_res.sum() > -1e10)
        # Check that SBert works in embed_column
        res1 = encoders.embed_column(
            testdata, testmodel, verbose=False, 
            replace_dict=None, col_name=None
        )
        self.assertTrue(res1.sum().sum() > -1e10)

    def test_columnembedder_sbert(self):
        tdata = pd.DataFrame({
            "first": pd.Series(testdata), 
            "second": pd.Series(testdata)
        })
        ce = encoders.ColumnEmbedder(
            testmodel, verbose=False, threads=1, 
            replace_dict=None, min_rows=1e10,
        )
        res = ce.transform(tdata)
        self.assertTrue(res.sum().sum() > -1e10)

    def test_model_routing(self):
        # Model to be routed
        model = {"en": testmodel}
        model_router = lambda x : "en"
        # Check that FSE works in embed_column
        #    with model_routing
        res1 = encoders.embed_column(
            testdata, model, model_router=model_router,
            verbose=False, replace_dict=None, col_name=None
        )
        self.assertTrue(res1.sum().sum() > -1e10)