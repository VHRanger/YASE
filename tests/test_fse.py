"""
Tests for fse library:

https://github.com/oborchers/Fast_Sentence_Embeddings
"""
from copy import deepcopy
from yase.pandas import embedding, stringprocessing
import gensim
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Normalizer
import unittest
from unittest import TestCase
import tempfile

import yase
from yase import mergeEmbeddings
from yase.pandas.stringprocessing import tokenizeCol, tokenize
from yase import mergingmethods, encoders, utils

if not utils.HAS_FSE:
    print("DOESNT HAVE FSE LIBRARY -- NOT TESTING")
else:
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

    m = utils.dict_to_gensim(testmodel)
    m2 = utils.gensim_to_fse(m)

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