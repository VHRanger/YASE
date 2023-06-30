"""
Separate tests here for arrow data formats
"""
import pyarrow as pa

import unittest
from unittest import TestCase


import embeddinglib
from embeddinglib import mergeEmbeddings
from embeddinglib.stringprocessing import tokenizeCol, tokenize
from embeddinglib import stringprocessing, mergingmethods, embedding, encoders

import common_test_setups as cts

testdata = pa.array(cts.testdata)

testmodel = {k: pa.array(v) for k, v in cts.testmodel.items()}

#
#
#
# https://arrow.apache.org/docs/python/generated/pyarrow.Table.html
#
#
# TODO: Add PyArrow processing in here
#
class TestArrowCompat(TestCase):
    """
    Testing the top level user functions
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = testdata
        tdata2 = testdata.cast(pa.binary()) # Wrong type -- needs to cast
        tmodel = cts.mockW2V(testmodel)

class TestArrowSKLAPIModels(TestCase):
    """
    Testing the top level user functions
    """
    def test_given_correct_input_output_correct(self):
        # Note this data is a list of lists (eg. one input each)
        # the outer list is not a correct input for tokenizeCol!
        tdata = pa.Table.from_pydict({
            "first": testdata, 
            "second": testdata,
        })
        tmodel = cts.mockW2V(testmodel)
        # m1 = encoders.ColumnEmbedder(
        #     tmodel, verbose=False, threads=1, replace_dict=None, min_rows=1e10)
        # res = [
        #     m1.transform(tdata),
        # ]
        # for r in res:
        #     self.assertTrue(r.iloc[0][0] > -9999999)




if __name__ == '__main__':
    unittest.main()