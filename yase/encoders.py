"""
This file contains SKLearn API compatible encoders
"""
from operator import itemgetter
import gc
from yase.pandas import stringprocessing
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import time

import pyarrow as pa

import yase
from yase import utils

import fse

def route_models(
        column, model, model_router, 
        vector_size, verbose=True):
    """
    Map sentences to a model key
    Normally used to map multiple languages
    """
    # Array of keys
    routes = np.array([model_router(x) for x in column])
    res = np.zeros(shape=(len(column), vector_size))
    for k in model:
        idx = np.where((routes == k))[0]
        res[idx] = embed_column(
            itemgetter(*idx)(column), model[k],
            model_router=None,
            verbose=verbose
        )
    return res

    # TODO: If routes has a key thats missing in model, 
    #       log it here
    # TODO: Make fallback model as well

def embed_column(column, model, model_router=None, verbose=True,
                 replace_dict=None, col_name=None):
    """
    Generates embedding feature matrices for string columns
    embedding_model should be a gensim KeyedVectors object

    model : One of:
        - python dict
        - gensim.models.keyedvectors
        - fse.models.base_s2v.BaseSentence2VecModel
        - sentence_transformers.SentenceTransformer
        The first two get routed to either pandas or arrow engine
        FSE models use the FSE engine.
            NOTE: Ensure you pass a proper FSE model, not the raw keyedvectors
        sentence_transformers use their own engine as well.
    """
    if model_router is not None and not isinstance(model, dict):
        raise ValueError(f"""
            If model router is not null, the embedding model should be a dictionary
            router_key -> embedding_model. Got {model} instead
        """)
    start_t = time.time()
    # Get Model Engine
    engine = utils.get_model_engine(model)
    vector_size = utils.get_model_vector_size(model)
    # Set result column names
    # TODO: make get_column_name method
    if col_name == None:
        try:
            col_name = column.name
        except:
            col_name = "embeddings"
    if verbose:
        print(f"Processing: {col_name}")
    res = None
    # If routing by models, start here and recurse
    if model_router:
        if isinstance(column, fse.inputs.BaseIndexedList):
            raise ValueError("data must be raw string column, got {column}")
        res = route_models(
            column, model, model_router, 
            vector_size=vector_size, verbose=verbose)
    elif engine == "fse":
        # If input is not native fse type, cast to sentence embedding
        if not isinstance(column, fse.inputs.BaseIndexedList):
            column = fse.SplitIndexedList(list(column))
        model.train(column)
        res = model.sv.vectors
    elif engine == "sbert":
        import sentence_transformers
        res = model.encode(column)
    elif engine == "pandas":
        # Preprocessing
        col = utils.normalize_column(column)
        # Tokenize list of sentences
        tokens = stringprocessing.tokenize(
            col, lower=True, split=True, replaceDict=replace_dict
        )
        # create sentence embeddings from tokens
        res = yase.embedding.sentenceEmbedding(
            tokens, model, verbose=verbose
        )
        # Cleanup memory before treating next column
        weights = None
        tokens = None
        gc.collect()
    # Post Processing
    # TODO: This is inefficient
    #    Return arrow memory instead
    #    Avoid expensive transposition
    #    Make caller's job to cast back to DF
    #    (using pa.Table.from_arrays)
    my_embeddings = pd.DataFrame(
        res,
        columns=[col_name + "_" + str(i) 
                for i in range(vector_size)]
    )
    if verbose:
        print(f"{column.name} time: {time.time() - start_t :.2f}")
    return my_embeddings


class ColumnEmbedder(TransformerMixin, BaseEstimator):
    def __init__(
        self, embedding_model,
        model_router=None,
        verbose=True, threads=1, 
        replace_dict=None, min_rows=150_000,
        ):
        """
        Column embedder takes in df of string columns, outputs matrices of floats
        Parameters
        ----------
        embedding_model : Compatible Model. One of:
            - python dict
            - gensim.models.keyedvectors
            - fse.models.base_s2v.BaseSentence2VecModel
            - sentence_transformers.SentenceTransformer
            The first two get routed to either pandas or arrow engine
            FSE models use the FSE engine.
                NOTE: Ensure you pass a proper FSE model, not the raw keyedvectors
            sentence_transformers use their own engine as well.
            NOTE: if `model_routing` is enabled, 
        model_routing : function str -> str
            If using a dictionary of models for `embedding_model`, then this function
            reads each sentence and returns a key to the dict.
        threads : int 
            number of processes/threads to use
            Uses joblib to parallelize embedding jobs
        min_rows : int
            threshold below which threading is turned off
            since process creation is heavy, we turn off threading below
                a certain threshold of rows in input df
        """
        if model_router is not None and not isinstance(embedding_model, dict):
            raise ValueError(f"""
                If model router is not null, the embedding model should be a dictionary
                router_key -> embedding_model. Got {embedding_model} instead
            """)
        self.embedding_model = embedding_model
        self.model_router = model_router
        self.verbose = verbose
        self.threads = threads
        self.min_rows = min_rows
        self.replace_dict = replace_dict

    def fit(self, X, y=None):
        """fit is a noop on this model"""
        return self

    def transform(self, X):
        """Perform Embedding per column
        Takes in string columns, outputs matrices of floats
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The input column of strings
        """
        start_t = time.time()
        threads_to_use = self.threads
        if len(X) < self.min_rows:
            threads_to_use = 1
        # Make a list of columns depending on the DF library
        col_list = []
        if isinstance(X, pd.DataFrame):
            col_list = [X[c] for c in X.columns]
        elif isinstance(X, pd.Series):
            col_list = [X]
        elif isinstance(X, pa.Table):
            col_list = X.columns
        # TODO: Add vaex here
        else: # assume its a list of lists
            col_list = X
        # Run job in parallel per column
        res = Parallel(n_jobs=threads_to_use)(
            delayed(embed_column)
            (col, self.embedding_model, self.model_router, 
             verbose=self.verbose, replace_dict=self.replace_dict
            ) 
            for col in col_list
        )
        res = pd.concat(res, axis=1)
        gc.collect()
        if self.verbose:
            print(f"Embedding time: {time.time() - start_t :.2f}")
        return res


class RobustOrdinalEncoder:
    def __init__(self, missing_val="N/A", nan_ratio=0.):
        """
        It differs from LabelEncoder by handling new classes 
        and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. 
        It gives unknown class id

        missing_val: str
            Value of the missing value for the column
            Since columns are categorical type, this can be a str
        nan_ratio: float in [0, 1]
            Ratio of missing_val to be artificially infected into column
            This can help with model robustness 
        """
        self.missing_val = missing_val
        self.nan_ratio = nan_ratio

    def fit(self, X, y=None):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        """
        cats = (pd.Series(list(np.unique(X)) + [self.missing_val])
                  .astype('category')
                  .cat.categories
        )
        self.mapping_ = dict(zip(cats, np.arange(len(cats))))
        return self

    def transform(self, X, training=False):
        """
        This will transform the data_list to id list where the new values 
        get assigned to Unknown class
        """
        res = pd.Series(X)
        # If nan_ratio, map random rows to NaN
        # Only do this if training setting is ON
        # (don't inject NaNs at inference)
        if (self.nan_ratio > 0) and training:
            n_rows = int(len(res) * self.nan_ratio)
            res[res.sample(n_rows).index] = None
        res = res.map(self.mapping_)
        return res.fillna(self.mapping_[self.missing_val]).astype(np.int32)

    def fit_transform(self, X, training=False):
        self.fit(X)
        return self.transform(X, training=training)
