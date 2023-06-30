from numpy import (
    zeros, dtype, float32, ascontiguousarray, 
    fromstring
)
import pandas as pd

######################################
#                                    #
#         Check available libs       #
#                                    #
######################################
# Arrow
# TODO: will be requirement
HAS_PA = False
try:
    import pyarrow as pa
    HAS_PA = True
except: pass
# Vaex
HAS_VX = False
try:
    import vaex as vx
    HAS_VX = True
except: pass
# Gensim
HAS_GENSIM = False
try:
    import gensim
    HAS_GENSIM = True
except: pass
# FSE
HAS_FSE = False
try:
    import fse
    HAS_FSE = True
except: pass
# SBERT
HAS_SBERT = False
try:
    import sentence_transformers
    HAS_SBERT = True
except: pass


def get_model_engine(model):
    if isinstance(model, sentence_transformers.SentenceTransformer):
        return "sbert"
    elif isinstance(model, fse.models.base_s2v.BaseSentence2VecModel):
        return "fse"
    else:
        return "pandas"


def get_model_vector_size(model):
    res = None
    if isinstance(model, sentence_transformers.SentenceTransformer):
        # TODO: This is super hackish, find a better wayS
        res = model[-2].__dict__['word_embedding_dimension']
    elif isinstance(model, fse.models.base_s2v.BaseSentence2VecModel):
        res = model.sv.vector_size
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        res = model.vector_size
    elif isinstance(model, dict):
        try:
            first_val = next(iter(model.values()))
            # Try the len of the first model
            if isinstance(first_val,
                    (sentence_transformers.SentenceTransformer,
                    fse.models.base_s2v.BaseSentence2VecModel,
                    gensim.models.keyedvectors.KeyedVectors,
                    dict)):
                res = get_model_vector_size(first_val)
            else: # If the dict is an embedding dict, use it
                res = len(first_val)
        except:
            raise ValueError(f"Model {model} type {type(model)} -- is wrong") 
    return int(res)


# TODO: This exists only for the old embedder
# fse and sbert dont need this
def normalize_column(column):
    """
    Make sure everything is cast to arrow before working on it
    """
    col = column
    if isinstance(column, pd.Series):
        if column.dtype not in ['string']:
            col = column.astype('str')
    elif HAS_PA:
        if isinstance(column, pa.Array):
            if not column.type.equals(pa.string()):
                col = column.cast('string')
    return col


def to_utf8(text, errors='strict', encoding='utf8'):
    """
    Convert a unicode or bytes string in the given encoding into a utf8 bytestring.
    """
    if isinstance(text, str):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return str(text, encoding, errors=errors).encode('utf8')


def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """
    Store the weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).
    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with open(fname, 'wb') as fout:
        fout.write(to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(float32)
                fout.write(to_utf8(word) + b" " + row.tobytes())
            else:
                fout.write(to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
