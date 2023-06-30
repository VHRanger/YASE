import numpy as np
import random

# TODO: remove gensim from here
# Also requires removing it from mockw2v
# which also means removing vector_size from other parts
import gensim

#################################
#                               #
# Test Data, mock models, etc.  #
#                               #
#################################

testdata = [
    "157681 Miles. Only $12,849", 
    "Used 2006 Ford F-150",
    "\"city, state\" format (quotes)", 
    "different 'quotes'",
]

testmodel = {
    "only": [1.0, -1.0],
    "Miles": [1.5, -1.5],
    "Used": [0.4, -1.0],
    "city": [0.2, -0.2],
    "format": [10.9, -3.99],
    "quotes": [3.0, -1.2],
}

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

class mockW2V(gensim.models.keyedvectors.KeyedVectors):
    """Mock gensim.models.keyedvectors class"""
    def __init__(self, embeddingDict):
        """init based on a word embedding dict[str -> np.array]"""
        self.embeddings = embeddingDict
        self.words = list(embeddingDict.keys())
        self.vector_size = len(self.embeddings[self.words[0]])
        for word in self.words:
            assert len(self.embeddings[word]) == self.vector_size, (
                "vector length not the same for all mock word2vec model input")

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
