TODO: add ciseau to string tokenization: https://github.com/JonathanRaiman/ciseau

TODO: add GLoVe as default embedding model (see csrgraphs GLoVe implementation)



# NLP utilities library

The goal of this library is to make it easy to transform lists of sentences or sets of keywords into a matrix of embedding. This can be done either at the sentence/document level or by grouping sentence embeddings into grouped embeddings.

Such matrices of documents can easily be queried using kd-trees (see notebook in examples) for the most similar document in training data to a queried sentence. It can also be used to cluster campaigns together solely by the text in the campaign.

The results can be tested for quality on a handcrafted evaluation dataset by checking how well the sentence embeddings cluster around the natural clusters of the existing ad campaigns.

The entire pipeline can be done in 4 lines:

    import gensim.downloader as model_api
    # Load pretrained gensim model
    
    model = model_api.load("glove-wiki-gigaword-300")
    # Tokenize list of sentences 
    tokens = stringprocessing.tokenize(ads['DESCRIPTION'], lower=True, split=True)
    
    # get word weights for higher quality embeddings
    weights = embedding.getWordWeights(col, "tf-idf")
    
    # create sentence embeddings from tokens
    my_embeddings = embedding.sentenceEmbedding(tokens, model, weights)

# REMAINING TODOs:

Add utilities to train own word2vec, retrain over existing word2vec, or doc2vec sentences. See here: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

Add other evaluation benchmarks (quora duplicate questions dataset, SemEval, ...)

Add other merging methods.

Add utilities to train weights on original word2vec corpus (instead of inference corpus)
