# Yet Another Sentence Embedding Library

The goal of this library is to make it easy to transform lists of sentences or sets of sentences into a matrix of embeddings (eg. one per sentence). This can be done either at the sentence/document level or by grouping sentence embeddings into grouped embeddings.

Such matrices of documents can easily be queried using kd-trees (see notebook in examples) for the most similar document in training data to a queried sentence. It can also be used to cluster document groups together solely by the text in the campaign.

The results can be tested for quality on a handcrafted evaluation dataset by checking how well the sentence embeddings cluster around the natural clusters of the existing ad campaigns.




### (Gensim) Weighed Sentence Embeddings with Gensim model
```python
    import gensim.downloader as model_api
    import yase
    # Load pretrained gensim model
    model = model_api.load("glove-wiki-gigaword-300")
    # Tokenize list of sentences 
    tokens = yase.tokenize(data, lower=True, split=True)
    # get word weights for higher quality embeddings
    weights = yase.getWordWeights(data, "tf-idf")
    # create sentence embeddings from tokens
    my_embeddings = embedding.sentenceEmbedding(tokens, model, weights)
```



## Running unit tests
```
python -m unittest discover tests
```
