TODO: Rename to textrep

TODO: add support for weights in new embedder

TODO: test new embedder more rigorously
      separate text test from embedding testing

TODO: fix wordweigts to pandas only version

TODO: add ciseau to string tokenization: https://github.com/JonathanRaiman/ciseau

TODO: Add WordPierce style tokenization: https://stackoverflow.com/questions/55382596/how-is-wordpiece-tokenization-helpful-to-effectively-deal-with-rare-words-proble/55416944#55416944

TODO: Add LM (BERT, etc.) embedding support
    https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html
    http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

# NLP utilities library

The goal of this library is to make it easy to transform lists of sentences or sets of sentences into a matrix of embeddings (eg. one per sentence). This can be done either at the sentence/document level or by grouping sentence embeddings into grouped embeddings.

Such matrices of documents can easily be queried using kd-trees (see notebook in examples) for the most similar document in training data to a queried sentence. It can also be used to cluster document groups together solely by the text in the campaign.

The results can be tested for quality on a handcrafted evaluation dataset by checking how well the sentence embeddings cluster around the natural clusters of the existing ad campaigns.

The entire pipeline can be done in 4 lines:
```python
    import gensim.downloader as model_api
    import embeddinglib
    # Load pretrained gensim model
    model = model_api.load("glove-wiki-gigaword-300")
    # Tokenize list of sentences 
    tokens = stringprocessing.tokenize(ads['DESCRIPTION'], lower=True, split=True)
    # get word weights for higher quality embeddings
    weights = embedding.getWordWeights(col, "tf-idf")
    # create sentence embeddings from tokens
    my_embeddings = embedding.sentenceEmbedding(tokens, model, weights)
```
