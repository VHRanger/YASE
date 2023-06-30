# TODO: 

10. fix imports -- rename -- move pandas stuff to pandas folder

10. Use MUSE Models in 9

-----
NEXT TASKS

- pyarrow based processing
    - using: https://arrow.apache.org/docs/python/generated/pyarrow.compute.ascii_split_whitespace.html#pyarrow.compute.ascii_split_whitespace

- Support Vaex

- Caching to avoid repeat embeddings

- support https://huggingface.co/blog/fasttext

- add support for weights in new embedder

- Add vaex streaming disk-to-disk support

- support SGPT https://github.com/Muennighoff/sgpt

- test new embedder more rigorously
      separate text test from embedding testing

- support PolaRS (maybe just through PyArrow)

- Add WordPierce style tokenization: https://stackoverflow.com/questions/55382596/how-is-wordpiece-tokenization-helpful-to-effectively-deal-with-rare-words-proble/55416944#55416944 (also in BERTTokenizer)

- Support making the whole pipeline into an object to put in other models

- make own fast se lib

# Yet Another Sentence Embedding Library

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


## Running unit tests

```
python -m unittest discover tests
```
