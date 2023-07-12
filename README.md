# Yet Another Sentence Embedding Library

The goal of this library is to make it easy to transform lists of sentences or sets of sentences into a matrix of embeddings (eg. one per sentence). This can be done either at the sentence/document level or by grouping sentence embeddings into grouped embeddings.

Such matrices of documents can easily be queried using kd-trees (see notebook in examples) for the most similar document in training data to a queried sentence. It can also be used to cluster document groups together solely by the text in the campaign.

The results can be tested for quality on a handcrafted evaluation dataset by checking how well the sentence embeddings cluster around the natural clusters of the existing ad campaigns.


# Examples

The library works out of the box with [gensim]() models, [sentence-transformers]() models, and [fse]() models.

```python
    import yase

    df = ...

    # Gensim Model
    import gensim.downloader as model_api
    gensim_model = model_api.load("glove-wiki-gigaword-300")
    gsb = yase.encoders.embed_column(df.body, model=gensim_model,verbose=True)
    
    # sentence-transformers model
    from sentence_transformers import SentenceTransformer
    sb_model = SentenceTransformer('all-MiniLM-L6-v2')
    rsb = yase.encoders.embed_column(df.body, model=sb_model, verbose=True)

    # FSE model
    from fse import Vectors, Average
    vecs = Vectors.from_pretrained("glove-wiki-gigaword-50")
    fse_model = Average(vecs)
    fsb = yase.encoders.embed_column(df.body, model=fse_model,verbose=True)

```

# Model Routing by Language Detection


## Running unit tests
```
python -m unittest discover tests
```
