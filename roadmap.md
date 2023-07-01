# TODO: 

10. Use MUSE Models in 9

- pyarrow based processing
    - using: https://arrow.apache.org/docs/python/generated/pyarrow.compute.ascii_split_whitespace.html#pyarrow.compute.ascii_split_whitespace

-----
NEXT TASKS

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

- SGPT support: https://github.com/Muennighoff/sgpt

- native fasttext support: https://huggingface.co/blog/fasttext