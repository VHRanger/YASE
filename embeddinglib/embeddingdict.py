
# TODO: support poincarre embeddings
class EmbeddingDict(dict):
    """
    Helper class to hold embedding model in raw python dict
    While respecting gensim API
    """
    def __init__(self, tup):
        super().__init__(tup)
        self.vector_size = None