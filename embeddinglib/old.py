def OOVWordStats(sentences: Iterable[Iterable[str]],
                 model,
                 ) -> dict:
    """
    Gets words out of Word2Vec model vocabulary

    :param sentences: 
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
        If they aren't, use stringprocessing.tokenize beforehand
    :param model: a trained embedding model 
        (usually gensim Keyedvectors object)
        Needs to support ["word"] -> vector style API
    :type sentences: Iterable[str]

    :return: a dictionary mapping OOV words to their number of occurences
            and a list of the rows where they appear
    :rtype: dict(word -> [number occurences, [row locations]])
    """
    # return value of missing words dict
    unique_words = set()
    total_words = 0
    total_missing = 0
    missing_rows = set()
    missing_words = collections.defaultdict(lambda: [0, []])
    for row_num in range(len(sentences)):
        row = sentences[row_num]
        for word in row:
            total_words += 1
            try:
                unique_words.add(word)
                model[word]
            except KeyError:
                missing_rows.add(row_num)
                total_missing += 1
                missing_words[word][0] += 1
                missing_words[word][1].append(row_num)
    res = {}
    res['unique_words'] = unique_words
    res['total_words'] = total_words
    res['total_missing'] = total_missing
    res['missing_words'] = missing_words.keys()
    res['missing_words_dict'] = missing_words
    res['missing_rows_list'] = missing_rows
    res['total_rows'] = len(sentences)
    res['oov_row_pct']= res['total_missing'] / res['total_words']
    return res


#
# TODO: remove, doesn't do anything
#
def mergeEmbeddings(embeddings: Iterable[Iterable[float]],
                    weights: Iterable[float]=None,
                    method: FunctionType=mergingmethods.sumMerge,
                    **kwargs) -> np.array:
    """
    Takes a list of embeddings and merges them into a single embeddings.
    Typical usecase is to create a sentence embedding from word embeddings.

    :param embeddings:
        Embedding matrix. 
        Each row represents one observation (eg. one word if we're merging a sentence embedding)
        for instance, an array of word embeddings
        note that weird numbers (np.NaN, np.inf, etc.) are undefined behavior
    :param weights: 
        weights on the embeddings, to be used in the method called
        weird numbers are undefined behavior here too
    :param method:
        A method is passed that takes either the format 
            Iterable[Iterable[float]], **kwargs 
        eg. the embeddings and of keyword arguments
        or the format 
            Iterable[Iterable[float]], Iterable[float], **kwargs
        which is (embeddings, weights, {keyword arguments})
    
        Methods include:
            "avg": method=np.sum, axis=0
                elementwise mean of word vectors. 
                This can be quality-degrading due to semantically meaningless 
                components having large values (see ICLR (2017), p.2, par.4)
            "sum": method=np.mean, axis=0
                elementwise sum of word vectors. 
                Also called "word centroid" in literature

    :return: a single array merged by the procedure
    :rtype: np.array[float]
    :raises: ValueError
    ------------------------------------------------------------------
    references:
        "Simple but tough to beat baseline", Arora et al. (ICLR 2017)
    """
    if len(embeddings) == 0:
        raise ValueError("embeddings input empty!")
    if weights is not None:
        if type(weights) != np.array:
            weights = np.array(weights)
        if weights.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "Incorrect shape of weights! weights: {0}, embeddings: {1}".format(
                     weights.shape, embeddings.shape))
    try:
        if weights is None:
            return method(embeddings, **kwargs)
        return method(embeddings, weights, **kwargs)
    except TypeError as te:
        print(("\n\nError calling defined method.\n "
               + "method called: {0}\n").format(method),
            "\n\nNOTE: This can happen if you are passing weights "
            "in a function that doesn't take them as the second argument!\n"
            "Function signature was:\n\t {0}".format(inspect.signature(method)),
            ("\nArgs passed were:"
              + "\n\tembeddings: {0}"
              + "\n\tweights: {1}"
              + "\n\tkwargs: {2}").format(
                  embeddings, weights, kwargs))
        raise(te)


def groupedEmbedding(documentCol: Iterable[Iterable[str]],
                     groupKeyCol: Iterable[int],
                     model,
                     weights: Iterable[Iterable[float]]=None,
                     word2SentenceMerge: FunctionType=mergingmethods.avgMerge,
                     word2SentenceKwargs: dict={},
                     sentence2GroupMerge: FunctionType=mergingmethods.avgMerge,
                     sentence2GroupKwargs: dict={},
                     verbose=True
                    ) -> dict:
    """
    Creates embeddings for Groups of documents
    Does this by first embedding the documents 
        then embedding each document embedding into a single embedding per group

    For instance, you can create paragraph embeddings by passing a list of split 
    sentences with groupkeycol being the paragraph number on each sentence.

    Methods for each embedding merging methods are input as params

    NOTE: This method expects the document column to be already split 
        and preprocessed (so in series of ['word', 'another', 'word'] format)

    returns a dictionary of each groupkey -> group embedding

    :param documentCol: 
        A iterable of iterables of words. 
        We assume the documents are already split in word lists
        eg. [["sale", "ford"], ["cheap", "good"]]
    :param groupKeyCol:
        a vector of group keys co-indexed with each document in documentCol.
        This could be each paragraph a document belongs to, or author, etc.
    :param model: a trained embedding model 
        (usually gensim Keyedvectors object)
        Needs to support ["word"] -> vector style API
    :param weights:
        weights on each words, to be used in the embedding merging method called
    :param word2SentenceMerge:
        Function or string in ['pooled', 'unique']
        If 'pooled', all the words are bucketed for the group 
            then merged as if it were a single sentence.
        If 'unique', all the unique words are bucketed for the group 
            then merged as if it were a single sentence.
        The function to merge word embeddings into sentence embeddings
            (passed to mergeEmbeddings.)
        Function is usually from mergingmethods.py
        Should respect the require function signature
            (embeddings, weights, {keyword arguments}) if weights are passed
            (embeddings, {keyword arguments}) if no weights are passed
    :param word2SentenceKwargs:
        keyword arguments for above function
    :param sentence2GroupMerge:
        function to merge sentence embeddings into grouped embeddings
        same requirements as word2SentenceMerge functions
    :param sentence2GroupKwargs:
        kwargs for above
    :param verbose:
        whether to print info output

    :return: a dict[group] -> group_embedding 
             for all groups in groupKeyCol
             where embeddings are merged from words to sentence
             then from sentence to group
    """
    if type(groupKeyCol) != pd.Series:
        groupKeyCol = pd.Series(groupKeyCol)
    if weights is not None and type(weights) != pd.Series:
        weights = np.array(weights, dtype='object')
    if len(documentCol) != len(groupKeyCol):
        raise ValueError("documentCol and groupKeycol should be co-indexed!"
                         "groupKeyCol is the group for each coindexed document")
    elif weights is not None and len(documentCol) != len(weights):
        raise ValueError("documentCol, weights should be co-indexed!"
                         "weights are weights for each word in coindexed document")
    groupEmbeddings = {}
    # If there's a sentence merging method, apply it before merging for each group
    if type(word2SentenceMerge) is FunctionType:
        for group in groupKeyCol.unique():
            # find documents for the group
            indices = documentCol.index[groupKeyCol == group].tolist()
            SentenceEmbeddings = sentenceEmbedding(
                documentCol.iloc[indices],
                model,
                weights[indices] if weights is not None else None,
                mergingFunction=word2SentenceMerge,
                mergingFunctionKwargs=word2SentenceKwargs,
                verbose=verbose)
            try:
                groupEmbeddings[group] = mergeEmbeddings(
                    SentenceEmbeddings,
                    weights=None, # TODO (mranger): add 2nd level weights fn??
                    method=sentence2GroupMerge,
                    **sentence2GroupKwargs)
            except ValueError as ve:
                if verbose:
                    err_str = "Bad sentence Embeddings: {0}.  {1}".format(
                                    SentenceEmbeddings, ve)
                    if verbose == 'log':
                        logger.warning(err_str)
                    else:    
                        print(err_str)
                pass
    # If word2SentenceMerge is None, add all words in 
    #    a group into one unordered "sentence"
    elif word2SentenceMerge in ['pooled', 'unique']:
        gp = pd.DataFrame({
            'data': documentCol,
            'groups': groupKeyCol
        })
        gp = gp.groupby('groups').agg({'data': list})
        # TODO: Should optionally .apply(np.unique) be here??
        gp.data = gp.data.apply(np.concatenate)
        if word2SentenceMerge == 'unique':
            gp.data = gp.data.apply(np.unique)
        gc.collect()
        try:
            embeds = sentenceEmbedding(
                gp.data,
                model,
                # TODO: add weights to pooled merge??
                mergingFunction=sentence2GroupMerge,
                mergingFunctionKwargs=sentence2GroupKwargs,
                verbose=verbose)
            gc.collect()
            groupEmbeddings = dict(zip(gp.index, embeds))
        except ValueError as ve:
            if verbose:
                err_str = "Bad sentence Embeddings: {0}.  {1}".format(
                                SentenceEmbeddings, ve)
                print(err_str)
            pass
    else:
        raise ValueError(
            'word2SentenceMerge should be a function, "pooled" or "unique"')
    return groupEmbeddings
