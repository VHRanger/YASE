# class testEmbeddingMerging(TestCase):
#     """
#     basic testing of the mergeEmbeddings function
#     """
#     def test_given_empty_input_raises(self):
#         with self.assertRaises(ValueError):
#             mergeEmbeddings([])

#     def test_given_normal_input_MergesEmbeddings_normally(self):
#         tembeddings = given_test_embeddings()
#         tweights = given_test_weights()

#         # default weight behavior is multiplication on np.sum
#         # and multiplying by weights of 1 is same as not having weights
#         np.testing.assert_array_almost_equal(
#             mergeEmbeddings(tembeddings),
#             mergeEmbeddings(tembeddings, tweights[0]))

#         # passing weights in default behavior works
#         np.testing.assert_array_almost_equal(
#             mergeEmbeddings(2 * tembeddings),
#             mergeEmbeddings(tembeddings, tweights[3]))

#         # Assure that passing weights on default behavior doesn't delete them
#         # as the weights = None line
#         self.assertIsNotNone(tweights[0])

#         # passing empty weight vector raises error
#         with self.assertRaises(ValueError):
#             mergeEmbeddings(tembeddings, [])
        
#         # tautological check on numpy mean function
#         np.testing.assert_array_almost_equal(
#             mergeEmbeddings(tembeddings, method=np.mean, axis=0),
#             np.mean(tembeddings, axis=0))

#         # Same with different func signature (for kwargs messiness)
#         np.testing.assert_array_almost_equal(
#             mergeEmbeddings(tembeddings, None, np.mean, axis=0),
#             np.mean(tembeddings, axis=0))                                        

#          # PCA Merging method
#         pcae = mergeEmbeddings(tembeddings, tweights[2], 
#                             method=mergingmethods.pcaMerge)
#         avge = mergeEmbeddings(tembeddings, tweights[0], 
#                             method=mergingmethods.avgMerge)
#         np.testing.assert_array_almost_equal(pcae.shape, avge.shape)

    
#     def test_given_normal_input_predefined_funcs_merge_correctly(self):
#         tembeddings = given_test_embeddings()
#         tweights = given_test_weights()
#         # avgMerge function works as expected
#         np.testing.assert_array_almost_equal(
#             mergeEmbeddings(tembeddings, None, mergingmethods.avgMerge),
#             np.mean(tembeddings, axis=0))
        
#         # weighed average
#         # note you need to multiply each row (word) by the weights
#         np.testing.assert_array_almost_equal(
#             mergeEmbeddings(tembeddings, tweights[2], 
#                             method=mergingmethods.avgMerge),
#             np.mean(tembeddings * np.array(tweights[2])[:, np.newaxis], axis=0))


# class TestGroupedEmbedding(TestCase):
#     """
#     basic checks for groupedEmbedding function
#     """
#     def test_given_list_input_input_untouched(self):
#         tdocs = pd.DataFrame(given_tdocs())
#         tgroups = given_tgroups()
#         w2vec = given_mock_keyedVectors()
#         gembeddings = embedding.groupedEmbedding(
#             tdocs, tgroups, model=w2vec, verbose=False)
#         self.assertTrue(type(tdocs) == list and type(tgroups) == list,
#                         msg="input should be unchanged")
#         self.assertEqual(len(gembeddings), len(set(tgroups)),
#                         msg="groupe dict should be # of categories")

    # def test_give_input_functions_applies_correctly(self):
    #     """
    #     test function from mergingmethods.py works with tautological equivalent
    #     """
    #     tdocs = given_tdocs()
    #     tgroups = given_tgroups()
    #     w2vec = given_mock_keyedVectors()
    #     meanEmbeddingsManual = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec, weights=None,
    #         word2SentenceMerge=np.mean,
    #         word2SentenceKwargs={"axis": 0},
    #         sentence2GroupMerge=np.mean,
    #         sentence2GroupKwargs={"axis": 0},
    #         verbose=False)
    #     meanEmbeddings = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec, weights=None,
    #         word2SentenceMerge=mergingmethods.avgMerge,
    #         sentence2GroupMerge=mergingmethods.avgMerge,
    #         verbose=False)
    #     for key in meanEmbeddings:
    #         np.testing.assert_array_almost_equal(
    #             meanEmbeddingsManual[key], 
    #             meanEmbeddings[key])

    # def test_given_pooling_applies_correctly(self):
    #     """
    #     test function from mergingmethods.py works with tautological equivalent
    #     """
    #     tdocs = given_tdocs()
    #     tgroups = given_tgroups()
    #     w2vec = given_mock_keyedVectors()
    #     meanEmbeddingsManual = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec,
    #         word2SentenceMerge='pooled',
    #         sentence2GroupMerge=np.mean,
    #         sentence2GroupKwargs={"axis": 0},
    #         verbose=False)
    #     meanEmbeddings = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec,
    #         word2SentenceMerge='pooled',
    #         sentence2GroupMerge=mergingmethods.avgMerge,
    #         verbose=False)
    #     for key in meanEmbeddings:
    #         np.testing.assert_array_almost_equal(
    #             meanEmbeddingsManual[key],
    #             meanEmbeddings[key])

    # def test_given_unique_pooling_applies_correctly(self):
    #     """
    #     test function from mergingmethods.py works with tautological equivalent
    #     """
    #     tdocs = given_tdocs()
    #     tgroups = given_tgroups()
    #     w2vec = given_mock_keyedVectors()
    #     meanEmbeddingsManual = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec,
    #         word2SentenceMerge='unique',
    #         sentence2GroupMerge=np.mean,
    #         sentence2GroupKwargs={"axis": 0},
    #         verbose=False)
    #     meanEmbeddings = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec,
    #         word2SentenceMerge='unique',
    #         sentence2GroupMerge=mergingmethods.avgMerge,
    #         verbose=False)
    #     for key in meanEmbeddings:
    #         np.testing.assert_array_almost_equal(
    #             meanEmbeddingsManual[key],
    #             meanEmbeddings[key])

    # def test_given_oov_applies_correctly(self):
    #     """
    #     Test that OOV words apply equally and don't break pipeline
    #     """
    #     tdocs = given_tdocs()
    #     tgroups = given_tgroups()
    #     w2vec = given_incomplete_keyedVectors()
    #     meanEmbeddingsManual = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec, weights=None,
    #         word2SentenceMerge=np.mean,
    #         word2SentenceKwargs={"axis": 0},
    #         sentence2GroupMerge=np.mean,
    #         sentence2GroupKwargs={"axis": 0},
    #         verbose=False)
    #     meanEmbeddings = embedding.groupedEmbedding(
    #         tdocs, tgroups, model=w2vec, weights=None,
    #         word2SentenceMerge=mergingmethods.avgMerge,
    #         sentence2GroupMerge=mergingmethods.avgMerge,
    #         verbose=False)
    #     for key in meanEmbeddings:
    #         np.testing.assert_array_almost_equal(
    #             meanEmbeddingsManual[key],
    #             meanEmbeddings[key])

    # def test_given_weights_tautological_works(self):
    #     """ 
    #     test that weights work normally in a tautological check
    #     """
    #     meanEmbeddings = embedding.groupedEmbedding(
    #         given_tdocs(), given_tgroups(), 
    #         model=given_incomplete_keyedVectors(), 
    #         weights=given_normalized_tweights(),
    #         word2SentenceMerge=mergingmethods.sumMerge,
    #         sentence2GroupMerge=mergingmethods.sumMerge,
    #         verbose=False)
    #     w2vec = given_incomplete_keyedVectors()
    #     for key in meanEmbeddings:
    #         self.assertEqual(len(meanEmbeddings[key]),
    #                          w2vec.vector_size)

    # def test_SIF_works(self):
    #     """ 
    #     test that components_to_remove works in grouped embedding merging
    #     """
    #     # Just there to integration test normalization
    #     meanEmbeddings_n = embedding.groupedEmbedding(
    #         given_tdocs(), given_tgroups(), 
    #         model=given_incomplete_keyedVectors(), 
    #         weights=given_normalized_tweights(),
    #         word2SentenceMerge=mergingmethods.sumMerge,
    #         sentence2GroupMerge=mergingmethods.avgMerge,
    #         sentence2GroupKwargs={"components_to_remove": 1, 
    #                               "normalize": True},
    #         verbose=False)
    #     meanEmbeddings = embedding.groupedEmbedding(
    #         given_tdocs(), given_tgroups(), 
    #         model=given_incomplete_keyedVectors(), 
    #         weights=given_normalized_tweights(),
    #         word2SentenceMerge=mergingmethods.sumMerge,
    #         sentence2GroupMerge=mergingmethods.avgMerge,
    #         sentence2GroupKwargs={"components_to_remove": 1, 
    #                               "normalize": False},
    #         verbose=False)
    #     w2vec = given_incomplete_keyedVectors()
    #     for key in meanEmbeddings:
    #         self.assertEqual(len(meanEmbeddings[key]),
    #                          w2vec.vector_size)
    #         self.assertEqual(len(meanEmbeddings_n[key]),
    #                 w2vec.vector_size)

    # def test_given_weight_embedding_mismatch_raises(self):
    #     """weight dimension != embedding dimension should raise"""
    #     bad_weights = given_normalized_tweights()
    #     bad_weights[0] = bad_weights[0][:-1]
    #     bad_weights[1] = bad_weights[0][:-2]
    #     with self.assertRaises(ValueError):
    #         try:
    #             meanEmbeddings = embedding.groupedEmbedding(
    #                 given_tdocs(), given_tgroups(), 
    #                 model=given_incomplete_keyedVectors(), 
    #                 weights=bad_weights,
    #                 word2SentenceMerge=mergingmethods.sumMerge,
    #                 sentence2GroupMerge=mergingmethods.sumMerge,
    #                 verbose=False)
    #             del meanEmbeddings # avoid pylint warning
    #         except IndexError: # can happen on edgecase of dropped word
    #             raise ValueError