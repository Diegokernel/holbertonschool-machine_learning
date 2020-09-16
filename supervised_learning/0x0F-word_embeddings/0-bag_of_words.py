#!/usr/bin/env python3
"""
0. Bag Of Words
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
    """

    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
