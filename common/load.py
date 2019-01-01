#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 02:35:30 2018

@author: xavier.qiu
"""
from gensim.models import KeyedVectors
from common.util import load_config
import os
import numpy as np
import gc


def load_embedding(type, word_index=None, voc_len=105000):
    """

    :param type:
    :param word_index:
    :return:
    """
    if type == "mix":
        config = load_config()
        embedding_matrix_1 = load_glove(word_index, config)
        embedding_matrix_3 = load_para(word_index, config)
        embedding_matrix = np.mean((embedding_matrix_1, embedding_matrix_3), axis=0)
        del embedding_matrix_1, embedding_matrix_3
        gc.collect()
        np.shape(embedding_matrix)
        return embedding_matrix
    else:
        return _load_embedding(type)
    pass


def load_glove(word_index, config):
    max_features = len(word_index) + 1
    EMBEDDING_FILE = config['glove_path']  # '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_fasttext(word_index, config):
    max_features = len(word_index) + 1
    EMBEDDING_FILE = config['wiki_news_path']  # '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index, config):
    max_features = len(word_index) + 1
    EMBEDDING_FILE = config['paragram_path']  # '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if
                            len(o) > 100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def _load_embedding(type):
    """
    :param type: google/glove/paragram/wiki=fasttext
    :return: word_embedding
    """
    config = load_config()
    # print(config)

    # load json to know where are the embedding files
    if "fasttext" in type:
        type = "wiki"

    if "google" in type or "Google" in type:
        return load_google_news(config['google_news_path'])
    elif "glove" in type or "GloVe" in type:
        path = config['glove_path']
    elif "paragram" in type:
        path = config['paragram_path']
    elif "wiki" in type:
        path = config['wiki_news_path']
    else:
        raise Exception("Please choose a word embedding, google news/glove/fasttext")

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if "wiki" in type:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(path) if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(path, encoding='latin'))

    return embeddings_index


def load_google_news(path='/content/data/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'):
    if not os.path.exists(path):
        raise Exception("google_news not exists, please download and modify the path in config.json")
    embeddings_index = KeyedVectors.load_word2vec_format(path, binary=True)
    return embeddings_index
