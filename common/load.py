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


def load_embedding(type):
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
