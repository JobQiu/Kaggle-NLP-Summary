#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 02:35:30 2018

@author: xavier.qiu
"""
from gensim.models import KeyedVectors
from common.util import load_config
import os


def load_embedding(type):
    """

    :param type:
    :return:
    """
    config = load_config()
    # print(config)

    # load json to know where are the embedding files

    if "google" in type or "Google" in type:
        return load_google_news(config['google_news_path'])
    elif "glove" in type or "GloVe" in type:
        pass
    elif "fasttext" in type:
        pass
    else:
        raise Exception("Please choose a word embedding, google news/glove/fasttext")


def load_google_news(path='/content/data/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'):
    if not os.path.exists(path):
        raise Exception("google_news not exists, please download and modify the path in config.json")
    embeddings_index = KeyedVectors.load_word2vec_format(path, binary=True)
    return embeddings_index


def load_glove(path=""):
    if not os.path.exists(path):
        raise Exception("glove not exists, please download and modify the path in config.json")

    pass


def load_fasttext(path=""):
    if not os.path.exists(path):
        raise Exception("fasttext not exists, please download and modify the path in config.json")

    pass
