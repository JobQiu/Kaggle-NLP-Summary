#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 02:35:30 2018

@author: xavier.qiu
"""
from gensim.models import KeyedVectors
import json


def load_embedding(type):
    if "google" in type or "Google" in type:
        pass
    elif "glove" in type or "GloVe" in type:
        pass
    elif "fasttext" in type:
        pass
    else:
        raise Exception("Please choose a word embedding, google news/glove/fasttext")

    pass


def load_google_news(path='/content/data/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'):
    embeddings_index = KeyedVectors.load_word2vec_format(path, binary=True)
    return embeddings_index


def load_glove(path=""):
    pass


def load_fasttext(path=""):
    pass
