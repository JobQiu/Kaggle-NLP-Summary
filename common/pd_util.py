#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 03:09:24 2018

@author: xavier.qiu
"""
from collections import Counter
import pandas as pd
from tqdm import tqdm
import gc
import operator

tqdm.pandas()


def check_coverage(vocab, embeddings_index):
    """

    :param vocab: the output Counter from the count method
    :param embeddings_index: embedding loaded from files
    :return: those words:their count that not find in word-embedding
    """
    a = []
    oov = {}
    find_amount = 0
    not_find_amount = 0
    for word in tqdm(vocab):
        if word in embeddings_index:
            a.append(word)
            find_amount += vocab[word]
        else:
            oov[word] = vocab[word]
            not_find_amount += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab     [kinds]'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text [amount]'.format(find_amount / (find_amount + not_find_amount)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    del a
    gc.collect()

    return sorted_x


def count(pd_series, verbose=True):
    """
    for example, vocab = count(train["question_text"])
c
    :param pd_series:
    :param verbose:
    :return: Counter of words in this pandas series
    """
    sentences = pd_series.progress_apply(lambda x: x.split()).values
    vocab = Counter()
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            vocab[word] += 1

    # save memory
    del sentences
    gc.collect()
    return vocab
