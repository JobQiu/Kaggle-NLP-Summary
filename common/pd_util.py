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

tqdm.pandas()


def count(pd_series, verbose=True):
    """
    for example, vocab = count(train["question_text"])

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
