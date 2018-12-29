#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:52:19 2018

@author: xavier.qiu
"""

from common.load import *
from common.pd_util import *
from common.preprocess import *
from common.util import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gc
import json
import pickle


class DataSet:

    def __init__(self, embedding='glove', voc_len = 95000, max_ques_len = 60, cache = False ):
        """

        :param embedding:
        """
        self.config = load_config()

        if cache and os.path.exists(os.path.join(self.config["data_dir"], "train_cache.csv")) and  os.path.exists(os.path.join(self.config["data_dir"], "test_cache.csv")):

            print("Loading Train df")
            self.train_df = pd.read_csv(os.path.join(self.config["data_dir"], "train_cache.csv"))
            print("Loading Test df")
            self.test_df = pd.read_csv(os.path.join(self.config["data_dir"], "test_cache.csv"))

            return


        print("Loading Train df")
        self.train_df = pd.read_csv(os.path.join(self.config["data_dir"], "train.csv"))
        print("Loading Test df")
        self.test_df = pd.read_csv(os.path.join(self.config["data_dir"], "test.csv"))
        self.embedding_type = embedding
        print("Loading Embedding - {}".format(embedding))
        self.embedding_index = load_embedding(self.embedding_type)

        self.preprocess("train")
        self.preprocess("test")

        self.voc_len = voc_len
        self.max_ques_len = max_ques_len

        self.word_index = None
        # convert question_text to question_ids_list
        self.word2indices()

        self.embedding_matrix = self. make_embed_matrix(self.embedding_index, self.word_index, self.voc_len)

        del self.word_index
        del self.embedding_index
        gc.collect()

        if cache:
            self.train_df.to_csv(os.path.join(self.config["data_dir"], "train_cache.csv"))
            self.test_df.to_csv(os.path.join(self.config["data_dir"], "test_cache.csv"))

        pass

    def make_embed_matrix(self, embeddings_index, word_index, len_voc):
        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        word_index = word_index
        embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))

        for word, i in word_index.items():
            if i >= len_voc:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def word2indices(self):

        t = Tokenizer(num_words=self.voc_len)
        t.fit_on_texts(self.train_df['treated_question'])
        self.word_index = t.word_index

        for dataset in [self.train_df, self.test_df]:
            dataset['question_ids'] = t.texts_to_sequences(dataset['treated_question'])
            dataset['question_ids'] = pad_sequences(dataset['question_ids'], maxlen=self.max_len)


    def getTrain(self):
        return self.train_df

    def getTest(self):
        return self.test_df

    def getEmbeddingMatrix(self):
        return self.embedding_matrix

    def preprocess(self, data_set, filters = [ "punct", "contraction", "special characters","misspell"]):
        """

        :param data_set:
        :return:
        """

        if data_set == "train":
            df = self.train_df
        else:
            df = self.test_df
        print("Pre-processing {}".format(data_set))
        df["treated_question"] = df["question_text"]

        if "numbers" in filters:
            print("Clean number ing ... ")
            df["treated_question"] = df["treated_question"].apply(lambda x: deal_with_numbers(x))

        if "punct" in filters:
            print("Clean punct ing ... ")
            df['treated_question'] = df['treated_question'].apply(lambda x: deal_with_punct(x))

        if "lower" in filters:
            print("Lowering ... ")
            df['treated_question'] = df['treated_question'].apply(lambda x: x.lower())

        if "special characters" in filters:
            print("Clean special chars ing ... ")
            df['treated_question'] = df['treated_question'].apply(lambda x: deal_with_special_characters(x))

        if "misspell" in filters:
            print("Clean misspell ing ...")
            df['treated_question'] = df['treated_question'].apply(lambda x: deal_with_misspell(x))


        vocab = count(df['question_text'])

        print("Calculating coverage ... ")
        oov = check_coverage(vocab, self.embedding_index)
        print(oov[:20])

        print("-" * 20)
        send_msg("Load Done")

    def compress_embedding(self ):
