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
import pickle
from tqdm import tqdm


class DataSet:

    def __init__(self, embedding='glove', voc_len=105000, max_ques_len=72, cache=True):
        """

        :param embedding:
        """
        self.config = load_config()
        self.embedding_type = embedding
        self.voc_len = voc_len
        self.max_ques_len = max_ques_len

        if cache and os.path.exists(os.path.join(self.config["data_dir"], "y_train.pickle")):
            with open(os.path.join(self.config["data_dir"], "x_train.pickle"), 'rb') as handle:
                self.x_train = pickle.load(handle)
            with open(os.path.join(self.config["data_dir"], "x_test.pickle"), 'rb') as handle:
                self.x_test = pickle.load(handle)
            with open(os.path.join(self.config["data_dir"], "y_train.pickle"), 'rb') as handle:
                self.y_train = pickle.load(handle)
            with open(os.path.join(self.config["data_dir"], "embedding_matrix.pickle"), 'rb') as handle:
                self.embedding_matrix = pickle.load(handle)

            return

        print("Loading Train df")
        self.train_df = pd.read_csv(os.path.join(self.config["data_dir"], "train.csv"))
        print("Loading Test df")
        self.test_df = pd.read_csv(os.path.join(self.config["data_dir"], "test.csv"))

        self.preprocess("train")
        self.preprocess("test")
        self.word_index = None
        # convert question_text to question_ids_list
        self.word2indices()

        print("Loading Embedding - {}".format(embedding))

        self.embedding_index = load_embedding(self.embedding_type, word_index=self.word_index, voc_len = self.voc_len)
        if self.embedding_type != "mix":
            self.embedding_matrix = self.make_embed_matrix(self.embedding_index, self.word_index, self.voc_len)
        else:
            self.embedding_matrix = self.embedding_index

        del self.word_index
        del self.embedding_index
        send_msg("Load Done")
        gc.collect()


        with open(os.path.join(self.config["data_dir"], "x_train.pickle"), 'wb') as handle:
            pickle.dump(self.x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.config["data_dir"], "x_test.pickle"), 'wb') as handle:
            pickle.dump(self.x_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.config["data_dir"], "y_train.pickle"), 'wb') as handle:
            pickle.dump(self.y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.config["data_dir"], "embedding_matrix.pickle"), 'wb') as handle:
            pickle.dump(self.embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def make_embed_matrix(self, embeddings_index, word_index, len_voc):
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        word_index = word_index
        embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))

        for word, i in tqdm(word_index.items()):
            if i >= len_voc:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def word2indices(self):

        t = Tokenizer(num_words=self.voc_len, filters='')

        x_train = self.train_df["treated_question"].fillna("_na_").values
        x_test = self.test_df["treated_question"].fillna("_na_").values

        t.fit_on_texts(list(x_train))

        self.word_index = t.word_index

        # Tokenize the sentences
        x_train = t.texts_to_sequences(x_train)
        x_test = t.texts_to_sequences(x_test)

        # Pad the sentences
        x_train = pad_sequences(x_train, maxlen=self.max_ques_len)
        x_test = pad_sequences(x_test, maxlen=self.max_ques_len)

        # Get the target values
        y_train = self.train_df['target'].values

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train

    def preprocess(self, data_set, filters=["punct", "contraction", "special characters", "misspell"]):
        """

        :param filters:
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
