#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:52:19 2018

@author: xavier.qiu
"""

from common.load import *
from common.pd_util import *
from common.preprocess import *

tqdm.pandas()


class DataSet:

    def __init__(self, embedding='glove'):
        """

        :param embedding:
        """
        self.config = load_config()
        self.train_df = pd.read_csv(os.path.join(self.config["data_dir"], "train.csv"))
        self.test_df = pd.read_csv(os.path.join(self.config["data_dir"], "test.csv"))
        self.embedding_type = embedding

        self.preprocess("train")
        self.preprocess("test")

        self.embedding_index = load_embedding(self.embedding_type)

        pass

    def getTrain(self):
        return self.train_df

    def getTest(self):
        return self.test_df

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
            df["treated_question"] = df["treated_question"].progress_apply(lambda x: deal_with_numbers(x))

        if "punct" in filters:
            print("Clean punct ing ... ")
            df['treated_question'] = df['treated_question'].progress_apply(lambda x: deal_with_punct(x))

        if "lower" in filters:
            print("Lowering ... ")
            df['treated_question'] = df['treated_question'].progress_apply(lambda x: x.lower())

        if "special characters" in filters:
            print("Clean special chars ing ... ")
            df['treated_question'] = df['treated_question'].progress_apply(lambda x: clean_special_chars(x))

        if "misspell" in filters:
            print("Clean misspell ing ...")
            df['treated_question'] = df['treated_question'].progress_apply(lambda x: deal_with_misspell(x))


        vocab = count(df['question_text'])

        print("Calculating coverage ... ")
        oov = check_coverage(vocab, self.embedding_index)
        print(oov[:20])

        print("-" * 20)
