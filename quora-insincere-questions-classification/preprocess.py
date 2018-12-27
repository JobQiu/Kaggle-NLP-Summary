#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:52:19 2018

@author: xavier.qiu
"""
import pandas as pd
from common.util import load_config
import os


class DataSet():

    def __init__(self):
        self.config = load_config()
        self.train_df = pd.read_csv(os.path.join(self.config["data_dir"], "train.csv"))
        self.test_df = pd.read_csv(os.path.join(self.config["data_dir"], "test.csv"))

        pass

    def getTrain(self):
        return self.train_df

    def getTest(self):
        return self.test_df
