#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:35:54 2018

@author: xavier.qiu
"""

import requests
import json
import os

def load_hyperparameters( ):
    """
    """
    pass

def load_config():
    """
    load config from Kaggle-NLP-Summary config.json
    :return:
    """
    json_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    with open(os.path.join(json_path, "config.json")) as f:
        config = json.load(f)
    print("Config - :")
    print(config)
    f.close()
    return config


def send_msg(msg="...",
             dingding_url="https://oapi.dingtalk.com/robot/send?access_token=774cd9150c43c35e43ec93bc6c91553a5c652417c10fd577bec117ed9f3e3182"
             ):
    '''
    this method is used to send myself a message to remind
    '''
    headers = {"Content-Type": "application/json; charset=utf-8"}

    post_data = {
        "msgtype": "text",
        "text": {
            "content": msg
        }
    }

    requests.post(dingding_url, headers=headers,
                  data=json.dumps(post_data))


def map_list_combination(params_list):
    """


    params = {
        "objective": ["regression"],
        "metric": ["rmse"],
        "num_leaves": [10,30,50],
        "min_child_weight": [40,50,60],
        "learning_rate": [0.01,0.03, 0.05, 0.06],
        "bagging_fraction": [0.6,0.7,0.8],
        "feature_fraction": [0.6,0.7,0.8],
        "bagging_frequency": [4,5,6],
        "bagging_seed": [2018],
        "verbosity": [-1]
    }
    :param map_list:
    :return:

        for this example, it will return all the combinations

    """

    res = deque([{}])
    for key in params_list:
        value_list = params_list[key]
        l = len(res)
        for i in range(l):
            cur_dict = res.popleft()
            for value in value_list:
                new_cur_dict = copy.deepcopy(cur_dict)
                new_cur_dict[key] = value
                res.insert(-1, (dict)(new_cur_dict))

    return res


# %%
test = False
if test:

    params_list = {
        "objective": ["regression"],
        "metric": ["rmse"],
        "num_leaves": [10, 30, 50],
        "min_child_weight": [40, 50, 60],
        "learning_rate": [0.01, 0.03, 0.05, 0.06],
        "bagging_fraction": [0.6, 0.7, 0.8],
        "feature_fraction": [0.6, 0.7, 0.8],
        "bagging_frequency": [4, 5, 6],
        "bagging_seed": [2018],
        "verbosity": [-1]
    }

    res = deque([{}])
    for key in params_list:
        value_list = params_list[key]
        l = len(res)
        for i in range(l):
            cur_dict = res.popleft()
            for value in value_list:
                new_cur_dict = copy.deepcopy(cur_dict)
                new_cur_dict[key] = value
                res.insert(-1, (dict)(new_cur_dict))
    print(res)
