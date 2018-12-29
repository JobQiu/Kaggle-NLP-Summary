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
