#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 02:12:36 2018

@author: xavier.qiu

https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

"""
import re


def clean():
    pass


def deal_with_misspell(x):
    """
    """
    pass


def deal_with_contraction():
    pass


def deal_with_special_characters():
    pass


def deal_with_punct(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


def deal_with_numbers(x):
    """
    hmm why is "##" in there? Simply because as a reprocessing all numbers bigger
    than 9 have been replaced by hashs. I.e. 15 becomes ## while 123 becomes ###
    or 15.80€ becomes ##.##€. So lets mimic this pre-processing step to further
    improve our embeddings coverage

    :param x:
    :return:
    """
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


mispell_dict = {'colour': 'color',
                'centre': 'center',
                'didnt': 'did not',
                'doesnt': 'does not',
                'isnt': 'is not',
                'shouldnt': 'should not',
                'favourite': 'favorite',
                'travelling': 'traveling',
                'counselling': 'counseling',
                'theatre': 'theater',
                'cancelled': 'canceled',
                'labour': 'labor',
                'organisation': 'organization',
                'wwii': 'world war 2',
                'citicise': 'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'Snapchat': 'social medium'
                }
