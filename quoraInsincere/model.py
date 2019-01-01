#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:52:11 2018

@author: xavier.qiu
"""

from keras.models import Model
from keras.layers import Dense, Embedding, Bidirectional, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, \
    concatenate, Input, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


def tweak_threshold(pred, truth):
    thresholds = []
    scores = []
    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        thresholds.append(thresh)
        score = f1_score(truth, (pred > thresh).astype(int))
        scores.append(score)
    return np.max(scores), thresholds[np.argmax(scores)]


def f1(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class BaseModel:

    pass


class CNNModel:

    def __init__(self,
                 data_set,
                 embed_size=300,
                 max_ques_len=60,
                 loss="binary_crossentropy",
                 embedding_trainable=True,
                 test_ratio=0.1,
                 ):
        self.embed_size = embed_size
        self.max_ques_len = max_ques_len
        self.embedding_trainable = embedding_trainable
        self.loss_type = loss
        self.data_set = data_set
        self.embedding_matrix = data_set.embedding_matrix
        self.test_ratio = test_ratio
        self.history = None

        # build model and ...
        self.model = self.build_model()

    def build_model(self):
        inp = Input(shape=(self.max_ques_len,))
        x = Embedding(self.voc_len, self.embed_size, weights=[self.embedding_matrix], trainable=True)(inp)
        x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        avg_pl = GlobalAveragePooling1D()(x)
        max_pl = GlobalMaxPooling1D()(x)
        concat = concatenate([avg_pl, max_pl])
        dense = Dense(64, activation="relu")(concat)
        drop = Dropout(0.1)(dense)
        output = Dense(1, activation="sigmoid")(drop)

        model = Model(inputs=inp, outputs=output)
        model.compile(loss=self.loss_type, optimizer=Adam(lr=0.0001), metrics=['accuracy', f1])

        return model

    def train(self, epoch, batch_size):
        X = self.data_set.x_train
        y = self.data_set.y_train

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_ratio, random_state=420)

        checkpoints = ModelCheckpoint('weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)
        self.history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
                                      validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])

        pred_val = self.model.predict(X_val, batch_size=512, verbose=1)
        score_val, threshold_val = tweak_threshold(pred_val, y_val)

        print(f"Scored {round(score_val, 4)} for threshold {threshold_val} with untreated texts on validation data")
        pass

    def predict(self):
        self.pred_output = self.model.predict(self.data_set.x_test, batch_size=512, verbose=1)
