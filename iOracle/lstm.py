import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Normalization, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

class lstm:
    def __init__(self, data):
        """data is a dictionary from feat_eng module
        data dictionary contains: X_train and X_test, y_train and y_test
        shapes should be: sequences:observations:features
        default observations: 30, default features: 14
        data should be scaled already"""
        self.data=data
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        if 'X_val' in data:
            self.X_val = data['X_val']
            self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.features_num = self.X_train.shape[2]
        self.observations_num = self.X_train.shape[1]
        self.sequences_num = self.X_train.shape[0]

    def initialize(self):
        reg_l1 = regularizers.L1(0.01)
        self.model = Sequential()
        self.model.add(LSTM(20, activation='tanh', return_sequences=True, kernel_regularizer=reg_l1))
        self.model.add(LSTM(20, activation='tanh'))
        self.model.add(Dense(15,activation="linear"))
        self.model.add(Dense(1,activation="linear"))
        return self.model

    def compile(self):
        self.model.compile(loss='mse', optimizer='rmsprop', metrics='mae')
        return self.model

    def fit(self):
        self.es = EarlyStopping(patience=5)
        if 'X_val' in self.data:
            self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=2, verbose=1, callbacks=self.es,
                        validation_data=(self.X_val, self.y_val), epochs= 20)
        else:
            self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=2, verbose=1, epochs= 20)
        return self.model, self.history

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test)
