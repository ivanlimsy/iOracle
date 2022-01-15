import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from iOracle.data import _get_dataframe
from sklearn.preprocessing import StandardScaler


class preproc:
    
    def __init__(self, df, start="2017-01-01", end="2022-01-01"):
        self.df = df
        self.start = start
        self.end = end
    
        
    def add_features(self):
        
        self.feature_names = []
        # add 14,50,200 day (MA, RSI, BB Width), volume, VIX and append names to feature names
        for n in [14, 50, 200]:
            self.df[str(n) + 'd' + ' ma'] = SMAIndicator(self.df['Adj Close'], window=n).sma_indicator()
            self.df[str(n) + 'd' + ' rsi'] = RSIIndicator(self.df['Adj Close'], window=n).rsi()
            self.df[str(n) + 'd' + ' bbwidth'] = BollingerBands(self.df['Adj Close'], window=n).bollinger_wband()
            feature_names = feature_names + [str(n) + 'd' + ' ma' , str(n) + 'd' + ' rsi', str(n) + 'd' + ' bbwidth']
        
        # add VIX column and to feature_names
        self.df["VIX"] = _get_dataframe("^VIX", start=self.start, end=self.end)["Adj Close"]
        self.feature_names.append("VIX")
        
        # add Volume to feature_names
        self.feature_names.append("Volume")
        
        # add Adj Close to feature_names
        self.feature_names.append("Adj Close")
        
        return self.df, self.feature_names
        
    
    def add_target(self):
        self.df['5d_future_close'] = self.df['Adj Close'].shift(-5)
        return self.df
    
    
    def scale_features(self):
        #Instantiate scaler
        scaler = StandardScaler()
        #fit and transform features
        self.df[self.feature_names] = scaler.fit_transform(self.df[self.feature_names])
        #Show scaled features
        return self.df   
    
    

