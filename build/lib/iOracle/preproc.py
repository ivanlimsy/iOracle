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
            self.feature_names = self.feature_names + [str(n) + 'd' + ' ma' , str(n) + 'd' + ' rsi', str(n) + 'd' + ' bbwidth']
        
        # add VIX column and to feature_names
        self.df["VIX"] = _get_dataframe("^VIX", start=self.start, end=self.end)["Adj Close"]
        self.feature_names.append("VIX")
        
        # add Volume to feature_names
        self.feature_names.append("Volume")
        
        # add Adj Close to feature_names
        self.feature_names.append("Adj Close")
        
    
    def add_target(self):
        self.df['5d_future_close'] = self.df['Adj Close'].shift(-5)
    
    
    def scale_features(self):
        #Instantiate scaler
        scaler = StandardScaler()
        #fit and transform features
        self.df[self.feature_names] = scaler.fit_transform(self.df[self.feature_names])
        #Show scaled features

    def get_X(self):
        return self.df[self.feature_names]   
    
    def rf_main(self):
        self.add_features()
        self.add_target()
        self.scale_features()
        return self.df, self.feature_names
        
if __name__ == "__main__":
    df = _get_dataframe(input('insert ticker'), start="2017-01-01", end="2022-01-01")
    pp = preproc(df)
    pp_df, feature_names = pp.rf_main()
    print(pp_df, feature_names)


