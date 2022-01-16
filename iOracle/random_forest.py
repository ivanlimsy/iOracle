import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import ta
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, ParameterGrid, train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime as dt
import joblib
from iOracle.data import _get_dataframe
from iOracle.TimedSplit import train_val_split
from iOracle.preproc import preproc
from sklearn.pipeline import Pipeline




def train_test_sets(scaled_df, features):
    total_train = scaled_df.loc[(scaled_df.index >= "2018-01-01") & (scaled_df.index <= "2020-12-31")]
    X_train = total_train[features]
    y_train = total_train['5d_future_close']
    total_test = scaled_df.loc[(scaled_df.index >= "2021-01-01") & (scaled_df.index <= "2021-12-23")]
    X_test = total_test[features]
    y_test = total_test['5d_future_close']
    return X_train, X_test, y_train, y_test


def split(X_train, end = '2020-12-31'):
        split = train_val_split(X_train, end)
        splits = split.split_by_index()
        return splits

class trainer:
    def __init__(self, X, y, splits):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.y = y
        self.splits = splits

    def rf_random_grid_search(self):
        
        # Trees 
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum samples to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples at each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        # Instantiate model
        rf = RandomForestRegressor()
        # Random search of parameters with 5 folds
        self.model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter = 100, cv = self.splits, verbose=2, random_state=42, n_jobs = -1)
    
    def run(self):
        self.model.fit(self.X, self.y)
        
    def evaluate(self, X_test, y_test):
        """evaluates y_test and return the MAE"""
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        return round(mae, 2)



if __name__ == "__main__":
    # Get ticker
    raw_df = _get_dataframe("AAPL", start="2017-01-01", end="2022-01-01")
    
    # Add all features
    pp = preproc(raw_df)
    pp_df, feature_names = pp.rf_main()
    
    # Custom train_test_split
    X_train, X_test, y_train, y_test = train_test_sets(pp_df, feature_names)
    
    # Custom nested split for time series CV
    splits = split(X_train, end='2020-12-31')
    print(splits)
    
    # # Instantiate trainer class
    # trainer = trainer(X_train, y_train, splits)
    
    # # Fit model
    # trainer.run()
    
    # # MAE
    # mae = trainer.evaluate(X_test, y_test)
    # print(mae)
   
    








