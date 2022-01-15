import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import ta
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, ParameterGrid, train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn import metrics
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






def rf_random_grid_search():
    
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
    model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter = 100, cv = splits, verbose=2, random_state=42, n_jobs = -1)
    
    return model


    









if __name__ == "__main__":
    # Get ticker
    df = _get_dataframe("AAPL", start="2017-01-01", end="2022-01-01")
    # add all features
    raw_df = preproc(df)
    feat_added_df, feature_names = raw_df.add_features()
    target_added_df = raw_df.add_target()
    
    
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model_locally()
    storage_upload()








