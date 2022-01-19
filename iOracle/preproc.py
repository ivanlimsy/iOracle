import yfinance as yf
import pandas as pd
import numpy as np
import ta
import datetime as dt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands
from iOracle.data import _get_dataframe
from sklearn.preprocessing import StandardScaler


class preproc:

    def __init__(self, df, start="2017-01-01", end="2022-01-01", train_df=''):
        self.df = df
        self.start = start
        self.end = end
        if not(isinstance(train_df, str)):
            self.train_df = train_df


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

    def scale_features_pred(self):
        #Instantiate scaler
        pp = preproc(self.train_df)
        pp.add_features()
        train_df = pp.df
        scaler = StandardScaler()
        #fit to original training data
        scaler.fit(train_df[self.feature_names])
        #transform
        self.df[self.feature_names] = scaler.transform(self.df[self.feature_names])
        #Show scaled features

    def get_X(self):
        return self.df[self.feature_names]

    def rf_main(self):
        self.add_features()
        self.add_target()
        self.scale_features()
        return self.df, self.feature_names

    def get_lstm_trainval_data(self):
        params = {'train_start':dt.datetime(2018,1,1,0,0,0),
                  'test_start':dt.datetime(2021,1,1,0,0,0),
                  'val_start':dt.datetime(2020,4,1,0,0,0),
                  'hist_days':30}
        self.df.drop(labels=['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
        self.df.dropna(axis=0, inplace=True)
        self.X = self.df.drop(labels='5d_future_close', axis=1).copy()
        self.y = self.df[['5d_future_close']].copy()
        #Setting X_test and y_test
        #trainval_length = int(len(df)*(1-params['test_size']))
        self.X_trainval_df = self.X[self.X.index<params['test_start']]
        self.X_trainval_df = self.X_trainval_df[self.X_trainval_df.index >= params['train_start']]
        self.y_trainval_df = self.y[self.y.index<params['test_start']]
        #X_trainval_df = X.iloc[0:trainval_length,:]
        #y_trainval_df = y.iloc[0:trainval_length,:]
        self.X_test_df = self.X[self.X.index>=params['test_start']]
        self.y_test_df = self.y[self.y.index>=params['test_start']]
        #X_test_df = X.iloc[trainval_length:-1,:]
        #y_test_df = y.iloc[trainval_length:-1,:]

        #Setting X_train/X_val, and y_train/y_val
        #train_length = int(len(X_trainval_df)*(1-params['val_size']))
        self.X_train_df = self.X_trainval_df[self.X_trainval_df.index<params['val_start']]
        self.y_train_df = self.y_trainval_df[self.y_trainval_df.index<params['val_start']]
        #X_train_df = X_trainval_df.iloc[0:train_length,:]
        #y_train_df = y_trainval_df.iloc[0:train_length,:]
        self.X_val_df = self.X_trainval_df[self.X_trainval_df.index>=params['val_start']]
        self.y_val_df = self.y_trainval_df[self.y_trainval_df.index>=params['val_start']]
        #X_val_df = X_trainval_df.iloc[train_length:-1,:]
        #y_val_df = y_trainval_df.iloc[train_length:-1,:]

        #Scaling X features
        # scaler = StandardScaler()   #already scaled in scale_features method
        # X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df))
        # scaler = StandardScaler()
        # X_val_df = pd.DataFrame(scaler.fit_transform(X_val_df))
        # scaler = StandardScaler()
        # X_test_df = pd.DataFrame(scaler.fit_transform(X_test_df))

        #E.g.100 observations, index 0 to 99
        #Historical days: 30 including the day itself
        #Future days: 5
        #Total days: 1-100 >> 0-99 >> iloc/range(0:100)
        #1st i: 30 >> iloc/range(29)
        #1st historical days: 1-30 >> 0:29 >> iloc/range(0:30)
        #1st future days: 31:35 >> 30:34 >> iloc/range(30,35)
        #Last i: 95 >> iloc/range(94)
        #Last historical days: 66:95 >> 65:94 >> iloc/range(65:95)
        #Last future days: 96:100 >> 95:99 >> iloc/range(95:100)

        #Preparing sequences of features for X_train and targets for y_train
        self.X_train = []
        self.y_train = []
        #(30-1):(100-5-1) >> 29:94
        for i in range((params['hist_days']-1),(len(self.X_train_df)-1)):
            #(29-30+1):(29+1) >> 0:30
            self.X_train.append(np.array(self.X_train_df.iloc[(i-params['hist_days']+1):i+1,:]))
            #(29+5):0 >> 34:0
            self.y_train.append(np.array(self.y_train_df.iloc[i,0]))

        #Preparing sequences of features for X_val and targets for y_val
        self.X_val = []
        self.y_val = []
        for i in range((params['hist_days']-1),(len(self.X_val_df)-1)):
            self.X_val.append(np.array(self.X_val_df.iloc[(i-params['hist_days']+1):i+1,:]))
            self.y_val.append(np.array(self.y_val_df.iloc[i,0]))

        #Preparing sequences of features for X_test and targets for y_test
        self.X_test = []
        self.y_test = []
        for i in range((params['hist_days']-1),(len(self.X_test_df)-1)):
            self.X_test.append(np.array(self.X_test_df.iloc[(i-params['hist_days']+1):i+1,:]))
            self.y_test.append(np.array(self.y_test_df.iloc[i, 0]))

        self.X_train = np.array(self.X_train); self.y_train = np.array(self.y_train)
        self.y_train = np.reshape(np.array([self.y_train]),(self.y_train.shape[0],1))
        self.X_val = np.array(self.X_val); self.y_val = np.array(self.y_val)
        self.y_val = np.reshape(np.array([self.y_val]),(self.y_val.shape[0],1))
        self.X_test = np.array(self.X_test); self.y_test = np.array(self.y_test)
        self.y_test = np.reshape(np.array([self.y_test]),(self.y_test.shape[0],1))

        #Shapes: (n_sequences, n_observations, n_features)
        print(f"Train shapes: X is {self.X_train.shape} and y is {self.y_train.shape}")
        print(f"Val shapes: X is {self.X_val.shape} and y is {self.y_val.shape}")
        print(f"Test shapes: X is {self.X_test.shape} and y is {self.y_test.shape}")
        print(f"Total number of sequences reduced from {len(self.df)} to {self.X_train.shape[0]+self.X_test.shape[0]}")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, self.X_train_df, self.X_val_df, self.X_test_df

    def get_lstm_train_data(self):
        params = {'train_start':dt.datetime(2018,1,1,0,0,0),
                  'test_start':dt.datetime(2021,1,1,0,0,0),
                  'val_start':dt.datetime(2020,4,1,0,0,0),
                  'hist_days':30}
        self.df.drop(labels=['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
        self.df.dropna(axis=0, inplace=True)
        self.X = self.df.drop(labels='5d_future_close', axis=1).copy()
        self.y = self.df[['5d_future_close']].copy()
        self.X_trainval_df = self.X[self.X.index<params['test_start']]
        self.X_trainval_df = self.X_trainval_df[self.X_trainval_df.index >= params['train_start']]
        self.y_trainval_df = self.y[self.y.index<params['test_start']]

        self.X_train_df = self.X_trainval_df[self.X_trainval_df.index<params['val_start']]
        self.y_train_df = self.y_trainval_df[self.y_trainval_df.index<params['val_start']]
        self.X_val_df = self.X_trainval_df[self.X_trainval_df.index>=params['val_start']]
        self.y_val_df = self.y_trainval_df[self.y_trainval_df.index>=params['val_start']]

        #Scaling X features
        # scaler = StandardScaler()
        # X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df))
        # scaler = StandardScaler()
        # X_val_df = pd.DataFrame(scaler.fit_transform(X_val_df))

        #Preparing sequences of features for X_train and targets for y_train
        self.X_train = []
        self.y_train = []
        #(30-1):(100-5-1) >> 29:94
        for i in range((params['hist_days']-1),(len(self.X_train_df)-1)):
            #(29-30+1):(29+1) >> 0:30
            self.X_train.append(np.array(self.X_train_df.iloc[(i-params['hist_days']+1):i+1,:]))
            #(29+5):0 >> 34:0
            self.y_train.append(np.array(self.y_train_df.iloc[i,0]))
        self.y_train_df = self.y_train_df.iloc[(params['hist_days']-1):-1,:]

        #Preparing sequences of features for X_val and targets for y_val
        self.X_val = []
        self.y_val = []
        for i in range((params['hist_days']-1),(len(self.X_val_df)-1)):
            self.X_val.append(np.array(self.X_val_df.iloc[(i-params['hist_days']+1):i+1,:]))
            self.y_val.append(np.array(self.y_val_df.iloc[i,0]))
        self.y_val_df = self.y_val_df.iloc[(params['hist_days']-1):-1,:]

        self.X_train = np.array(self.X_train); self.y_train = np.array(self.y_train)
        self.y_train = np.reshape(np.array([self.y_train]),(self.y_train.shape[0],1))
        self.X_val = np.array(self.X_val); y_val = np.array(self.y_val)
        self.y_val = np.reshape(np.array([self.y_val]),(np.array(self.y_val).shape[0],1))

        return self.X_train, self.X_val, self.y_train, self.y_val, self.X_train_df, self.X_val_df, self.y_train_df, self.y_val_df


    def lstm_main(self, op='trainval'):
        self.add_features()
        return self.df
        self.add_target()
        self.scale_features()
        if op == 'trainval':
            self.get_lstm_trainval_data()
            returned_data = {'X_train':self.X_train, 'X_val':self.X_val, 'X_test':self.X_test,
                             'y_train':self.y_train, 'y_val':self.y_val, 'y_test':self.y_test,
                             'X_train_df':self.X_train_df, 'X_val_df':self.X_val_df, 'X_test_df':self.X_test_df,}
        elif op == 'train':
            self.get_lstm_train_data()
            returned_data = {'X_train':self.X_train, 'X_test':self.X_val,
                             'y_train':self.y_train, 'y_test':self.y_val,
                             'X_train_df':self.X_train_df, 'X_test_df':self.X_val_df,
                             'y_train_df':self.y_train_df, 'y_test_df':self.y_val_df}
        return returned_data

    def get_lstm_pred_X(self):
        # date_format = "%Y-%m-%d"
        # converted_date = dt.datetime.strptime(date, date_format)
        # end = (converted_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        # start = (converted_date - dt.timedelta(days=500)).strftime("%Y-%m-%d")
        # self.df = _get_dataframe('AAPL', start, end)
        # self.add_features()
        # self.scale_features()
        self.add_features()
        self.feature_names = ['Adj Close', 'Volume', '14d bbwidth', '14d rsi', '14d ma',
                              '50d bbwidth', '50d rsi', '50d ma', '200d bbwidth', '200d rsi', '200d ma',
                              'VIX']

        #LSTM model's order of features are as follows
        # ['Adj Close', 'Volume', 'w_bol_14', 'rsi_14', 'ma_14', 'w_bol_50',
        #     'rsi_50', 'ma_50', 'w_bol_200', 'rsi_200', 'ma_200',
        #     'vix_adj_close']
        self.scale_features_pred()
        lstm_pred_X_df = self.get_X()
        # lstm_pred_X_df = self.get_RF_pred_X()      #Commented out to manually reorder features
        lstm_pred_X = []
        for i in range((30 - 1), (len(lstm_pred_X_df) - 1)):
            #(29-30+1):(29+1) >> 0:30
            lstm_pred_X.append(np.array(lstm_pred_X_df.iloc[(i-30+1):i+1,:]))
        lstm_pred_X = np.array(lstm_pred_X)
        return lstm_pred_X[-35:]

    def get_RF_pred_X(self):
        self.add_features()
        self.scale_features_pred()
        return self.get_X()




# if __name__ == "__main__":
#     df = _get_dataframe(input('insert ticker'), start="2017-01-01", end="2022-01-01")
#     pp = preproc(df)
#     pp_df, feature_names = pp.rf_main()
#     print(pp_df, feature_names)
