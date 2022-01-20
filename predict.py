import os
from google.cloud import storage
import joblib
import pandas as pd
from datetime import date, datetime, timedelta
from iOracle import data
from iOracle.preproc import preproc
from iOracle.params import BUCKET_NAME, MODEL_FOLDER
from sklearn.metrics import mean_absolute_error


def subtract_years(dt, years):
    try:
        dt = dt.replace(year=dt.year-years)
    except ValueError:
        dt = dt.replace(year=dt.year-years, day=dt.day-1)
    return datetime.strftime(dt, '%Y-%m-%d')

end = date.today()
start = subtract_years(end, 2)

def get_df(ticker_name):
    df = data._get_dataframe(ticker_name, start, end)
    return df

def get_train_df(ticker_name):
    df = data.read_local(f"raw_data/{ticker_name}_train.csv")
    return df

def get_X_pred_RF(df, train_df, start, end):
    pp = preproc(df, start=start, end=end, train_df=train_df)
    return pp.get_RF_pred_X()

def get_X_pred_lstm(df, train_df, start, end):
    pp = preproc(df, start=start, end=end, train_df=train_df)
    return pp.get_lstm_pred_X()

bucket = BUCKET_NAME
bucket_folder = MODEL_FOLDER
rf_model_name = 'rf_random.joblib'
lstm_model_name = 'lstm.joblib'

def download_model(model_name):
    # client = storage.Client().bucket(bucket)
    client = storage.Client.from_service_account_json('service-account-file.json').bucket(bucket)
    storage_location = f'{bucket_folder}/{model_name}'
    blob = client.blob(storage_location)
    blob.download_to_filename(f'{model_name}.joblib')
    model = joblib.load(f'{model_name}.joblib')
    os.remove(f'{model_name}.joblib')
    return model

def main(ticker_name):
    df = get_df(ticker_name)
    actual_df = df[['Adj Close']].rename(columns={'Adj Close': 'actual'}).iloc[-30:]

    # get X
    train_df =  get_train_df(ticker_name)
    df1 = df.copy()
    rf_X_pred = get_X_pred_RF(df1,train_df, start, end).iloc[-35:]
    df2=df.copy()
    lstm_X_pred = get_X_pred_lstm(df2,train_df, start, end)[-35:]
    
    #download models
    rf_model = download_model(rf_model_name)
    lstm_model = download_model(lstm_model_name)

    # # get predictions
    rf_pred = rf_model.predict(rf_X_pred)
    lstm_pred = lstm_model.predict(lstm_X_pred).flatten()

    # # prep for ensemble
    ensemble_X_pred = pd.DataFrame({'RF_pred':rf_pred[:30], 'LSTM_pred':lstm_pred[:30]})
    ensemble_model_name = 'rf_lstm_linear.joblib'
    ensemble_model = download_model(ensemble_model_name)
    ensemble_pred = ensemble_model.predict(ensemble_X_pred)

    
    # get comparison with actual results
    compare_df = pd.DataFrame({'prediction': ensemble_pred[:30]}, index = actual_df.index).join(actual_df)
    mae = mean_absolute_error(compare_df['actual'], compare_df['prediction'])
    
    pred_df = pd.DataFrame({'prediction': ensemble_pred[-5:]})
    

    # final output
    return compare_df, pred_df, mae



if __name__ == '__main__':
    print(main('aapl'))


