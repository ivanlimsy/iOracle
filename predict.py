import os
from google.cloud import storage
import joblib
import pandas as pd
from datetime import date, datetime
from iOracle import data
from iOracle.preproc import preproc
from iOracle.params import BUCKET_NAME, MODEL_FOLDER


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
    df = data.read_local(f"../raw_data/{ticker_name}_train.csv")
    return df

def get_X_test_RF(df, train_df, start, end):
    pp = preproc(df, start=start, end=end, train_df=train_df)
    return pp.get_RF_pred_X()

bucket = BUCKET_NAME
bucket_folder = MODEL_FOLDER
model_name = 'rf_random.joblib'

def download_model(model_name):
    # client = storage.Client().bucket(bucket)
    client = storage.Client.from_service_account_json('service-account-file.json').bucket(bucket)
    storage_location = f'{bucket_folder}/{model_name}'
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    model = joblib.load('model.joblib')
    os.remove('model.joblib')
    return model

def main(ticker_name):
    df = get_df(ticker_name)
    actual_df = df[['Adj Close']].rename(columns={'Adj Close': 'actual'}).iloc[-30:]

    train_df =  get_train_df(ticker_name)
    rf_test = get_X_test_RF(df,train_df, start, end).iloc[-35:]
    rf_model = download_model(model_name)

    # get predictions
    pred = rf_model.predict(rf_test)
    
    # get comparison with actual results
    ind_compare = actual_df.index
    compare_df = pd.DataFrame(pred[:30], index=ind_compare, columns=['compare'])
    compare_df = compare_df.join(actual_df)

    # actual predictions
    ind_pred = [f'Day {n}' for n in range(1, 6)]
    pred_df = pd.DataFrame(pred[-5:], index=ind_pred, columns=['prediction'])

    # final output
    return compare_df, pred_df


if __name__ == '__main__':
    print(main('aapl'))

