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

def get_X_test_RF(df, start, end):
    pp = preproc(df, start, end)
    pp.add_features()
    pp.scale_features()
    return pp.get_X()

bucket = BUCKET_NAME
bucket_folder = MODEL_FOLDER
model_name = 'rf_random.joblib'

def download_model(model_name):
    client = storage.Client().bucket(bucket)
    storage_location = f'{bucket_folder}/{model_name}'
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    model = joblib.load('model.joblib')
    os.remove('model.joblib')
    return model

def main(ticker_name):
    df = get_df(ticker_name)
    rf_test = get_X_test_RF(df, start, end).iloc[-35:]
    rf_model = download_model(model_name)

    #get indices
    # get indices
    index = df.index[-30:]
    index = list(index)
    index = [x.date() for x in index]
    index.extend([f'Day{n}' for n in range(1, 6)])

    # get predictions
    pred = rf_model.predict(rf_test)
    pred_df = pd.DataFrame(pred, index=index, columns=['pred'])

    return pred_df

if __name__ == '__main__':
    print(main('aapl'))
