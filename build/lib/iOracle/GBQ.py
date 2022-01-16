# imports
from google.cloud import bigquery as bq
from datetime import datetime as dt
from time import sleep

from data import save_gbq

# parameters
project_id = 'ioracle'
ticker_name = 'aapl'
train_start = '2018-01-01'
# model parameters_dict??
NON_SEASONAL_ORDER, INCLUDE_DRIFT = (0,1,5), False

def predict(tgt_date, ticker_name=ticker_name):
    '''
    
    '''
    #Setting parameters
    table_name = f'{ticker_name}_data'
    model_name= f'{ticker_name}_arima_predict'

    # upload latest data from yfinance 
    save_gbq(ticker_name, table_name)

    # initialize client
    client = bq.Client(project=project_id)

    #delete any previous models of same name
    query1 = f"DROP MODEL IF EXISTS `{project_id}.main.{model_name}`"
    client.query(query1)

    # train model
    query2 = f"""
            CREATE MODEL IF NOT EXISTS `{project_id}.main.{model_name}`
                OPTIONS(MODEL_TYPE='ARIMA_PLUS',
                         time_series_timestamp_col='Date',
                         time_series_data_col='Adj_Close',
                         DATA_FREQUENCY = 'DAILY',
                         HOLIDAY_REGION = 'GLOBAL',
                         CLEAN_SPIKES_AND_DIPS = FALSE,
                         AUTO_ARIMA = FALSE,
                         NON_SEASONAL_ORDER = {NON_SEASONAL_ORDER},
                         INCLUDE_DRIFT = {INCLUDE_DRIFT}) AS
                SELECT Date, Adj_Close
                FROM `{project_id}.main.{table_name}`
                WHERE Date >= '{train_start}' 
                ORDER BY Date ASC
            """
    client.query(query2)

    #get predicted value
    query3 = f"""SELECT * FROM ML.FORECAST(MODEL `{project_id}.main.{model_name}`,
                STRUCT(50 AS horizon)
                )"""
    while True:
        try:
            df = client.query(query3).to_dataframe()
            df['Date'] = df['forecast_timestamp'].apply(lambda x: x.date())
            df = df.set_index('Date')
            pred = df.loc[dt.strptime(tgt_date, '%Y-%m-%d').date(), 'forecast_value']
            return pred
        except:
            sleep(1)

    
# main
if __name__ == "__main__":
    print(predict('2022-02-02'))
    
