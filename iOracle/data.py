# Imports
import yfinance as yf
import pandas as pd
from datetime import datetime as dt

# parameters
project_id = "ioracle"


# Functions
def _get_dataframe(ticker_name, start, end):
    """
    get_dataframe(ticker_name, start, end)
    Downloads OHLC,adj close and volume of ticker_name from yahoo finance 
    from start till end (not inclusive of end)
    returns dataframe

    """
    df = yf.download(ticker_name, start=start, end=end)
    return df


def _get_start_end(kwargs):
    '''
    get_start_end(kwargs)
    from kwargs, get start, end dates
    if not stated, will return default values
    return start, end dates
    '''
    start = kwargs.get('start', "2017-01-01")
    end = kwargs.get('end', dt.today().strftime('%Y-%m-%d')) #not inclusive
    return start, end
         

def save_local(ticker_name, path_filename, **kwargs):
    """
    save_local(path_filename, ticker_name, **kwargs)
    uses _get_dataframe to download df from yfinance
    save df to local path
    """
    start, end = _get_start_end(kwargs)
      
    df = _get_dataframe(ticker_name, start=start, end=end)
    if len(df) != 0:
        df.to_csv(path_filename)
        print(f"{ticker_name} from {start} to {end} saved to {path_filename}")
        

def save_gbq(ticker_name, table_name, **kwargs):
    """
    save_to_gbq(ticker_name, table_name)
    convert df to uploadable format for gbq
    uses _get_dataframe to download df from yfinance
    uploads as table_name in gbq
    """
    start, end = _get_start_end(kwargs)
             
    temp_df = _get_dataframe(ticker_name, start=start, end=end)
    
    if len(temp_df) != 0: # check that df is not empty
        temp_df = temp_df.rename(columns={'Adj Close': 'Adj_Close'}).reset_index()
        temp_df.to_gbq(f'{project_id}.main.{table_name}', 
                        project_id=project_id, 
                        table_schema = [{'name': 'Date','type':'DATE'}], #hard code schema for date from DATETIME to DATE
                        if_exists='replace'
                    )
        print(f"{ticker_name} from {start} to {end} saved to {project_id}.main.{table_name}")
        

def read_local(path_filename):
    """
    read_local(path_filename)
    reads the csv file and parses date col as date, setting the date as the index
    returns the df
    """
    df = pd.read_csv(path_filename)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')
    

# read from gbq (undo changes)
def read_gbq(table_name, **kwargs):
    """
    read_gbq(table_name)
    sends a SQL query to gbq and downloads table table_name as df
    sorts the date column and sets it as the index
    returns the df
    """

    sql = f"SELECT * FROM `{project_id}.main.{table_name}` "

    df = pd.read_gbq(sql, project_id=project_id)
    df = df.sort_values('Date').set_index('Date')   
    return df

def data_main(ticker_name, path_filename, table_name, **kwargs):
    import os
    from google.cloud import bigquery as bq 

    print("Test download to local")
    save_local(ticker_name, path_filename, **kwargs)

    print("Test read from local")
    print(read_local(path_filename))

    print(f"Deleting local file {path_filename}")
    os.remove(path_filename)
    print(f"Deleted local file {path_filename}")

    print("Test upload to gbq")
    save_gbq(ticker_name, table_name, **kwargs)

    print("Test download from gbq")
    print(read_gbq(table_name, **kwargs))

    print(f"Deleting {table_name} from gbq")
    client = bq.Client(project=project_id)
    client.delete_table(f"{project_id}.main.{table_name}")
    print(f"Deleted {table_name} from gbq")


# main
if __name__ == "__main__":
    data_main('aapl', 'test.csv', 'test_data')