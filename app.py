import streamlit as st
import pandas as pd
from datetime import date
import requests



st.markdown("""
    # iOracle
    # """)


# sidebar choices
side = st.sidebar

side.markdown('## Choose the stock and the number of days to predict')
stock = side.selectbox("Stock", ("Apple",))
# num_days = side.slider(f"Number of Days from {date.today()}", 1, 14, value=5, step =1)

# can add more in future
ticker_dict  = {'Apple': 'aapl'}

# get api
api_url = "https://rfpred-dh3l3t4ama-ew.a.run.app/predict"

### might need to change 
params = {"ticker_name": ticker_dict.get(stock, 0)}

response = requests.get(
    api_url,
    params=params).json()

st.write(response['prediction'])
