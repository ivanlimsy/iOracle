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
api_url = "https://rfpred2-dh3l3t4ama-ew.a.run.app/predict"

### might need to change 
params = {"ticker_name": ticker_dict.get(stock, 0)}

response = requests.get(
    api_url,
    params=params).json()

# compare_df = st.dataframe(response[0])

# pred_df = st.dataframe(response[1])
compare_df = pd.DataFrame(response[0])
compare_df.index = pd.to_datetime(compare_df.index)
compare_df.columns = ['prediction', 'actual']

pred_df = pd.DataFrame(response[1])
# final_df = compare_df.append(pred_df)

st.line_chart(compare_df)
st.line_chart(pred_df)



