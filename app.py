from platform import platform
import streamlit as st
import pandas as pd
from datetime import date
import requests
import datetime as dt
import ta
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt


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

# st.line_chart(compare_df)
# st.line_chart(pred_df)

# Plot actual vs prediction
plt.plot(compare_df['prediction'], label='prediction')
plt.plot(compare_df['actual'], label='actual')
plt.legend()
fig = plt.gcf()

st.pyplot(fig)

# Fix index and join compare_df and pred_df
pred_df.index = [compare_df.index[n]+dt.timedelta(days=7) for n in range(-5, 0)]
final_df = compare_df.append(pred_df)

# Create bollinger bands
hband = BollingerBands(final_df['prediction'], window=5).bollinger_hband()
mband = BollingerBands(final_df['prediction'], window=5).bollinger_mavg()
lband = BollingerBands(final_df['prediction'], window=5).bollinger_lband()

bb_series = pd.DataFrame({'hband':hband, 'mband':mband, 'lband':lband})

final_df = final_df.merge(bb_series, left_index=True, right_index=True)

# Plot graph
plt.plot(final_df['hband'], label='hband')
plt.plot(final_df['lband'], label='lband')
plt.plot(final_df['actual'], label='actual')
plt.plot(final_df['prediction'], label='prediction')
plt.fill_between(final_df.index, final_df['hband'], final_df['lband'], alpha=0.2)
plt.legend()