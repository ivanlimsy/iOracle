import streamlit as st
import pandas as pd
import requests
import datetime as dt
import ta
from ta.volatility import BollingerBands
import plotly.graph_objs as go

st.title('iOracle')
st.subheader('Predicting the Future Prices of Apple Stock with Machine Learning')
st.markdown('''
            This Data Science project illustrates how an Ensemble Method with LSTM and Random Forest 
            was used to predict the 5-Day future price of Apple's stock based on the past 30 days worth of data.
            ''')

# sidebar choices
side = st.sidebar

side.markdown('## Choose the stock to predict')
stock = side.selectbox("Stock", ("Apple",))

# can add more in future
ticker_dict  = {'Apple': 'aapl'}

# get api
api_url = "https://ioraclev2-dh3l3t4ama-ew.a.run.app/predict"

### might need to change 
params = {"ticker_name": ticker_dict.get(stock)}

response = requests.get(
    api_url,
    params=params).json()

compare_df = pd.DataFrame(response[0])
compare_df.index = pd.to_datetime(compare_df.index)


pred_df = pd.DataFrame(response[1])


# Fix index and join compare_df and pred_df
pred_df.index = [compare_df.index[n]+dt.timedelta(days=7) for n in range(-5, 0)]
final_df = compare_df.append(pred_df)

# Create bollinger bands
hband = BollingerBands(final_df['prediction'], window=5).bollinger_hband()
mband = BollingerBands(final_df['prediction'], window=5).bollinger_mavg()
lband = BollingerBands(final_df['prediction'], window=5).bollinger_lband()

bb_series = pd.DataFrame({'hband':hband, 'mband':mband, 'lband':lband})

final_df = final_df.merge(bb_series, left_index=True, right_index=True)

mode1='lines+markers'
mode2='lines'

fig = go.Figure([
    
    go.Scatter(
        name='Lower Band',
        x=final_df.index,
        y=final_df['lband'],
        mode=mode2,
        line=dict(color='orange', dash='dash'),
        showlegend=True
    ),
    go.Scatter(
        name='Upper Band',
        x=final_df.index,
        y=final_df['hband'],
        mode=mode2,
        line=dict(color='orange', dash='dash'),
        showlegend=True,
        fillcolor='rgba(255, 265, 0, 0.2)',
        fill='tonexty'
    ),
        go.Scatter(
        name='Prediction',
        x=final_df.index,
        y=final_df['prediction'],
        mode=mode1,
        marker=dict(color='red'),
        showlegend=True
    ),
    go.Scatter(
        name='Actual',
        x=final_df.index,
        y=final_df['actual'],
        mode=mode1,
        marker=dict(color='blue'),
        showlegend=True
    )

])

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    title='Model Predictions: Actual vs Predicted'
)

st.plotly_chart(fig)

st.metric("MAE in dollars", f"{round(response[2],2)}")
