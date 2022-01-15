import streamlit as st
import pandas as pd
from datetime import date



st.markdown("""
    # iOracle
    # """)


# sidebar choices
side = st.sidebar

side.markdown('## Choose the stock and the number of days to predict')
stock = side.selectbox("Stock", ("Apple",))
# num_days = side.slider(f"Number of Days from {date.today()}", 1, 14, value=5, step =1)

# get api
api_url = ""

### might need to change 
params = {"Stock": stock}




