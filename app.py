import streamlit as st
import yfinance as yf
import pandas as pd

st.title("NSE Data Test")

ticker = "RELIANCE.NS"
st.write(f"Fetching data for {ticker}...")

try:
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty:
        st.error("No data returned — yfinance may be blocked or NSE data unavailable")
    else:
        st.success(f"Data fetched successfully! {len(df)} rows found")
        st.write("Latest 5 rows:")
        st.dataframe(df.tail())
        close = df["Close"].squeeze()
        st.write(f"Latest close price: ₹{round(float(close.iloc[-1]), 2)}")
except Exception as e:
    st.error(f"Error: {e}")
