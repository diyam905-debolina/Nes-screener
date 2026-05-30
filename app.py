import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="NSE Screener", layout="wide")
st.title("📈 NSE Stock Screener Dashboard")
st.caption("Filters: EMA 50/200 · RSI 55–70 · Volume 1.5x · Week +3% · Price ₹50–₹2000")

STOCKS = [
    "DIXON.NS","KAYNES.NS","ELID.NS","IFCI.NS","ANANTRAJ.NS",
    "JUBLFOOD.NS","TATAELXSI.NS","PERSISTENT.NS","HBLPOWER.NS",
    "GRINDWELL.NS","JYOTICNC.NS","SYRMA.NS","CRAFTSMAN.NS",
    "FINPIPE.NS","LATENTVIEW.NS","BIKAJI.NS","CAMS.NS",
    "EQUITASBNK.NS","APTUS.NS","MEDPLUS.NS"
]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=86400)
def get_screened_stocks():
    results = []
    progress = st.progress(0, text="Scanning NSE stocks...")
    for i, ticker in enumerate(STOCKS):
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty or len(df) < 200:
                continue
            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()
            ema50 = close.ewm(span=50).mean()
            ema200 = close.ewm(span=200).mean()
            rsi = compute_rsi(close)
            avg_vol_20 = volume.rolling(20).mean()
            latest_close = float(close.iloc[-1])
            latest_ema50 = float(ema50.iloc[-1])
            latest_ema200 = float(ema200.iloc[-1])
            latest_rsi = float(rsi.iloc[-1])
            latest_vol = float(volume.iloc[-1])
            latest_avg_vol = float(avg_vol_20.iloc[-1])
            week_perf = ((latest_close - float(close.iloc[-6])) / float(close.iloc[-6])) * 100
            high_52w = float(close.rolling(252).max().iloc[-1])
            dist_52w = ((high_52w - latest_close) / high_52w) * 100
            if (latest_close > latest_ema50 and
                latest_ema50 > latest_ema200 and
                55 <= latest_rsi <= 70 and
                latest_vol >= 1.5 * latest_avg_vol and
                week_perf >= 3 and
                latest_avg_vol >= 500000 and
                50 <= latest_close <= 2000 and
                dist_52w <= 15):
                results.append({
                    "Symbol": ticker.replace(".NS",""),
                    "Price (₹)": round(latest_close, 2),
                    "Week %": round(week_perf, 2),
                    "RSI 14": round(latest_rsi, 1),
                    "Vol Ratio": round(latest_vol / latest_avg_vol, 2),
                    "52W High Dist %": round(dist_52w, 1),
                })
        except:
            pass
        progress.progress((i+1)/len(STOCKS), text=f"Scanning... {ticker}")
    progress.empty()
    return pd.DataFrame(results)

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()

df = get_screened_stocks()

col1, col2, col3 = st.columns(3)
col1.metric("Stocks Matched", len(df))
col2.metric("Avg RSI", round(df["RSI 14"].mean(), 1) if not df.empty else "—")
col3.metric("Avg Week %", f"{round(df['Week %'].mean(), 2)}%" if not df.empty else "—")

if not df.empty:
    st.dataframe(df.sort_values("Week %", ascending=False), use_container_width=True)
else:
    st.warning("No stocks matched today's filters. Try refreshing.")
