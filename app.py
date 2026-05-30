import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="NSE Screener", layout="wide")
st.title("📈 NSE Stock Screener Dashboard")
st.caption("Filters: EMA 50/200 · RSI 55–70 · Volume 1.5x · MACD · ADX · Week +1% · Price ₹50–₹5000")

STOCKS = [
    "DIXON.NS","KAYNES.NS","ELID.NS","IFCI.NS","ANANTRAJ.NS",
    "JUBLFOOD.NS","TATAELXSI.NS","PERSISTENT.NS","HBLPOWER.NS",
    "GRINDWELL.NS","JYOTICNC.NS","SYRMA.NS","CRAFTSMAN.NS",
    "FINPIPE.NS","LATENTVIEW.NS","BIKAJI.NS","CAMS.NS",
    "EQUITASBNK.NS","APTUS.NS","MEDPLUS.NS","CDSL.NS",
    "ANGELONE.NS","NAZARA.NS","IRCTC.NS","RVNL.NS",
    "IRFC.NS","HUDCO.NS","NBCC.NS","BEL.NS","HAL.NS",
    "COCHINSHIP.NS","MAZAGON.NS","GRSE.NS","INOXWIND.NS",
    "SUZLON.NS","VOLTAMP.NS","FINEORG.NS","GALAXYSURF.NS",
    "NOCIL.NS","AARTIDRUGS.NS","APLLTD.NS","ERIS.NS",
    "GLAND.NS","JBCHEPHARM.NS","RAINBOW.NS","KIMS.NS",
    "CAMPUS.NS","VEDANT.NS","TITAN.NS","KALYANKJIL.NS",
    "ABSLAMC.NS","HDFCAMC.NS","NUVAMA.NS","360ONE.NS",
    "FIVESTAR.NS","CREDITACC.NS","UJJIVANSFB.NS","METROPOLIS.NS",
    "INDIGOPNTS.NS","ASIANPAINT.NS","BERGER.NS","KANSAINER.NS",
    "APLAPOLLO.NS","RATNAMANI.NS","WELSPUNLIV.NS","POLYMED.NS",
    "SUPREMEIND.NS","ASTRAL.NS","FINOLEX.NS","PRINCEPIPE.NS",
    "HAVELLS.NS","POLYCAB.NS","KEI.NS","RRKABEL.NS",
    "APOLLOTYRE.NS","BALKRISIND.NS","TIINDIA.NS","CEAT.NS",
    "MOTHERSON.NS","SUNDRMFAST.NS","ENDURANCE.NS","SUPRAJIT.NS",
    "LALPATHLAB.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
    "BIOCON.NS","ALKEM.NS","IPCALAB.NS","NATCOPHARM.NS",
    "TATACOMM.NS","TANLA.NS","ROUTE.NS","MSTCLTD.NS",
    "HAPPSTMNDS.NS","CYIENT.NS","KPITTECH.NS","MASTEK.NS",
    "SONACOMS.NS","OLECTRA.NS","GPIL.NS","JINDALSTEL.NS",
    "ABIRLANUVO.NS","ATUL.NS","BASF.NS","DEEPAKNTR.NS"
]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    return macd_line, signal_line

def compute_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

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
            high = df["High"].squeeze()
            low = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            ema50 = close.ewm(span=50).mean()
            ema200 = close.ewm(span=200).mean()
            rsi = compute_rsi(close)
            macd_line, signal_line = compute_macd(close)
            adx = compute_adx(high, low, close)
            avg_vol_20 = volume.rolling(20).mean()

            latest_close = float(close.iloc[-1])
            latest_ema50 = float(ema50.iloc[-1])
            latest_ema200 = float(ema200.iloc[-1])
            latest_rsi = float(rsi.iloc[-1])
            latest_macd = float(macd_line.iloc[-1])
            latest_signal = float(signal_line.iloc[-1])
            latest_adx = float(adx.iloc[-1])
            latest_vol = float(volume.iloc[-1])
            latest_avg_vol = float(avg_vol_20.iloc[-1])
            week_perf = ((latest_close - float(close.iloc[-6])) / float(close.iloc[-6])) * 100
            high_52w = float(close.rolling(252).max().iloc[-1])
            dist_52w = ((high_52w - latest_close) / high_52w) * 100

            if (latest_close > latest_ema50 and
                latest_ema50 > latest_ema200 and
                50 <= latest_rsi <= 75 and
                latest_macd > latest_signal and
                latest_adx > 20 and
                latest_vol >= 1.0 * latest_avg_vol and
                week_perf >= 1 and
                latest_avg_vol >= 200000 and
                50 <= latest_close <= 5000 and
                dist_52w <= 20):

                if latest_rsi > 70:
                    signal = "⚠️ Near Exit"
                elif latest_macd > latest_signal and latest_adx > 25:
                    signal = "✅ Strong Buy"
                else:
                    signal = "👀 Watch"

                results.append({
                    "Symbol": ticker.replace(".NS", ""),
                    "Price (₹)": round(latest_close, 2),
                    "Signal": signal,
                    "Week %": round(week_perf, 2),
                    "RSI 14": round(latest_rsi, 1),
                    "MACD": "Above Signal ✅" if latest_macd > latest_signal else "Below Signal ❌",
                    "ADX": round(latest_adx, 1),
                    "Vol Ratio": round(latest_vol / latest_avg_vol, 2),
                    "52W High Dist %": round(dist_52w, 1),
                })
        except:
            pass
        progress.progress((i + 1) / len(STOCKS), text=f"Scanning... {ticker}")
    progress.empty()
    return pd.DataFrame(results)

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

df = get_screened_stocks()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks Matched", len(df))
col2.metric("Strong Buy", len(df[df["Signal"] == "✅ Strong Buy"]) if not df.empty else 0)
col3.metric("Avg RSI", round(df["RSI 14"].mean(), 1) if not df.empty else "—")
col4.metric("Avg Week %", f"{round(df['Week %'].mean(), 2)}%" if not df.empty else "—")

if not df.empty:
    st.subheader("Matched Stocks")
    st.dataframe(df.sort_values("Signal", ascending=True), use_container_width=True)

    st.subheader("📖 How to read this table")
    st.info("""
    ✅ Strong Buy → MACD up + ADX strong + RSI healthy. Good to study for entry.
    👀 Watch → Filters passed but trend not fully strong yet. Keep on radar.
    ⚠️ Near Exit → RSI getting high (above 70). If you hold this stock, consider booking profit.
    """)
else:
    st.warning("No stocks matched today's filters. Try refreshing.")
