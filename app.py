import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
    "TATACOMM.NS","TANLA.NS","ROUTE.NS","HAPPSTMNDS.NS",
    "CYIENT.NS","KPITTECH.NS","MASTEK.NS","SONACOMS.NS",
    "OLECTRA.NS","GPIL.NS","JINDALSTEL.NS","DEEPAKNTR.NS"
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
    return dx.rolling(period).mean()

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
                week_perf >= -5 and
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
                    "Ticker": ticker,
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

@st.cache_data(ttl=86400)
def get_chart_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    return df

def plot_stock_chart(ticker, symbol):
    df = get_chart_data(ticker)
    if df.empty:
        st.warning("No data available.")
        return

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    open_ = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    ema50 = close.ewm(span=50).mean()
    ema200 = close.ewm(span=200).mean()
    rsi = compute_rsi(close)
    macd_line, signal_line = compute_macd(close)
    adx = compute_adx(high, low, close)

    # Golden cross points
    golden_cross = []
    death_cross = []
    for i in range(1, len(ema50)):
        if ema50.iloc[i] > ema200.iloc[i] and ema50.iloc[i-1] <= ema200.iloc[i-1]:
            golden_cross.append(i)
        elif ema50.iloc[i] < ema200.iloc[i] and ema50.iloc[i-1] >= ema200.iloc[i-1]:
            death_cross.append(i)

    # RSI overbought/oversold points
    rsi_overbought = rsi[rsi > 70]
    rsi_oversold = rsi[rsi < 30]

    # 52W high and low
    high_52w_val = float(close.max())
    low_52w_val = float(close.min())
    high_52w_date = close.idxmax()
    low_52w_date = close.idxmin()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=open_, high=high, low=low, close=close,
        name="Price",
        increasing_line_color="#1D9E75",
        decreasing_line_color="#E24B4A"
    ))

    # EMA lines
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA 50",
        line=dict(color="#378ADD", width=1.5), opacity=0.9))
    fig.add_trace(go.Scatter(x=df.index, y=ema200, name="EMA 200",
        line=dict(color="#EF9F27", width=1.5), opacity=0.9))

    # Golden cross markers
    for idx in golden_cross:
        fig.add_annotation(x=df.index[idx], y=float(close.iloc[idx]),
            text="🟢 Golden Cross", showarrow=True, arrowhead=2,
            arrowcolor="#1D9E75", font=dict(size=10, color="#1D9E75"),
            bgcolor="white", bordercolor="#1D9E75", ay=-40)

    # Death cross markers
    for idx in death_cross:
        fig.add_annotation(x=df.index[idx], y=float(close.iloc[idx]),
            text="🔴 Death Cross", showarrow=True, arrowhead=2,
            arrowcolor="#E24B4A", font=dict(size=10, color="#E24B4A"),
            bgcolor="white", bordercolor="#E24B4A", ay=40)

    # 52W high marker
    fig.add_annotation(x=high_52w_date, y=high_52w_val,
        text=f"52W High ₹{round(high_52w_val,1)}", showarrow=True,
        arrowhead=2, arrowcolor="#534AB7", font=dict(size=10, color="#534AB7"),
        bgcolor="white", bordercolor="#534AB7", ay=-50)

    # 52W low marker
    fig.add_annotation(x=low_52w_date, y=low_52w_val,
        text=f"52W Low ₹{round(low_52w_val,1)}", showarrow=True,
        arrowhead=2, arrowcolor="#E24B4A", font=dict(size=10, color="#E24B4A"),
        bgcolor="white", bordercolor="#E24B4A", ay=50)

    fig.update_layout(
        title=f"{symbol} — 1 Year Price Chart with Key Highlights",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI 14",
        line=dict(color="#534AB7", width=2)))
    fig2.add_hline(y=70, line_dash="dash", line_color="#E24B4A",
        annotation_text="Overbought (70)", annotation_position="right")
    fig2.add_hline(y=30, line_dash="dash", line_color="#1D9E75",
        annotation_text="Oversold (30)", annotation_position="right")
    fig2.add_hline(y=55, line_dash="dot", line_color="#378ADD",
        annotation_text="Target zone start (55)", annotation_position="right")

    # Mark overbought zones
    if not rsi_overbought.empty:
        fig2.add_trace(go.Scatter(
            x=rsi_overbought.index, y=rsi_overbought,
            mode="markers", name="Overbought",
            marker=dict(color="#E24B4A", size=6, symbol="circle")))

    if not rsi_oversold.empty:
        fig2.add_trace(go.Scatter(
            x=rsi_oversold.index, y=rsi_oversold,
            mode="markers", name="Oversold",
            marker=dict(color="#1D9E75", size=6, symbol="circle")))

    fig2.update_layout(
        title=f"{symbol} — RSI 14 (Momentum)",
        height=250,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig2, use_container_width=True)

    # MACD chart
    fig3 = go.Figure()
    macd_hist = macd_line - signal_line
    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in macd_hist]
    fig3.add_trace(go.Bar(x=df.index, y=macd_hist, name="MACD Histogram",
        marker_color=colors, opacity=0.7))
    fig3.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
        line=dict(color="#378ADD", width=1.5)))
    fig3.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal",
        line=dict(color="#EF9F27", width=1.5)))
    fig3.update_layout(
        title=f"{symbol} — MACD (Trend Momentum)",
        height=250,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ADX chart
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=adx, name="ADX",
        line=dict(color="#D85A30", width=2), fill="tozeroy",
        fillcolor="rgba(216,90,48,0.1)"))
    fig4.add_hline(y=25, line_dash="dash", line_color="#1D9E75",
        annotation_text="Strong trend (25)", annotation_position="right")
    fig4.add_hline(y=20, line_dash="dot", line_color="#EF9F27",
        annotation_text="Weak trend (20)", annotation_position="right")
    fig4.update_layout(
        title=f"{symbol} — ADX (Trend Strength)",
        height=250,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Key highlights summary
    st.subheader("📌 Key Highlights")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("52W High", f"₹{round(high_52w_val, 1)}")
    h2.metric("52W Low", f"₹{round(low_52w_val, 1)}")
    h3.metric("Current RSI", round(float(rsi.iloc[-1]), 1))
    h4.metric("ADX Strength", round(float(adx.iloc[-1]), 1))

    # Trend summary
    current_rsi = float(rsi.iloc[-1])
    current_adx = float(adx.iloc[-1])
    current_macd = float(macd_line.iloc[-1])
    current_signal = float(signal_line.iloc[-1])

    st.markdown("**Trend Summary:**")
    if current_macd > current_signal and current_adx > 25 and current_rsi < 70:
        st.success("✅ Strong uptrend — MACD positive, ADX strong, RSI healthy. Good zone to study for entry.")
    elif current_rsi > 70:
        st.warning("⚠️ Stock is overbought — RSI above 70. If you hold, consider booking partial profit.")
    elif current_adx < 20:
        st.info("👀 Weak trend — ADX below 20. Wait for stronger trend before entering.")
    else:
        st.info("👀 Moderate trend — Keep watching. Not the strongest entry point yet.")

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
    display_df = df.drop(columns=["Ticker"])
    st.dataframe(display_df.sort_values("Signal", ascending=True), use_container_width=True)

    st.subheader("📊 Historical Chart Analysis")
    st.caption("Select any matched stock below to see its full chart with trend highlights")

    selected = st.selectbox(
        "Choose a stock to analyse:",
        options=df["Symbol"].tolist(),
        index=0
    )

    if selected:
        ticker_row = df[df["Symbol"] == selected].iloc[0]
        ticker = ticker_row["Ticker"]
        st.markdown(f"### {selected} — Detailed Analysis")
        plot_stock_chart(ticker, selected)

    st.subheader("📖 How to read the signals")
    st.info("""
    ✅ Strong Buy → MACD up + ADX strong + RSI healthy. Good to study for entry.
    👀 Watch → Filters passed but trend not fully strong yet. Keep on radar.
    ⚠️ Near Exit → RSI getting high (above 70). If you hold this stock, consider booking profit.
    """)
else:
    st.warning("No stocks matched today's filters. Come back on Monday after 10 AM or tap Refresh.")
