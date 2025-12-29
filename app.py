import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 頁面設定 ---
st.set_page_config(page_title="DailyDip Pro AI", layout="wide", page_icon="🚀")

# --- 1. 設定板塊與觀察清單 ---
SECTORS = {
    "💎 Mag 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "MU", "INTC", "ARM"],
    "☁️ Software": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PLTR"],
    "🏦 Finance": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 核心功能 (使用 Cache 加速) ---

@st.cache_data(ttl=3600) # 緩存 1 小時，不用每次重新下載
def download_data():
    with st.spinner('🚀 下載市場數據中... (首次執行需約 30 秒)'):
        data_d = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=False)
        data_h = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    return data_d, data_h

def identify_smc_features(df):
    """SMC 特徵識別"""
    features = {"FVG": [], "DISP": []}
    # 簡單 FVG 識別
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({'type': 'Bullish', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': df.index[i-1]})
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({'type': 'Bearish', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': df.index[i-1]})
    return features

def plot_chart(df, ticker, timeframe):
    """使用 Streamlit 顯示圖表"""
    if len(df) < 30: return None
    
    # 準備數據
    plot_df = df.tail(60)
    swing_high = plot_df['High'].max()
    swing_low = plot_df['Low'].min()
    eq = (swing_high + swing_low) / 2
    smc = identify_smc_features(plot_df)

    # 設定風格
    mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
    
    # 繪圖
    hlines = dict(hlines=[swing_high, swing_low, eq], colors=['#ef4444', '#10b981', '#3b82f6'], linewidths=[1, 1, 0.5], linestyle=['--', '--', '-.'])
    
    fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
        title=dict(title=f"{ticker} - {timeframe}", color='white', size=12),
        hlines=hlines, figsize=(6, 4), returnfig=True)
    
    ax = axlist[0]
    x_min, x_max = ax.get_xlim()
    
    # 畫 FVG
    for fvg in smc['FVG']:
        try:
            idx = plot_df.index.get_loc(fvg['index'])
            color = '#10b981' if fvg['type'] == 'Bullish' else '#ef4444'
            rect = patches.Rectangle((idx, fvg['bottom']), x_max-idx, fvg['top']-fvg['bottom'], linewidth=0, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
        except: pass
        
    return fig, swing_high, swing_low, eq

# --- 3. 主程式介面 ---

st.title("🚀 DailyDip Pro: AI Market Scanner")
st.markdown("SMC Analysis • Dual Timeframe • AI Strategy")

# 1. 獲取數據
try:
    data_daily, data_hourly = download_data()
    
    # 處理 SPY 回報率
    if isinstance(data_daily.columns, pd.MultiIndex):
        spy_ret = data_daily['SPY']['Close'].pct_change()
    else:
        spy_ret = data_daily['Close'].pct_change() # Fallback

except Exception as e:
    st.error(f"數據下載失敗: {e}")
    st.stop()

# 2. 側邊欄篩選器
st.sidebar.header("🔍 篩選設定")
min_vol = st.sidebar.number_input("最小月成交額 (USD)", value=900000000)
min_beta = st.sidebar.slider("最小 Beta", 0.0, 3.0, 1.0)
filter_on = st.sidebar.checkbox("僅顯示符合篩選條件的股票", value=True)

# 3. 分析與顯示
tabs = st.tabs(list(SECTORS.keys()))

for i, (sector_name, tickers) in enumerate(SECTORS.items()):
    with tabs[i]:
        st.subheader(f"{sector_name}")
        
        # 使用 Columns 佈局 (每行 3 張卡片)
        cols = st.columns(3)
        col_idx = 0
        
        for t in tickers:
            try:
                # 處理數據
                if isinstance(data_daily.columns, pd.MultiIndex):
                    try:
                        df_d = data_daily[t].dropna()
                        df_h = data_hourly[t].dropna()
                    except: continue
                else: continue

                if len(df_d) < 200: continue
                
                curr_price = df_d['Close'].iloc[-1]
                
                # 計算指標
                sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
                
                stock_ret = df_d['Close'].pct_change()
                combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
                beta = combo['S'].cov(combo['M']) / combo['M'].var() if len(combo) > 30 else 0
                
                # 篩選判斷
                is_pass = (curr_price > sma200 and vol > min_vol and beta >= min_beta)
                
                # 訊號判斷 (快速計算)
                tp = df_d['High'].tail(20).max()
                sl = df_d['Low'].tail(20).min()
                range_len = tp - sl
                pos_pct = (curr_price - sl) / range_len if range_len > 0 else 0.5
                signal = "LONG" if pos_pct < 0.4 else "WAIT"
                
                # 如果開啟篩選且不符合，則跳過
                if filter_on and not (is_pass or signal == "LONG"):
                    continue

                # 顯示卡片
                with cols[col_idx % 3]:
                    # 邊框與標題
                    with st.container(border=True):
                        st.markdown(f"### {t} <span style='float:right; font-size:0.8em; padding:2px 6px; border-radius:4px; background:{'rgba(16,185,129,0.2)' if signal=='LONG' else 'rgba(148,163,184,0.1)'}; color:{'#10b981' if signal=='LONG' else '#94a3b8'}'>{signal}</span>", unsafe_allow_html=True)
                        st.metric("Price", f"${curr_price:.2f}", delta=f"Beta: {beta:.2f}")
                        
                        # AI 分析文字
                        if signal == "LONG":
                            rr = (tp - curr_price) / (curr_price - sl*0.98) if (curr_price - sl*0.98) > 0 else 0
                            st.success(f"**Action:** Buy (Discount)\n\n**TP:** ${tp:.2f} | **SL:** ${sl*0.98:.2f} | **RR:** {rr:.1f}R")
                        else:
                            eq = (tp + sl) / 2
                            st.warning(f"**Action:** Wait\n\nPrice in Premium. Wait for pullback to EQ: ${eq:.2f}")

                        # 展開看圖表 (這是解決卡頓的關鍵！用戶點擊才畫圖)
                        with st.expander("查看圖表 (Daily & Hourly)"):
                            # 只有展開時才畫圖，節省超多資源
                            fig_d, _, _, _ = plot_chart(df_d, t, "Daily")
                            st.pyplot(fig_d)
                            
                            fig_h, _, _, _ = plot_chart(df_h if not df_h.empty else df_d, t, "Hourly")
                            st.pyplot(fig_h)
                            
                col_idx += 1
                
            except Exception as e:
                continue
