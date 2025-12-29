# --- 步驟 0: 強制安裝必要套件 ---
import sys
import subprocess
print("⚙️ 正在檢查並安裝必要套件...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "mplfinance"])
print("✅ 套件準備完成！\n")

import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# --- 1. 設定觀察清單 (100+ 檔熱門美股) ---
# 包含七巨頭、半導體、SaaS、金融、傳產龍頭
tickers = [
    # Mag 7 & Tech Giants
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO",
    "ORCL", "CRM", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "IBM", "UBER", "ABNB",
    "PLTR", "NOW", "SNOW", "PANW", "CRWD", "PALW", "SQ", "SHOP", "COIN", "MSTR",
    "HOOD", "DKNG", "RBLX", "U", "TTD", "NET", "ZM", "DOCU", "TEAM", "MDB",
    # Semiconductor
    "TSM", "ASML", "AMAT", "LRCX", "MU", "ADI", "NXPI", "MRVL", "KLAC", "ON",
    "GGFS", "INTC", "STM", "ARM", "SMCI",
    # Finance & Payments
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "V", "MA", "AXP", "PYPL", "COF",
    # Healthcare
    "LLY", "JNJ", "UNH", "ABBV", "MRK", "PFE", "TMO", "DHR", "ISRG", "VRTX", "REGN",
    # Consumer & Retail
    "WMT", "COST", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE", "LULU", "CMG", "KO", "PEP", "PG",
    # Industrial & Energy
    "XOM", "CVX", "COP", "SLB", "GE", "CAT", "DE", "BA", "LMT", "RTX", "HON", "UPS", "UNP",
    # Entertainment & Comm
    "DIS", "CMCSA", "TMUS", "VZ", "T", "SPOT"
]

# 確保不重複
tickers = list(set(tickers))

# --- 2. 篩選條件 (Strict Filters) ---
FILTER_SMA_PERIOD = 200
FILTER_MIN_CAP = 2000000000        # 2 Billion USD
FILTER_MIN_MONTHLY_VOL = 900000000 # 900 Million USD
FILTER_MIN_BETA = 1.0              # Beta >= 1

# --- 3. 核心功能 ---

def generate_chart_base64(df, ticker):
    try:
        window = 20
        if len(df) < window: return None, 0, 0
        
        swing_high = df['High'].tail(window).max()
        swing_low = df['Low'].tail(window).min()
        current_price = df['Close'].iloc[-1]
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        hlines = dict(hlines=[swing_high, swing_low], colors=['#ef4444', '#10b981'], linewidths=[1, 1], linestyle='--')

        fig, _ = mpf.plot(df.tail(60), type='candle', style=s, volume=False,
            title=dict(title=f"{ticker}", color='white', size=10),
            hlines=hlines, figsize=(4, 2.5), returnfig=True) # 縮小尺寸以適應大量圖片
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low
    except:
        return None, 0, 0

def get_market_caps(ticker_list):
    # 由於 yf.download 不含市值，我們需要單獨抓取
    # 為了速度，這裡使用簡單迴圈，但只抓 info
    caps = {}
    print(f"   Fetching Market Caps for {len(ticker_list)} stocks...")
    for t in ticker_list:
        try:
            caps[t] = yf.Ticker(t).info.get('marketCap', 0)
        except:
            caps[t] = 0
    return caps

# --- 4. 主程式 ---
print(f"🚀 正在下載 100+ 檔股票數據 ({len(tickers)} Tickers)...")

# 1. 批量下載價格數據 (大幅加速)
data = yf.download(tickers + ["SPY"], period="1y", group_by='ticker', progress=True)

print("🔍 正在進行技術分析與篩選 (SMC + Screener)...")

# 準備大盤數據算 Beta
spy_close = data['SPY']['Close']
spy_ret = spy_close.pct_change()

# 抓取市值 (這步比較慢，需耐心等待)
market_caps = get_market_caps(tickers)

smc_cards_html = ""
screener_rows_html = ""
passed_count = 0

for t in tickers:
    try:
        df = data[t].dropna()
        if df.empty or len(df) < 200: continue
        
        current_price = df['Close'].iloc[-1]
        
        # --- 計算指標 ---
        # 1. SMA 200
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        
        # 2. Beta
        stock_ret = df['Close'].pct_change()
        # 對齊索引
        aligned_data = pd.DataFrame({'Stock': stock_ret, 'Market': spy_ret}).dropna()
        if len(aligned_data) < 30: 
            beta = 0
        else:
            cov = aligned_data['Stock'].rolling(252).cov(aligned_data['Market']).iloc[-1]
            var = aligned_data['Market'].rolling(252).var().iloc[-1]
            beta = cov / var if var != 0 else 0
            
        # 3. 月成交額
        dollar_vol = (df['Close'] * df['Volume']).rolling(21).mean().iloc[-1] * 21
        
        # 4. 市值
        mkt_cap = market_caps.get(t, 0)
        
        # --- 判斷篩選條件 ---
        pass_filter = (
            current_price > sma200 and
            mkt_cap > FILTER_MIN_CAP and
            dollar_vol > FILTER_MIN_MONTHLY_VOL and
            beta >= FILTER_MIN_BETA
        )
        
        # --- 生成 SMC 圖表與信號 ---
        img_src, tp, sl = generate_chart_base64(df, t)
        if not img_src: continue
        
        s_low, s_high = sl, tp
        range_len = s_high - s_low
        pos_pct = (current_price - s_low) / range_len if range_len > 0 else 0.5
        
        signal = "LONG" if pos_pct < 0.4 else "WAIT"
        cls = "b-long" if signal == "LONG" else "b-wait"
        
        # 生成 SMC 卡片 HTML
        # 交易計畫
        rr = (tp - current_price) / (current_price - sl*0.98) if (current_price - sl*0.98) > 0 else 0
        setup_html = ""
        if signal == "LONG":
            setup_html = f"<div class='plan'>TP: <span class='g'>${tp:.1f}</span> | RR: <span class='y'>{rr:.1f}R</span></div>"
        
        smc_cards_html += f"""
        <div class="card" onclick="openModal('{img_src}', '{t}')">
            <div class="head"><b>{t}</b> <span class="badge {cls}">{signal}</span></div>
            <div class="price">${current_price:.2f}</div>
            {setup_html}
        </div>"""
        
        # 生成篩選器表格行 (只顯示通過的)
        if pass_filter:
            passed_count += 1
            vol_str = f"${dollar_vol/1e6:.0f}M"
            cap_str = f"${mkt_cap/1e9:.1f}B"
            screener_rows_html += f"""
            <tr>
                <td><b>{t}</b></td>
                <td>${current_price:.2f}</td>
                <td class="g">Above</td>
                <td>{beta:.2f}</td>
                <td>{vol_str}</td>
                <td>{cap_str}</td>
                <td><span class="badge {cls}">{signal}</span></td>
            </tr>"""
            
    except Exception as e:
        continue

print(f"\n✅ 分析完成！共 {passed_count} 檔股票符合嚴格篩選條件。")

# --- 5. 生成網頁 ---
full_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; }}
    body {{ background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding:20px; }}
    
    /* Tabs */
    .tabs {{ display:flex; gap:10px; border-bottom:1px solid #334155; padding-bottom:10px; margin-bottom:20px; }}
    .tab {{ padding:10px 20px; background:#334155; border-radius:6px; cursor:pointer; color:#94a3b8; font-weight:bold; transition:0.2s; }}
    .tab:hover {{ background:#475569; }}
    .tab.active {{ background:var(--acc); color:white; }}
    
    .content {{ display:none; animation: fadeIn 0.4s; }}
    .content.active {{ display:block; }}
    @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}

    /* SMC Grid */
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:12px; }}
    .card {{ background:var(--card); border:1px solid #334155; border-radius:8px; padding:12px; cursor:pointer; transition:0.2s; }}
    .card:hover {{ border-color:var(--acc); transform:translateY(-3px); }}
    
    .head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }}
    .price {{ font-size:1.1rem; font-weight:bold; color:#cbd5e1; }}
    .badge {{ padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; }}
    .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
    .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #334155; }}
    .plan {{ font-size:0.75rem; background:rgba(0,0,0,0.3); padding:4px; border-radius:4px; margin-top:5px; color:#94a3b8; }}
    .g {{ color:var(--g); }} .y {{ color:#fbbf24; }}

    /* Screener Table */
    .table-container {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; min-width:600px; }}
    th, td {{ padding:12px; text-align:left; border-bottom:1px solid #334155; }}
    th {{ background:#334155; font-size:0.8rem; color:#94a3b8; text-transform:uppercase; }}
    tr:hover {{ background:rgba(255,255,255,0.05); }}
    
    /* Modal */
    .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); justify-content:center; align-items:center; z-index:999; backdrop-filter:blur(3px); }}
    .m-content {{ background:var(--card); padding:20px; border-radius:12px; max-width:600px; width:95%; border:1px solid #475569; text-align:center; }}
    .m-content img {{ width:100%; border-radius:8px; border:1px solid #334155; }}
    .close-btn {{ margin-top:15px; background:var(--acc); color:white; border:none; padding:10px 30px; border-radius:6px; cursor:pointer; font-weight:bold; }}
</style>
</head>
<body>

<div class="tabs">
    <div class="tab active" onclick="setTab('smc', this)">📊 Market Overview ({len(tickers)})</div>
    <div class="tab" onclick="setTab('screen', this)">🔍 Strict Screener ({passed_count})</div>
</div>

<div id="smc" class="content active">
    <div class="grid">{smc_cards_html}</div>
</div>

<div id="screen" class="content">
    <div style="margin-bottom:15px; padding:10px; background:rgba(59,130,246,0.15); border-left:4px solid var(--acc); color:#cbd5e1; font-size:0.9rem; border-radius:4px;">
        <b>🎯 Screening Criteria:</b><br>
        1. Price > 200 SMA (Uptrend)<br>
        2. Market Cap > $2 Billion<br>
        3. Monthly Vol > $900 Million<br>
        4. Beta >= 1 (High Volatility)
    </div>
    <div class="table-container">
        <table>
            <thead><tr><th>Ticker</th><th>Price</th><th>vs 200SMA</th><th>Beta</th><th>Mth Vol</th><th>Cap</th><th>Signal</th></tr></thead>
            <tbody>
                {screener_rows_html if screener_rows_html else "<tr><td colspan='7' style='text-align:center;padding:30px'>No stocks match the criteria.</td></tr>"}
            </tbody>
        </table>
    </div>
</div>

<div id="modal" class="modal" onclick="this.style.display='none'">
    <div class="m-content" onclick="event.stopPropagation()">
        <h3 id="m-title" style="margin-top:0; color:white">Chart</h3>
        <img id="m-img" src="">
        <br>
        <button class="close-btn" onclick="document.getElementById('modal').style.display='none'">Close</button>
    </div>
</div>

<script>
function setTab(id, el) {{
    document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    el.classList.add('active');
}}
function openModal(src, title) {{
    document.getElementById('modal').style.display = 'flex';
    document.getElementById('m-img').src = src;
    document.getElementById('m-title').innerText = title + " Structure";
}}
</script>

</body>
</html>
"""

display(HTML(full_html))
