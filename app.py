# --- 步驟 0: 安裝必要套件 ---
import sys
import subprocess
print("⚙️ 正在安裝必要套件...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "mplfinance"])
print("✅ 安裝完成！\n")

import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# --- 1. 設定板塊與觀察清單 (Sector Categorization) ---
SECTORS = {
    "💎 Magnificent 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO", "ORCL"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "LRCX", "MU", "ADI", "MRVL", "KLAC", "ON", "INTC", "ARM", "SMCI"],
    "☁️ Software & SaaS": ["CRM", "ADBE", "PLTR", "NOW", "SNOW", "PANW", "CRWD", "SQ", "SHOP", "NET", "MDB", "TEAM"],
    "🚀 Crypto & High Growth": ["COIN", "MSTR", "HOOD", "DKNG", "RBLX", "U", "TTD", "ZM", "DOCU", "CVNA"],
    "🏦 Finance & Fintech": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "PYPL"],
    "💊 Healthcare": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "PFE", "ISRG", "VRTX"],
    "🛒 Consumer & Retail": ["WMT", "COST", "TGT", "HD", "MCD", "SBUX", "NKE", "KO", "PEP", "PG"],
    "🛢️ Industrial & Energy": ["XOM", "CVX", "SLB", "GE", "CAT", "DE", "BA", "LMT", "UPS", "UNP"],
    "🎬 Entertainment & Comm": ["DIS", "NFLX", "CMCSA", "TMUS", "VZ", "T", "SPOT", "UBER", "ABNB"]
}

# 扁平化清單用於下載
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 篩選參數 ---
FILTER_SMA_PERIOD = 200
FILTER_MIN_CAP = 2000000000       # 2 Billion
FILTER_MIN_MONTHLY_VOL = 900000000 
FILTER_MIN_BETA = 1.0

# --- 3. 核心繪圖函式 (支援註解) ---
def generate_chart_image(df, ticker, timeframe):
    try:
        window = 20
        if len(df) < window: return None, 0, 0
        
        swing_high = df['High'].tail(window).max()
        swing_low = df['Low'].tail(window).min()
        eq = (swing_high + swing_low) / 2
        current = df['Close'].iloc[-1]
        
        # 風格設定
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        
        # 線條設定
        hlines = dict(hlines=[swing_high, swing_low, eq], colors=['#ef4444', '#10b981', '#3b82f6'], linewidths=[1.5, 1.5, 1], linestyle=['--', '--', '-.'])

        # 繪圖
        fig, axlist = mpf.plot(df.tail(60), type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe}", color='white', size=12),
            hlines=hlines, figsize=(8, 5), returnfig=True)
        
        # --- 自動註解 (Annotation) ---
        ax = axlist[0]
        x_pos = len(df.tail(60)) - 1
        
        # 標註 BSL (目標)
        ax.text(x_pos, swing_high, f' BSL (Target)\n ${swing_high:.2f}', color='#ef4444', fontsize=9, fontweight='bold', verticalalignment='bottom')
        # 標註 SSL (止損)
        ax.text(x_pos, swing_low, f' SSL (Support)\n ${swing_low:.2f}', color='#10b981', fontsize=9, fontweight='bold', verticalalignment='top')
        # 標註 EQ
        ax.text(0, eq, f' EQ (50%)', color='#3b82f6', fontsize=8, alpha=0.7)

        # 存檔
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low
    except:
        return None, 0, 0

# --- 4. 數據下載與處理 ---
print(f"🚀 正在下載 {len(ALL_TICKERS)} 檔股票數據...")
print("   - 下載日線數據 (Daily)...")
data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=True)

print("   - 下載小時線數據 (Hourly)...")
data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=True)

# 準備大盤 Beta 計算
spy_ret = data_daily['SPY']['Close'].pct_change()

# 獲取市值 (簡易版，大量抓取會慢，這邊用模擬檢查或快速跳過)
# 為了演示流暢度，這裡假設清單內股票皆為大型股符合資格
# 實際使用可解除下方註解
# market_caps = {} 
# for t in ALL_TICKERS: try: market_caps[t] = yf.Ticker(t).info.get('marketCap', 0); except: market_caps[t] = 0

print("\n🔍 正在進行板塊分析與 AI 策略運算...")

sector_html_blocks = ""
screener_rows = ""
passed_count = 0

for sector, tickers in SECTORS.items():
    
    cards_in_sector = ""
    
    for t in tickers:
        try:
            # 取得該股數據
            df_d = data_daily[t].dropna()
            df_h = data_hourly[t].dropna()
            
            if len(df_d) < 200: continue
            
            current_price = df_d['Close'].iloc[-1]
            
            # --- 篩選計算 ---
            sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
            dollar_vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
            
            # Beta
            stock_ret = df_d['Close'].pct_change()
            combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
            if len(combo) > 30:
                beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1]
            else:
                beta = 0
            
            # 判斷篩選
            pass_filter = (
                current_price > sma200 and 
                dollar_vol > FILTER_MIN_MONTHLY_VOL and 
                beta >= FILTER_MIN_BETA
            )

            # --- SMC 圖表生成 ---
            img_d, tp, sl = generate_chart_image(df_d, t, "Daily Chart")
            if not img_d: continue
            
            # 小時圖 (若無數據則用日線替代)
            img_h, _, _ = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly Chart")
            
            # 訊號判斷
            s_low, s_high = sl, tp
            range_len = s_high - s_low
            pos_pct = (current_price - s_low) / range_len if range_len > 0 else 0.5
            
            signal = "LONG" if pos_pct < 0.4 else "WAIT"
            cls = "b-long" if signal == "LONG" else "b-wait"
            
            # --- AI 分析與交易計畫 ---
            ai_analysis = ""
            setup_display = ""
            
            if signal == "LONG":
                # 交易參數
                entry = current_price
                stop_loss = s_low * 0.98 # 技術止損
                take_profit = s_high
                risk = entry - stop_loss
                reward = take_profit - entry
                rr = reward / risk if risk > 0 else 0
                
                # AI 文案
                trend_st = "Bullish (Above 200MA)" if current_price > sma200 else "Recovering"
                ai_analysis = (
                    f"<b>AI Analysis:</b> {t} is in a {trend_st} trend. "
                    f"Price is in the Discount Zone (<40%). "
                    f"Structure suggests a sweep of liquidity at ${s_low:.2f} (SSL). "
                    f"Targeting upside liquidity at ${s_high:.2f} (BSL)."
                )
                
                setup_display = f"""
                <div class="setup-grid">
                    <div><span class="lbl">ENTRY</span><br><b>${entry:.2f}</b></div>
                    <div><span class="lbl">🛑 SL</span><br><b class="r">${stop_loss:.2f}</b></div>
                    <div><span class="lbl">🎯 TP</span><br><b class="g">${take_profit:.2f}</b></div>
                    <div><span class="lbl">RR</span><br><b class="y">{rr:.1f}R</b></div>
                </div>
                """
            else:
                ai_analysis = f"<b>AI Analysis:</b> {t} is currently in Premium pricing. Wait for a pullback to the Discount zone (${(s_low + range_len*0.4):.2f}) before entering."
                setup_display = "<div class='setup-wait'>⏳ Wait for Setup</div>"

            # --- 生成卡片 HTML ---
            cards_in_sector += f"""
            <div class="card" onclick="openModal('{t}', '{img_d}', '{img_h}', '{signal}', '{ai_analysis}')">
                <div class="head">
                    <div><div class="code">{t}</div><div class="price">${current_price:.2f}</div></div>
                    <span class="badge {cls}">{signal}</span>
                </div>
                {setup_display}
                <div class="hint">Tap for D1 & H1 Charts ↗</div>
            </div>
            """
            
            # --- 生成篩選器表格 ---
            if pass_filter:
                passed_count += 1
                screener_rows += f"""
                <tr>
                    <td><b>{t}</b></td>
                    <td>${current_price:.2f}</td>
                    <td class="g">Pass</td>
                    <td>{beta:.2f}</td>
                    <td>${dollar_vol/1e6:.0f}M</td>
                    <td><span class="badge {cls}">{signal}</span></td>
                </tr>
                """

        except Exception as e:
            continue
            
    # 將該板塊加入 HTML 區塊
    if cards_in_sector:
        sector_html_blocks += f"""
        <h3 class="sector-title">{sector}</h3>
        <div class="grid">
            {cards_in_sector}
        </div>
        """

print(f"\n✅ 分析完成！共 {passed_count} 檔股票符合嚴格篩選。正在生成網頁...")

# --- 5. 網頁生成 ---
full_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
    body {{ background:var(--bg); color:var(--text); font-family:-apple-system, BlinkMacSystemFont, sans-serif; margin:0; padding:20px; }}
    
    /* Tabs */
    .tabs {{ display:flex; gap:10px; border-bottom:1px solid #334155; padding-bottom:15px; margin-bottom:20px; position:sticky; top:0; background:var(--bg); z-index:10; }}
    .tab {{ padding:10px 20px; background:#334155; border-radius:8px; cursor:pointer; color:#94a3b8; font-weight:bold; transition:0.2s; }}
    .tab.active {{ background:var(--acc); color:white; }}
    
    .content {{ display:none; animation:fadeIn 0.4s; }}
    .content.active {{ display:block; }}
    @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}

    /* Layout */
    .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:30px 0 15px; color:#e2e8f0; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:15px; }}
    
    /* Card */
    .card {{ background:var(--card); border:1px solid #334155; border-radius:12px; padding:15px; cursor:pointer; transition:0.2s; position:relative; }}
    .card:hover {{ border-color:var(--acc); transform:translateY(-3px); }}
    
    .head {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }}
    .code {{ font-size:1.4rem; font-weight:900; }}
    .price {{ color:#94a3b8; font-family:monospace; font-size:1.1rem; }}
    
    .badge {{ padding:4px 8px; border-radius:6px; font-size:0.8rem; font-weight:bold; height:fit-content; }}
    .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
    .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #334155; }}
    
    /* Setup Grid inside Card */
    .setup-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; background:rgba(0,0,0,0.3); padding:10px; border-radius:8px; margin-top:5px; text-align:center; }}
    .lbl {{ font-size:0.7rem; color:#64748b; font-weight:bold; }}
    .setup-wait {{ background:rgba(0,0,0,0.3); padding:15px; border-radius:8px; text-align:center; color:#64748b; font-size:0.9rem; }}
    
    .g {{ color:var(--g); }} .r {{ color:var(--r); }} .y {{ color:var(--y); }}
    .hint {{ font-size:0.75rem; color:var(--acc); text-align:right; margin-top:10px; opacity:0.8; }}

    /* Screener Table */
    table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; }}
    th, td {{ padding:12px; text-align:left; border-bottom:1px solid #334155; }}
    th {{ background:#334155; color:#94a3b8; font-size:0.85rem; text-transform:uppercase; }}
    tr:hover {{ background:rgba(255,255,255,0.05); }}

    /* Modal */
    .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:999; justify-content:center; align-items:start; overflow-y:auto; padding:20px; }}
    .m-content {{ background:var(--card); width:100%; max-width:800px; padding:20px; border-radius:12px; border:1px solid #475569; margin-top:20px; }}
    
    .chart-box {{ margin-bottom:20px; }}
    .chart-title {{ color:var(--acc); font-weight:bold; margin-bottom:5px; display:block; }}
    .m-content img {{ width:100%; border-radius:8px; border:1px solid #334155; }}
    
    .ai-box {{ background:rgba(59,130,246,0.1); border-left:4px solid var(--acc); padding:15px; border-radius:4px; margin-bottom:20px; line-height:1.5; color:#e2e8f0; }}
    
    .close-btn {{ width:100%; padding:15px; background:var(--acc); color:white; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:1rem; }}
</style>
</head>
<body>

<div class="tabs">
    <div class="tab active" onclick="setTab('overview', this)">📊 Sectors & Charts</div>
    <div class="tab" onclick="setTab('screener', this)">🔍 Strict Screener ({passed_count})</div>
</div>

<div id="overview" class="content active">
    {sector_html_blocks}
</div>

<div id="screener" class="content">
    <div style="background:rgba(16,185,129,0.1); padding:15px; border-radius:8px; margin-bottom:20px; border:1px solid var(--g);">
        <b>🎯 Filter Criteria:</b> Price > 200 SMA • High Liquidity • High Volatility (Beta > 1)
    </div>
    <table>
        <thead><tr><th>Ticker</th><th>Price</th><th>200SMA</th><th>Beta</th><th>Vol</th><th>Signal</th></tr></thead>
        <tbody>
            {screener_rows if screener_rows else "<tr><td colspan='6' style='text-align:center;padding:20px'>No stocks match currently.</td></tr>"}
        </tbody>
    </table>
</div>

<div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
    <div class="m-content" onclick="event.stopPropagation()">
        <h2 id="m-ticker" style="margin-top:0; color:white"></h2>
        
        <div id="m-ai" class="ai-box"></div>
        
        <div class="chart-box">
            <span class="chart-title">📅 Daily Structure (Trend & Levels)</span>
            <img id="img-d" src="">
        </div>
        
        <div class="chart-box">
            <span class="chart-title">⏱️ Hourly Structure (Entry Confirmation)</span>
            <img id="img-h" src="">
        </div>
        
        <button class="close-btn" onclick="document.getElementById('modal').style.display='none'">Close Analysis</button>
    </div>
</div>

<script>
function setTab(id, el) {{
    document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    el.classList.add('active');
}}

function openModal(ticker, d_src, h_src, signal, ai_text) {{
    document.getElementById('modal').style.display = 'flex';
    document.getElementById('m-ticker').innerText = ticker + " (" + signal + ")";
    document.getElementById('m-ai').innerHTML = ai_text;
    document.getElementById('img-d').src = d_src;
    document.getElementById('img-h').src = h_src;
}}
</script>

</body>
</html>
"""

print("\n✅ 網站生成完畢！請向下捲動查看 👇")
display(HTML(full_html))
