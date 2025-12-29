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
import matplotlib.patches as patches


# --- 1. 設定板塊與觀察清單 ---
SECTORS = {
    "💎 Magnificent 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "MU", "INTC", "ARM"],
    "☁️ Software & Crypto": ["PLTR", "COIN", "MSTR", "CRM", "SNOW"],
    "🏦 Finance & Retail": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]


# --- 2. 篩選參數 ---
FILTER_SMA_PERIOD = 200
FILTER_MIN_CAP = 2000000000
FILTER_MIN_MONTHLY_VOL = 900000000
FILTER_MIN_BETA = 1.0


# --- 3. SMC 核心識別邏輯 ---
def identify_smc_features(df):
    """識別 FVG, EQH/EQL, Displacement"""
    features = {"FVG": [], "EQH": [], "EQL": [], "DISP": []}
    
    # 1. 識別 Displacement (大於平均實體 2.5 倍)
    df['Body'] = abs(df['Close'] - df['Open'])
    avg_body = df['Body'].rolling(20).mean()
    features['DISP'] = df.index[df['Body'] > avg_body * 2.5].tolist()


    # 2. 識別 FVG (Fair Value Gaps)
    # 只需要找最近的幾個即可，避免圖表太亂
    for i in range(2, len(df)):
        # Bullish FVG: Low[i] > High[i-2]
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({
                'type': 'Bullish',
                'top': df['Low'].iloc[i],
                'bottom': df['High'].iloc[i-2],
                'index': df.index[i-1] # 畫在中間那根
            })
        # Bearish FVG: High[i] < Low[i-2]
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({
                'type': 'Bearish',
                'top': df['Low'].iloc[i-2],
                'bottom': df['High'].iloc[i],
                'index': df.index[i-1]
            })


    # 3. 識別 EQH / EQL (簡單版：尋找最近 30 根 K 線內的近似高低點)
    # 這裡我們只標示顯著的 Swing High/Low
    window = 5
    df['IsPivotHigh'] = df['High'] == df['High'].rolling(window*2+1, center=True).max()
    df['IsPivotLow'] = df['Low'] == df['Low'].rolling(window*2+1, center=True).min()
    
    highs = df[df['IsPivotHigh']]['High']
    lows = df[df['IsPivotLow']]['Low']
    
    # 檢查是否有價位非常接近的 (Equal Highs/Lows)
    threshold = 0.002 # 0.2% 誤差
    
    # 檢查 EQH
    checked_highs = []
    for date, price in highs.items():
        for date2, price2 in highs.items():
            if date == date2: continue
            if abs(price - price2) / price < threshold:
                # 避免重複標記
                if not any(abs(h - price)/price < threshold for h in checked_highs):
                    features['EQH'].append({'price': (price+price2)/2, 'date': max(date, date2)})
                    checked_highs.append(price)
    
    # 檢查 EQL
    checked_lows = []
    for date, price in lows.items():
        for date2, price2 in lows.items():
            if date == date2: continue
            if abs(price - price2) / price < threshold:
                if not any(abs(l - price)/price < threshold for l in checked_lows):
                    features['EQL'].append({'price': (price+price2)/2, 'date': max(date, date2)})
                    checked_lows.append(price)
                    
    return features


# --- 4. 繪圖核心函式 (Visualizing SMC) ---
def generate_chart_image(df, ticker, timeframe):
    try:
        plot_df = df.tail(60) # 只畫最後 60 根
        if len(plot_df) < 30: return None, 0, 0
        
        # 基礎 SMC 結構
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        
        # 識別進階 SMC 特徵 (基於這 60 根)
        smc = identify_smc_features(plot_df)
        
        # 風格設定
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        
        # 繪圖
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe} (SMC Smart Money)", color='white', size=11),
            figsize=(8, 5), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # --- A. Premium vs Discount Zones (背景色) ---
        # Premium (上層) - 淡淡的紅色
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.1)
        ax.add_patch(rect_prem)
        ax.text(x_min+1, swing_high, "PREMIUM (Sell)", color='#fca5a5', fontsize=8, verticalalignment='top', fontweight='bold', alpha=0.7)
        
        # Discount (下層) - 淡淡的綠色
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.1)
        ax.add_patch(rect_disc)
        ax.text(x_min+1, swing_low, "DISCOUNT (Buy)", color='#86efac', fontsize=8, verticalalignment='bottom', fontweight='bold', alpha=0.7)
        
        # EQ Line
        ax.axhline(eq, color='#3b82f6', linestyle='-.', linewidth=1, alpha=0.6)
        
        # --- B. FVG (Fair Value Gaps) ---
        for fvg in smc['FVG']:
            # 找到該 FVG 在圖表上的 x 座標
            try:
                idx = plot_df.index.get_loc(fvg['index'])
                # 畫矩形延伸到右邊
                color = '#10b981' if fvg['type'] == 'Bullish' else '#ef4444'
                rect = patches.Rectangle((idx, fvg['bottom']), x_max-idx, fvg['top']-fvg['bottom'], linewidth=0, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            except: pass


        # --- C. EQH / EQL ---
        for eqh in smc['EQH']:
             ax.axhline(eqh['price'], color='white', linestyle=':', linewidth=1.5)
             ax.text(x_max-2, eqh['price'], "EQH (Liq.)", color='white', fontsize=7, fontweight='bold', backgroundcolor='#333')


        for eql in smc['EQL']:
             ax.axhline(eql['price'], color='white', linestyle=':', linewidth=1.5)
             ax.text(x_max-2, eql['price'], "EQL (Liq.)", color='white', fontsize=7, fontweight='bold', backgroundcolor='#333')


        # --- D. Displacement (標記大動能 K 線) ---
        for disp_date in smc['DISP']:
            if disp_date in plot_df.index:
                idx = plot_df.index.get_loc(disp_date)
                high_val = plot_df.loc[disp_date]['High']
                ax.text(idx, high_val*1.01, "Disp.", color='#fbbf24', fontsize=6, ha='center')


        # 存檔
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low
    except Exception as e:
        print(f"Chart Error {ticker}: {e}")
        return None, 0, 0


# --- 5. 數據下載與處理 ---
print(f"🚀 正在下載數據...")
data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", group_by='ticker', progress=True)
data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=True)
spy_ret = data_daily['SPY']['Close'].pct_change()


print("\n🔍 正在進行 SMC 深度分析...")


sector_html_blocks = ""
passed_count = 0
screener_rows = ""


for sector, tickers in SECTORS.items():
    cards_in_sector = ""
    for t in tickers:
        try:
            df_d = data_daily[t].dropna()
            df_h = data_hourly[t].dropna()
            if len(df_d) < 200: continue
            
            current_price = df_d['Close'].iloc[-1]
            
            # 篩選條件
            sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
            dollar_vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
            combo = pd.DataFrame({'S': df_d['Close'].pct_change(), 'M': spy_ret}).dropna()
            beta = combo['S'].cov(combo['M']) / combo['M'].var() if len(combo)>30 else 0
            
            pass_filter = (current_price > sma200 and dollar_vol > FILTER_MIN_MONTHLY_VOL and beta >= FILTER_MIN_BETA)


            # 生成圖表 (含 SMC 特徵)
            img_d, tp, sl = generate_chart_image(df_d, t, "Daily")
            if not img_d: continue
            img_h, _, _ = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly")
            
            # 交易訊號與文案
            s_low, s_high = sl, tp
            range_len = s_high - s_low
            pos_pct = (current_price - s_low) / range_len if range_len > 0 else 0.5
            signal = "LONG" if pos_pct < 0.4 else "WAIT"
            cls = "b-long" if signal == "LONG" else "b-wait"
            
            # AI 分析文案
            if signal == "LONG":
                rr = (tp - current_price) / (current_price - sl*0.98) if (current_price - sl*0.98) > 0 else 0
                ai_text = (
                    f"<b>🟢 AI Bullish Scan for {t}:</b><br>"
                    f"1. <b>Displacement:</b> Look for energetic candles breaking structure.<br>"
                    f"2. <b>Liquidity:</b> Price is sweeping SSL at ${sl:.2f}.<br>"
                    f"3. <b>FVG:</b> Watch for entry inside Bullish Fair Value Gaps on H1.<br>"
                    f"4. <b>Action:</b> Buy in Discount Zone. Target BSL at ${tp:.2f}."
                )
                setup_html = f"""
                <div class="setup-grid">
                    <div><span>ENTRY</span><br><b>${current_price:.2f}</b></div>
                    <div><span>🛑 SL</span><br><b class="r">${sl:.2f}</b></div>
                    <div><span>🎯 TP</span><br><b class="g">${tp:.2f}</b></div>
                    <div><span>RR</span><br><b class="y">{rr:.1f}R</b></div>
                </div>"""
            else:
                ai_text = (
                    f"<b>⏳ AI Neutral Scan for {t}:</b><br>"
                    f"Price is currently in the <b>Premium Zone</b> (>50%).<br>"
                    f"Institutions typically sell here. Wait for price to drop into the Discount Zone (<${(s_low+range_len*0.4):.2f}) or look for Short setups at EQH."
                )
                setup_html = "<div class='setup-wait'>⏳ Wait for Discount</div>"


            # 卡片 HTML
            cards_in_sector += f"""
            <div class="card" onclick="openModal('{t}', '{img_d}', '{img_h}', '{signal}', `{ai_text}`)">
                <div class="head">
                    <div><div class="code">{t}</div><div class="price">${current_price:.2f}</div></div>
                    <span class="badge {cls}">{signal}</span>
                </div>
                {setup_html}
                <div class="hint">Tap for SMC Charts ↗</div>
            </div>
            """
            
            if pass_filter:
                passed_count += 1
                screener_rows += f"<tr><td><b>{t}</b></td><td>${current_price:.2f}</td><td class='g'>Pass</td><td>{beta:.2f}</td><td><span class='badge {cls}'>{signal}</span></td></tr>"


        except Exception as e: continue
            
    if cards_in_sector:
        sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"


print(f"\n✅ 分析完成！共 {passed_count} 檔股票符合嚴格篩選。正在生成網頁...")


# --- 6. 生成網頁 ---
full_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
    body {{ background:var(--bg); color:var(--text); font-family:-apple-system, sans-serif; margin:0; padding:20px; }}
    .tabs {{ display:flex; gap:10px; border-bottom:1px solid #334155; padding-bottom:15px; margin-bottom:20px; position:sticky; top:0; background:var(--bg); z-index:10; }}
    .tab {{ padding:10px 20px; background:#334155; border-radius:8px; cursor:pointer; color:#94a3b8; font-weight:bold; }}
    .tab.active {{ background:var(--acc); color:white; }}
    .content {{ display:none; animation:fadeIn 0.5s; }} .content.active {{ display:block; }}
    @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
    
    .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:30px 0 15px; color:#e2e8f0; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:15px; }}
    .card {{ background:var(--card); border:1px solid #334155; border-radius:12px; padding:15px; cursor:pointer; transition:0.2s; }}
    .card:hover {{ border-color:var(--acc); transform:translateY(-3px); }}
    .head {{ display:flex; justify-content:space-between; margin-bottom:10px; }}
    .code {{ font-size:1.4rem; font-weight:900; }} .price {{ color:#94a3b8; font-family:monospace; }}
    .badge {{ padding:4px 8px; border-radius:6px; font-size:0.8rem; font-weight:bold; height:fit-content; }}
    .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
    .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #334155; }}
    .setup-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; background:rgba(0,0,0,0.3); padding:10px; border-radius:8px; text-align:center; font-size:0.8rem; }}
    .setup-wait {{ background:rgba(0,0,0,0.3); padding:15px; border-radius:8px; text-align:center; color:#64748b; font-size:0.9rem; }}
    .g {{ color:var(--g); }} .r {{ color:var(--r); }} .y {{ color:var(--y); }}
    .hint {{ font-size:0.75rem; color:var(--acc); text-align:right; margin-top:10px; opacity:0.8; }}
    
    table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; }}
    th, td {{ padding:12px; text-align:left; border-bottom:1px solid #334155; }}
    th {{ background:#334155; color:#94a3b8; font-size:0.85rem; }}
    
    .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:999; justify-content:center; align-items:start; overflow-y:auto; padding:20px; }}
    .m-content {{ background:var(--card); width:100%; max-width:800px; padding:20px; border-radius:12px; border:1px solid #475569; margin-top:20px; }}
    .chart-box {{ margin-bottom:20px; }} .chart-title {{ color:var(--acc); font-weight:bold; display:block; margin-bottom:5px; }}
    .m-content img {{ width:100%; border-radius:8px; border:1px solid #334155; }}
    .ai-box {{ background:rgba(16,185,129,0.1); border-left:4px solid var(--g); padding:15px; border-radius:4px; margin-bottom:20px; line-height:1.6; color:#e2e8f0; font-size:0.9rem; }}
    .close-btn {{ width:100%; padding:15px; background:var(--acc); color:white; border:none; border-radius:8px; font-weight:bold; cursor:pointer; font-size:1rem; }}
</style>
</head>
<body>


<div class="tabs">
    <div class="tab active" onclick="setTab('overview', this)">📊 Sectors & SMC Analysis</div>
    <div class="tab" onclick="setTab('screener', this)">🔍 Strict Screener ({passed_count})</div>
</div>


<div id="overview" class="content active">{sector_html_blocks}</div>


<div id="screener" class="content">
    <div style="background:rgba(59,130,246,0.1); padding:15px; border-radius:8px; margin-bottom:20px; border:1px solid var(--acc); color:#cbd5e1;">
        <b>🎯 Filters:</b> Price > 200 SMA • Beta >= 1 • High Volume • High Cap
    </div>
    <table><thead><tr><th>Ticker</th><th>Price</th><th>Trend</th><th>Beta</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table>
</div>


<div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
    <div class="m-content" onclick="event.stopPropagation()">
        <h2 id="m-ticker" style="margin-top:0; color:white"></h2>
        <div id="m-ai" class="ai-box"></div>
        <div class="chart-box"><span class="chart-title">📅 Daily Structure (Premium vs Discount)</span><img id="img-d" src=""></div>
        <div class="chart-box"><span class="chart-title">⏱️ Hourly Structure (Execution)</span><img id="img-h" src=""></div>
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


display(HTML(full_html))
------------------------------------------------------------------------


