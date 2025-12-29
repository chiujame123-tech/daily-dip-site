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
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({
                'type': 'Bullish', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': df.index[i-1]
            })
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({
                'type': 'Bearish', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': df.index[i-1]
            })

    # 3. 識別 EQH / EQL
    window = 5
    highs = df[df['High'] == df['High'].rolling(window*2+1, center=True).max()]['High']
    lows = df[df['Low'] == df['Low'].rolling(window*2+1, center=True).min()]['Low']
    
    threshold = 0.002
    checked_highs = []
    for date, price in highs.items():
        for date2, price2 in highs.items():
            if date == date2: continue
            if abs(price - price2) / price < threshold:
                if not any(abs(h - price)/price < threshold for h in checked_highs):
                    features['EQH'].append({'price': (price+price2)/2, 'date': max(date, date2)})
                    checked_highs.append(price)
    
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
        plot_df = df.tail(60)
        if len(plot_df) < 30: return None, 0, 0
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        
        smc = identify_smc_features(plot_df)
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe} (SMC)", color='white', size=11),
            figsize=(8, 5), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # Zones
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.1)
        ax.add_patch(rect_prem)
        ax.text(x_min+1, swing_high, "PREMIUM", color='#fca5a5', fontsize=8, verticalalignment='top', alpha=0.7)
        
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.1)
        ax.add_patch(rect_disc)
        ax.text(x_min+1, swing_low, "DISCOUNT", color='#86efac', fontsize=8, verticalalignment='bottom', alpha=0.7)
        
        ax.axhline(eq, color='#3b82f6', linestyle='-.', linewidth=1, alpha=0.6)
        
        # Features
        for fvg in smc['FVG']:
            try:
                idx = plot_df.index.get_loc(fvg['index'])
                color = '#10b981' if fvg['type'] == 'Bullish' else '#ef4444'
                rect = patches.Rectangle((idx, fvg['bottom']), x_max-idx, fvg['top']-fvg['bottom'], linewidth=0, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            except: pass

        for eqh in smc['EQH']:
             ax.axhline(eqh['price'], color='white', linestyle=':', linewidth=1.5)
             ax.text(x_max-2, eqh['price'], "EQH", color='white', fontsize=7, backgroundcolor='#333')

        for eql in smc['EQL']:
             ax.axhline(eql['price'], color='white', linestyle=':', linewidth=1.5)
             ax.text(x_max-2, eql['price'], "EQL", color='white', fontsize=7, backgroundcolor='#333')

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

print("\n🔍 正在進行 SMC 深度分析與生成部署建議...")

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

            # 生成圖表
            img_d, tp, sl = generate_chart_image(df_d, t, "Daily")
            if not img_d: continue
            img_h, _, _ = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly")
            
            # 交易訊號
            s_low, s_high = sl, tp
            range_len = s_high - s_low
            pos_pct = (current_price - s_low) / range_len if range_len > 0 else 0.5
            signal = "LONG" if pos_pct < 0.4 else "WAIT"
            cls = "b-long" if signal == "LONG" else "b-wait"
            
            # --- 🔥 新增：AI 部署建議邏輯 (Deployment Logic) ---
            deployment_html = ""
            trend_str = "Bullish (>200MA)" if current_price > sma200 else "Neutral"
            
            if signal == "LONG":
                # 做多情境
                entry_target = current_price
                stop_loss = sl * 0.98  # 前低 - 2%
                take_profit = tp       # 前高 BSL
                risk = entry_target - stop_loss
                reward = take_profit - entry_target
                rr = reward / risk if risk > 0 else 0
                
                deployment_html = f"""
                <div class="deploy-box long">
                    <div class="deploy-title">✅ 建議部署：現價買入 / 分批建倉</div>
                    <ul class="deploy-list">
                        <li><b>入手價位：</b> ${entry_target:.2f} (目前處於折價區)</li>
                        <li><b>止損位置：</b> ${stop_loss:.2f} (前低下方 2% 緩衝)</li>
                        <li><b>獲利目標：</b> ${take_profit:.2f} (上方流動性 BSL)</li>
                        <li><b>操作理由：</b> 股價已回落至 Discount Zone (<40%)，且維持{trend_str}趨勢。潛在盈虧比 {rr:.1f}R。</li>
                    </ul>
                </div>
                """
            else:
                # 觀望情境
                eq_price = (s_high + s_low) / 2
                discount_entry = s_low + (range_len * 0.4)
                
                deployment_html = f"""
                <div class="deploy-box wait">
                    <div class="deploy-title">⏳ 建議部署：等待回調 (Do Not Chase)</div>
                    <ul class="deploy-list">
                        <li><b>觀察價位：</b> 等待回落至平衡點 <b>${eq_price:.2f}</b> 以下。</li>
                        <li><b>理想買點：</b> <b>${discount_entry:.2f}</b> (進入折價區)。</li>
                        <li><b>操作理由：</b> 目前股價處於 Premium Zone (溢價區)，追高風險較大。耐心等待價格回測下方支撐或 FVG。</li>
                    </ul>
                </div>
                """
            
            # 清理 HTML 字串以便傳遞給 JS
            deploy_clean = deployment_html.replace('"', '&quot;').replace('\n', '')

            # 卡片 HTML
            cards_in_sector += f"""
            <div class="card" onclick="openModal('{t}', '{img_d}', '{img_h}', '{signal}', '{deploy_clean}')">
                <div class="head">
                    <div><div class="code">{t}</div><div class="price">${current_price:.2f}</div></div>
                    <span class="badge {cls}">{signal}</span>
                </div>
                <div class="hint">Tap for Strategy ↗</div>
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
    .hint {{ font-size:0.75rem; color:var(--acc); text-align:right; margin-top:10px; opacity:0.8; }}
    
    table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:10px; overflow:hidden; }}
    th, td {{ padding:12px; text-align:left; border-bottom:1px solid #334155; }}
    th {{ background:#334155; color:#94a3b8; font-size:0.85rem; }}
    
    .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:999; justify-content:center; align-items:start; overflow-y:auto; padding:20px; }}
    .m-content {{ background:var(--card); width:100%; max-width:800px; padding:20px; border-radius:12px; border:1px solid #475569; margin-top:20px; }}
    
    /* Deployment Box Styles */
    .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:20px; border-left:4px solid; }}
    .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
    .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
    .deploy-title {{ font-weight:bold; margin-bottom:10px; font-size:1.1rem; color:#fff; }}
    .deploy-list {{ margin:0; padding-left:20px; color:#cbd5e1; font-size:0.95rem; line-height:1.6; }}
    .deploy-list li {{ margin-bottom:5px; }}
    
    .chart-box {{ margin-bottom:20px; }} .chart-title {{ color:var(--acc); font-weight:bold; display:block; margin-bottom:5px; }}
    .m-content img {{ width:100%; border-radius:8px; border:1px solid #334155; }}
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
        
        <div id="m-deploy"></div>
        
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
function openModal(ticker, d_src, h_src, signal, deploy_html) {{
    document.getElementById('modal').style.display = 'flex';
    document.getElementById('m-ticker').innerText = ticker + " (" + signal + ")";
    document.getElementById('m-deploy').innerHTML = deploy_html;
    document.getElementById('img-d').src = d_src;
    document.getElementById('img-h').src = h_src;
}}
</script>

</body>
</html>
"""

display(HTML(full_html))
