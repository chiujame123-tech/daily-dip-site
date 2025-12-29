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

# --- 1. 設定板塊與觀察清單 ---
SECTORS = {
    "💎 Mag 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO", "ORCL"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "LRCX", "MU", "ADI", "MRVL", "KLAC", "ON", "INTC"],
    "☁️ Software": ["PLTR", "CRM", "ADBE", "NOW", "SNOW", "PANW", "CRWD", "SQ", "SHOP", "NET"],
    "🚀 High Growth": ["COIN", "MSTR", "HOOD", "DKNG", "RBLX", "U", "TTD", "ZM", "DOCU"],
    "🏦 Finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP"],
    "💊 Healthcare": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "PFE", "ISRG"],
    "🛒 Consumer": ["WMT", "COST", "TGT", "HD", "MCD", "SBUX", "NKE", "KO", "PEP"],
    "🛢️ Industrial": ["XOM", "CVX", "SLB", "GE", "CAT", "DE", "BA"],
    "🎬 Entertainment": ["DIS", "NFLX", "CMCSA", "TMUS", "VZ", "UBER", "ABNB"]
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 篩選參數 ---
FILTER_SMA_PERIOD = 200
FILTER_MIN_MONTHLY_VOL = 900000000 
FILTER_MIN_BETA = 1.0

# --- 3. 核心繪圖 (已優化：小尺寸+低解析度，解決 Loading 問題) ---
def generate_chart_image(df, ticker, timeframe):
    try:
        window = 20
        if len(df) < window: return None, 0, 0, 0
        
        swing_high = df['High'].tail(window).max()
        swing_low = df['Low'].tail(window).min()
        eq = (swing_high + swing_low) / 2
        
        # 風格
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        hlines = dict(hlines=[swing_high, swing_low, eq], colors=['#ef4444', '#10b981', '#3b82f6'], linewidths=[1, 1, 0.5], linestyle=['--', '--', '-.'])

        # ⚡️ 優化重點：縮小尺寸 (figsize)
        fig, axlist = mpf.plot(df.tail(50), type='candle', style=s, volume=False,
            title=dict(title=f"{ticker}-{timeframe}", color='white', size=8),
            hlines=hlines, figsize=(3.5, 2.5), returnfig=True)
        
        # 簡易註解
        ax = axlist[0]
        x_pos = len(df.tail(50)) - 1
        ax.text(x_pos, swing_high, 'BSL', color='#ef4444', fontsize=6)
        ax.text(x_pos, swing_low, 'SSL', color='#10b981', fontsize=6)

        buf = BytesIO()
        # ⚡️ 優化重點：降低 DPI 至 50 (大幅減少檔案大小)
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=50)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low, eq
    except:
        return None, 0, 0, 0

# --- 4. 主程式 ---
print(f"🚀 正在下載數據 (包含圖片壓縮優化)...")
data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=True)
data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=True)
spy_ret = data_daily['SPY']['Close'].pct_change()

print("\n🔍 正在生成 AI 部署建議與圖表...")

sector_html_blocks = ""
screener_rows = ""
passed_count = 0

for sector, tickers in SECTORS.items():
    cards_in_sector = ""
    # 每個板塊只顯示前 12 檔，避免過載
    for t in tickers[:12]:
        try:
            df_d = data_daily[t].dropna()
            df_h = data_hourly[t].dropna()
            if len(df_d) < 200: continue
            
            curr_price = df_d['Close'].iloc[-1]
            
            # --- 篩選 ---
            sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
            vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
            
            stock_ret = df_d['Close'].pct_change()
            combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
            beta = 0
            if len(combo) > 30:
                beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1]
            
            pass_filter = (curr_price > sma200 and vol > FILTER_MIN_MONTHLY_VOL and beta >= FILTER_MIN_BETA)

            # --- 繪圖與數據 ---
            img_d, tp, sl, eq = generate_chart_image(df_d, t, "D1")
            if not img_d: continue
            img_h, _, _, _ = generate_chart_image(df_h if not df_h.empty else df_d, t, "H1")
            
            # SMC 訊號
            range_len = tp - sl
            pos_pct = (curr_price - sl) / range_len if range_len > 0 else 0.5
            signal = "LONG" if pos_pct < 0.4 else "WAIT"
            cls = "b-long" if signal == "LONG" else "b-wait"
            
            # --- 🔥 新增：AI 部署建議邏輯 (Deployment Logic) ---
            deployment_html = ""
            trend_str = "上升趨勢 (Above 200MA)" if curr_price > sma200 else "震盪/回調中"
            
            if signal == "LONG":
                # 做多情境
                entry_zone_top = sl + (range_len * 0.4)
                rr = (tp - curr_price) / (curr_price - sl*0.98) if (curr_price - sl*0.98) > 0 else 0
                
                deployment_html = f"""
                <div class="deploy-box long">
                    <div class="deploy-title">✅ 建議部署：現價買入 / 分批建倉</div>
                    <ul class="deploy-list">
                        <li><b>入手價位：</b> ${curr_price:.2f} (目前處於折價區)</li>
                        <li><b>止損位置：</b> ${sl*0.98:.2f} (前低下方緩衝)</li>
                        <li><b>獲利目標：</b> ${tp:.2f} (上方流動性 BSL)</li>
                        <li><b>操作理由：</b> 股價回落至 Discount Zone (<40%)，且維持{trend_str}，盈虧比 {rr:.1f}R 具吸引力。</li>
                    </ul>
                </div>
                """
            else:
                # 觀望情境
                buy_target = eq  # 建議在平衡點 (50%) 接回
                discount_entry = sl + (range_len * 0.4) # 或等到進入折價區
                
                deployment_html = f"""
                <div class="deploy-box wait">
                    <div class="deploy-title">⏳ 建議部署：等待回調 (Do Not Chase)</div>
                    <ul class="deploy-list">
                        <li><b>觀察價位：</b> 等待回落至 <b>${buy_target:.2f}</b> (Equilibrium) 或更低。</li>
                        <li><b>入手價位：</b> 理想買點在 <b>${discount_entry:.2f}</b> 以下。</li>
                        <li><b>操作理由：</b> 目前股價處於溢價區 (Premium)，追高風險大。需等待價格回測平衡點或下方支撐再進場。</li>
                    </ul>
                </div>
                """

            # --- 組合卡片 HTML ---
            # 為了彈窗傳遞 HTML，需要做一點跳脫處理
            deploy_clean = deployment_html.replace('"', '&quot;').replace('\n', '')

            cards_in_sector += f"""
            <div class="card" onclick="openModal('{t}', '{img_d}', '{img_h}', '{signal}', '{deploy_clean}')">
                <div class="head">
                    <div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                    <span class="badge {cls}">{signal}</span>
                </div>
                <div class="hint">Tap for Strategy ↗</div>
            </div>
            """
            
            if pass_filter:
                passed_count += 1
                screener_rows += f"""
                <tr><td><b>{t}</b></td><td>${curr_price:.2f}</td><td class="g">✔</td><td>{beta:.2f}</td><td><span class="badge {cls}">{signal}</span></td></tr>
                """
        except: continue
            
    if cards_in_sector:
        sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

print(f"\n✅ 分析完成！共 {passed_count} 檔精選股票。網頁生成中...")

# --- 5. 網頁生成 ---
full_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
    body {{ background:var(--bg); color:var(--text); font-family:-apple-system, sans-serif; margin:0; padding:10px; }}
    
    /* Tabs */
    .tabs {{ display:flex; gap:10px; border-bottom:1px solid #334155; padding-bottom:10px; margin-bottom:15px; position:sticky; top:0; background:var(--bg); z-index:10; }}
    .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; color:#94a3b8; font-weight:bold; font-size:0.9rem; }}
    .tab.active {{ background:var(--acc); color:white; }}
    .content {{ display:none; }} .content.active {{ display:block; }}

    /* Layout */
    .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; color:#e2e8f0; font-size:1.1rem; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap:10px; }}
    
    /* Card */
    .card {{ background:var(--card); border:1px solid #334155; border-radius:8px; padding:12px; cursor:pointer; transition:0.2s; }}
    .card:hover {{ border-color:var(--acc); transform:translateY(-2px); }}
    .head {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:5px; }}
    .code {{ font-size:1.2rem; font-weight:900; }}
    .price {{ color:#94a3b8; font-family:monospace; }}
    .badge {{ padding:3px 6px; border-radius:4px; font-size:0.75rem; font-weight:bold; }}
    .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
    .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #334155; }}
    .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
    
    /* Screener Table */
    table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; font-size:0.9rem; }}
    th, td {{ padding:10px; text-align:left; border-bottom:1px solid #334155; }}
    th {{ background:#334155; color:#94a3b8; }} .g {{ color:var(--g); }}

    /* Modal & Deployment Box */
    .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:999; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
    .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; border:1px solid #475569; margin-top:10px; }}
    
    .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
    .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
    .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
    .deploy-title {{ font-weight:bold; margin-bottom:10px; font-size:1rem; color:#fff; }}
    .deploy-list {{ margin:0; padding-left:20px; color:#cbd5e1; font-size:0.9rem; line-height:1.6; }}
    .deploy-list li {{ margin-bottom:5px; }}
    
    .chart-box {{ margin-bottom:15px; }}
    .chart-lbl {{ color:var(--acc); font-size:0.85rem; font-weight:bold; display:block; margin-bottom:5px; }}
    .m-content img {{ width:100%; border-radius:6px; border:1px solid #334155; }}
    .close-btn {{ width:100%; padding:12px; background:var(--acc); color:white; border:none; border-radius:6px; font-weight:bold; font-size:1rem; cursor:pointer; }}
</style>
</head>
<body>

<div class="tabs">
    <div class="tab active" onclick="setTab('overview', this)">📊 Market</div>
    <div class="tab" onclick="setTab('screener', this)">🔍 Screener ({passed_count})</div>
</div>

<div id="overview" class="content active">
    {sector_html_blocks}
</div>

<div id="screener" class="content">
    <table>
        <thead><tr><th>Ticker</th><th>Price</th><th>Trend</th><th>Beta</th><th>Signal</th></tr></thead>
        <tbody>{screener_rows}</tbody>
    </table>
</div>

<div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
    <div class="m-content" onclick="event.stopPropagation()">
        <h2 id="m-ticker" style="margin-top:0; color:white"></h2>
        
        <div id="m-deploy"></div>
        
        <div class="chart-box">
            <span class="chart-lbl">📅 Daily Chart</span>
            <img id="img-d" src="">
        </div>
        <div class="chart-box">
            <span class="chart-lbl">⏱️ Hourly Chart</span>
            <img id="img-h" src="">
        </div>
        
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
