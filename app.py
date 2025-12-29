import os
import sys
import threading
import time
import base64
from io import BytesIO
from datetime import datetime

# --- Web Server 框架 ---
from flask import Flask, render_template_string

# --- 數據分析套件 ---
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 設定後端，避免在無螢幕環境報錯
import matplotlib.pyplot as plt

app = Flask(__name__)

# 全域變數用來儲存最新的 HTML 內容
LATEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="10"> <style>
        body { background: #0f172a; color: white; font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .loader { text-align: center; }
        .spinner { border: 4px solid #334155; border-top: 4px solid #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="loader">
        <h1>🚀 System Initializing...</h1>
        <p>AI Analyst is scanning 100+ stocks.</p>
        <p>This may take 2-3 minutes. Please wait...</p>
        <div class="spinner"></div>
    </div>
</body>
</html>
"""

# --- 設定參數 ---
# 移除了 SQ 以避免報錯，您也可以加回去
SECTORS = {
    "💎 Mag 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO", "ORCL"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "LRCX", "MU", "ADI", "MRVL", "KLAC", "ON", "INTC"],
    "☁️ Software": ["PLTR", "CRM", "ADBE", "NOW", "SNOW", "PANW", "CRWD", "SHOP", "NET"],
    "🚀 High Growth": ["COIN", "MSTR", "HOOD", "DKNG", "RBLX", "U", "TTD", "ZM", "DOCU"],
    "🏦 Finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP"],
    "💊 Healthcare": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "PFE", "ISRG"],
    "🛒 Consumer": ["WMT", "COST", "TGT", "HD", "MCD", "SBUX", "NKE", "KO", "PEP"],
    "🛢️ Industrial": ["XOM", "CVX", "SLB", "GE", "CAT", "DE", "BA"],
    "🎬 Entertainment": ["DIS", "NFLX", "CMCSA", "TMUS", "VZ", "UBER", "ABNB"]
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

FILTER_SMA_PERIOD = 200
FILTER_MIN_MONTHLY_VOL = 900000000 
FILTER_MIN_BETA = 1.0

# --- 核心繪圖函式 ---
def generate_chart_image(df, ticker, timeframe):
    try:
        window = 20
        if len(df) < window: return None, 0, 0, 0
        
        swing_high = df['High'].tail(window).max()
        swing_low = df['Low'].tail(window).min()
        eq = (swing_high + swing_low) / 2
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        hlines = dict(hlines=[swing_high, swing_low, eq], colors=['#ef4444', '#10b981', '#3b82f6'], linewidths=[1, 1, 0.5], linestyle=['--', '--', '-.'])

        fig, axlist = mpf.plot(df.tail(50), type='candle', style=s, volume=False,
            title=dict(title=f"{ticker}-{timeframe}", color='white', size=8),
            hlines=hlines, figsize=(3.5, 2.5), returnfig=True)
        
        ax = axlist[0]
        x_pos = len(df.tail(50)) - 1
        ax.text(x_pos, swing_high, 'BSL', color='#ef4444', fontsize=6)
        ax.text(x_pos, swing_low, 'SSL', color='#10b981', fontsize=6)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=50)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low, eq
    except:
        return None, 0, 0, 0

# --- 背景任務：執行分析 ---
def run_analysis_loop():
    global LATEST_HTML
    print("Background Worker Started...")
    
    while True:
        try:
            print(f"[{datetime.now()}] Starting Analysis Cycle...")
            
            # 1. 下載數據
            data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=False)
            data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
            
            # 處理大盤
            if isinstance(data_daily.columns, pd.MultiIndex):
                spy_close = data_daily['SPY']['Close']
            else:
                spy_close = data_daily['Close'] # Fallback
            spy_ret = spy_close.pct_change()

            sector_html_blocks = ""
            screener_rows = ""
            passed_count = 0

            for sector, tickers in SECTORS.items():
                cards_in_sector = ""
                # 限制每個板塊處理數量以加快速度
                for t in tickers[:12]: 
                    try:
                        # 處理 MultiIndex
                        if isinstance(data_daily.columns, pd.MultiIndex):
                            try:
                                df_d = data_daily[t].dropna()
                                df_h = data_hourly[t].dropna()
                            except KeyError:
                                continue # 跳過下載失敗的股票
                        else:
                            continue

                        if len(df_d) < 200: continue
                        
                        curr_price = df_d['Close'].iloc[-1]
                        if isinstance(curr_price, pd.Series): curr_price = curr_price.iloc[0]

                        # 篩選指標
                        sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                        if isinstance(sma200, pd.Series): sma200 = sma200.iloc[0]
                        
                        vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
                        if isinstance(vol, pd.Series): vol = vol.iloc[0]

                        stock_ret = df_d['Close'].pct_change()
                        combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
                        beta = 0
                        if len(combo) > 30:
                            beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1]

                        pass_filter = (curr_price > sma200 and vol > FILTER_MIN_MONTHLY_VOL and beta >= FILTER_MIN_BETA)

                        # 繪圖
                        img_d, tp, sl, eq = generate_chart_image(df_d, t, "D1")
                        if not img_d: continue
                        img_h, _, _, _ = generate_chart_image(df_h if not df_h.empty else df_d, t, "H1")

                        # 訊號
                        range_len = tp - sl
                        pos_pct = (curr_price - sl) / range_len if range_len > 0 else 0.5
                        signal = "LONG" if pos_pct < 0.4 else "WAIT"
                        cls = "b-long" if signal == "LONG" else "b-wait"

                        # AI 部署建議
                        deployment_html = ""
                        trend_str = "Above 200MA" if curr_price > sma200 else "Recovering"

                        if signal == "LONG":
                            entry = curr_price
                            stop_loss = sl * 0.98
                            take_profit = tp
                            rr = (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
                            
                            deployment_html = f"""
                            <div class="deploy-box long">
                                <div class="deploy-title">✅ LONG SETUP</div>
                                <ul class="deploy-list">
                                    <li><b>Entry:</b> ${entry:.2f} (Discount Zone)</li>
                                    <li><b>Stop Loss:</b> ${stop_loss:.2f}</li>
                                    <li><b>Target:</b> ${take_profit:.2f}</li>
                                    <li><b>R:R:</b> {rr:.1f}R ({trend_str})</li>
                                </ul>
                            </div>"""
                        else:
                            buy_target = eq
                            deployment_html = f"""
                            <div class="deploy-box wait">
                                <div class="deploy-title">⏳ WAIT</div>
                                <ul class="deploy-list">
                                    <li>Price is in Premium zone.</li>
                                    <li>Wait for pullback to <b>${buy_target:.2f}</b>.</li>
                                    <li>Do not chase highs.</li>
                                </ul>
                            </div>"""

                        deploy_clean = deployment_html.replace('"', '&quot;').replace('\n', '')

                        cards_in_sector += f"""
                        <div class="card" onclick="openModal('{t}', '{img_d}', '{img_h}', '{signal}', '{deploy_clean}')">
                            <div class="head">
                                <div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                                <span class="badge {cls}">{signal}</span>
                            </div>
                            <div class="hint">Tap for Strategy ↗</div>
                        </div>"""

                        if pass_filter:
                            passed_count += 1
                            screener_rows += f"<tr><td><b>{t}</b></td><td>${curr_price:.2f}</td><td class='g'>✔</td><td>{beta:.2f}</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

                    except Exception as e:
                        print(f"Error processing {t}: {e}")
                        continue

                if cards_in_sector:
                    sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

            # 生成最終 HTML
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            LATEST_HTML = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>DailyDip Pro</title>
            <style>
                :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
                body {{ background:var(--bg); color:var(--text); font-family:-apple-system, sans-serif; margin:0; padding:10px; }}
                .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; position:sticky; top:0; background:var(--bg); z-index:10; border-bottom:1px solid #334155; }}
                .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; color:#94a3b8; font-weight:bold; font-size:0.9rem; }}
                .tab.active {{ background:var(--acc); color:white; }}
                .content {{ display:none; }} .content.active {{ display:block; }}
                .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; color:#e2e8f0; font-size:1.1rem; }}
                .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; }}
                .card {{ background:var(--card); border:1px solid #334155; border-radius:8px; padding:12px; cursor:pointer; }}
                .head {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:5px; }}
                .code {{ font-size:1.1rem; font-weight:900; }}
                .price {{ color:#94a3b8; font-family:monospace; }}
                .badge {{ padding:3px 6px; border-radius:4px; font-size:0.75rem; font-weight:bold; }}
                .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
                .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #334155; }}
                .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
                table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; font-size:0.9rem; }}
                th, td {{ padding:10px; text-align:left; border-bottom:1px solid #334155; }}
                th {{ background:#334155; color:#94a3b8; }} .g {{ color:var(--g); }}
                .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:999; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
                .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; border:1px solid #475569; margin-top:10px; }}
                .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
                .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
                .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
                .chart-lbl {{ color:var(--acc); font-size:0.85rem; font-weight:bold; display:block; margin-bottom:5px; }}
                .m-content img {{ width:100%; border-radius:6px; border:1px solid #334155; margin-bottom:15px; }}
                .close-btn {{ width:100%; padding:12px; background:var(--acc); color:white; border:none; border-radius:6px; font-weight:bold; font-size:1rem; cursor:pointer; }}
                .update-time {{ font-size: 0.7rem; color: #64748b; text-align: center; margin-top: 20px; }}
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
            
            <div class="update-time">Last Updated: {timestamp}</div>

            <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
                <div class="m-content" onclick="event.stopPropagation()">
                    <h2 id="m-ticker" style="margin-top:0; color:white"></h2>
                    <div id="m-deploy"></div>
                    <div class="chart-box"><span class="chart-lbl">📅 Daily Chart</span><img id="img-d" src=""></div>
                    <div class="chart-box"><span class="chart-lbl">⏱️ Hourly Chart</span><img id="img-h" src=""></div>
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
            
            print(f"[{timestamp}] Analysis Complete. Updated HTML.")
            
            # 等待 1 小時後再更新 (避免 API 限制)
            time.sleep(3600)

        except Exception as e:
            print(f"Global Loop Error: {e}")
            time.sleep(60)

# --- 啟動背景執行緒 ---
analysis_thread = threading.Thread(target=run_analysis_loop, daemon=True)
analysis_thread.start()

# --- Flask 路由 ---
@app.route('/')
def home():
    return render_template_string(LATEST_HTML)

# --- 啟動 Web Server ---
if __name__ == '__main__':
    # 這是關鍵：讀取環境變數中的 PORT，如果沒有則使用 10000 (Render 預設)
    port = int(os.environ.get("PORT", 10000))
    # host='0.0.0.0' 代表允許外部訪問
    app.run(host='0.0.0.0', port=port)
