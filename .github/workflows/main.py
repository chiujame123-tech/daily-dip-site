import os
import requests
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
import json
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# --- 全域錯誤收集 ---
ERROR_LOG = []

def log_error(msg):
    print(f"❌ {msg}")
    ERROR_LOG.append(msg)

# --- 0. 讀取 API KEY ---
API_KEY = os.environ.get("POLYGON_API_KEY")

# --- 1. 設定觀察清單 ---
SECTORS = {
    "💎 科技巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM"],
    "☁️ 軟體與SaaS": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PANW"],
    "🏦 金融": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 數據獲取 ---
def get_polygon_data(ticker, multiplier=1, timespan='day', limit=100):
    if not API_KEY: return None
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('status') != 'OK' or not data.get('results'): return None
        df = pd.DataFrame(data['results'])
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        log_error(f"Data Error {ticker}: {e}")
        return None

def get_weekly_hot_news():
    if not API_KEY: return "<div style='padding:20px'>API Key Missing</div>"
    news_html = ""
    try:
        last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/reference/news?ticker=SPY,QQQ,NVDA,TSLA,AAPL&published_utc.gte={last_week}&limit=10&sort=published_utc&order=desc&apiKey={API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('results'):
            for item in data['results']:
                title = item.get('title')
                url = item.get('article_url')
                pub = item.get('publisher', {}).get('name', 'Unknown')
                news_html += f"<div class='news-item'><div class='news-meta'>{pub}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else:
            news_html = "<div style='padding:20px'>無相關新聞</div>"
    except Exception as e:
        log_error(f"News Error: {e}")
        news_html = f"News Error: {e}"
    return news_html

# --- 3. SMC & 繪圖 ---
def calculate_smc_levels(df):
    window = 50
    recent = df.tail(window)
    bsl = recent['High'].max()
    ssl = recent['Low'].min()
    eq = (bsl + ssl) / 2
    best_entry = eq
    for i in range(len(recent)-1, 2, -1):
        if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
            if recent['Low'].iloc[i] < eq:
                best_entry = recent['Low'].iloc[i]
                break
    return bsl, ssl, eq, best_entry, ssl * 0.99

def generate_chart_image(df, ticker, timeframe, entry, sl, tp):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 30: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        hlines = dict(hlines=[tp, entry, sl], colors=['#10b981', '#3b82f6', '#ef4444'], linewidths=[1,1,1], linestyle=['-','--','-'])
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False, title=dict(title=f"{ticker}-{timeframe}", color='white', size=10), hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        ax.text(x_min, tp, f" TP ${tp:.2f}", color='#10b981', fontsize=7, va='bottom')
        ax.text(x_min, entry, f" ENTRY ${entry:.2f}", color='#3b82f6', fontsize=7, va='bottom')
        ax.text(x_min, sl, f" SL ${sl:.2f}", color='#ef4444', fontsize=7, va='top')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low
    except Exception as e:
        log_error(f"Plot Error {ticker}: {e}")
        return None

# --- 4. 主程式 ---
def main():
    print("🚀 Starting Analysis...")
    
    # 防呆機制：如果沒有 Key，顯示錯誤頁面
    if not API_KEY:
        log_error("Missing POLYGON_API_KEY in GitHub Secrets")
    
    weekly_news_html = get_weekly_hot_news() if API_KEY else "API Key Missing"
    
    sector_html_blocks = ""
    screener_rows = ""
    APP_DATA = {}
    passed_count = 0

    if API_KEY:
        for sector, tickers in SECTORS.items():
            cards_in_sector = ""
            for t in tickers:
                try:
                    df_d = get_polygon_data(t, 1, 'day')
                    if df_d is None or len(df_d) < 50: continue
                    
                    df_h = get_polygon_data(t, 1, 'hour')
                    if df_h is None: df_h = df_d

                    curr_price = df_d['Close'].iloc[-1]
                    sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                    if pd.isna(sma200): sma200 = curr_price

                    bsl, ssl, eq, entry, sl = calculate_smc_levels(df_d)
                    tp = bsl

                    img_d = generate_chart_image(df_d, t, "Daily", entry, sl, tp)
                    if not img_d: continue
                    img_d_src = img_d[0]
                    
                    img_h = generate_chart_image(df_h, t, "Hourly", entry, sl, tp)
                    img_h_src = img_h[0] if img_h else ""

                    is_bullish = curr_price > sma200
                    signal = "LONG" if is_bullish and curr_price < eq else "WAIT"
                    cls = "b-long" if signal == "LONG" else "b-wait"
                    
                    risk = entry - sl
                    reward = tp - entry
                    rr = reward / risk if risk > 0 else 0
                    
                    # AI Text
                    if signal == "LONG":
                        ai_html = f"<div class='deploy-box long'><div class='deploy-title'>✅ LONG SETUP</div><ul class='deploy-list'><li><b>Entry:</b> ${entry:.2f}</li><li><b>SL:</b> ${sl:.2f}</li><li><b>TP:</b> ${tp:.2f}</li><li><b>RR:</b> {rr:.1f}R</li></ul></div>"
                    else:
                        ai_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><ul class='deploy-list'><li>Price in Premium.</li><li>Wait for: ${entry:.2f}</li></ul></div>"

                    APP_DATA[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d_src, "img_h": img_h_src}

                    cards_in_sector += f"""
                    <div class="card" onclick="openModal('{t}')">
                        <div class="head"><div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div><span class="badge {cls}">{signal}</span></div>
                        <div class="hint">Tap for Analysis ↗</div>
                    </div>"""
                    
                    if is_bullish:
                        passed_count += 1
                        screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>Bullish</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

                except Exception as e:
                    log_error(f"Process Error {t}: {e}")
                    continue
            
            if cards_in_sector:
                sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

    json_data = json.dumps(APP_DATA)
    error_display = "<br>".join(ERROR_LOG) if ERROR_LOG else ""

    # 生成 HTML (無論是否有錯誤，都會生成這個檔案，避免 404)
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DailyDip Pro</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
        body {{ background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding:10px; }}
        .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; border-bottom:1px solid #333; overflow-x:auto; }}
        .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; font-weight:bold; font-size:0.9rem; white-space:nowrap; }}
        .tab.active {{ background:var(--acc); color:white; }}
        .content {{ display:none; }} .content.active {{ display:block; }}
        .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
        .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:10px; cursor:pointer; }}
        .head {{ display:flex; justify-content:space-between; margin-bottom:5px; }}
        .code {{ font-weight:900; }} .price {{ color:#94a3b8; font-family:monospace; }}
        .badge {{ padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; }}
        .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
        .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #555; }}
        .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
        th, td {{ padding:8px; text-align:left; border-bottom:1px solid #333; }}
        .g {{ color:var(--g); }}
        .news-item {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:15px; margin-bottom:10px; }}
        .news-title {{ color:var(--text); text-decoration:none; font-weight:bold; display:block; }}
        .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:99; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
        .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; margin-top:20px; border:1px solid #555; }}
        .m-content img {{ width:100%; border-radius:6px; margin-bottom:10px; }}
        .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
        .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
        .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
        .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:6px; font-weight:bold; margin-top:10px; }}
        .error-log {{ background:rgba(239,68,68,0.1); border:1px solid #ef4444; color:#ef4444; padding:10px; margin-bottom:20px; border-radius:6px; font-size:0.8rem; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 Market</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 Screener</div>
            <div class="tab" onclick="setTab('news', this)">📰 News</div>
        </div>
        
        {f'<div class="error-log"><b>System Warnings:</b><br>{error_display}</div>' if error_display else ''}

        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Trend</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table></div>
        <div id="news" class="content"><h3 class="sector-title">Weekly Hot News</h3>{weekly_news_html}</div>
        
        <div class="time" style="text-align:center; color:#666; margin-top:30px; font-size:0.7rem;">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><b>Daily Structure</b><div id="chart-d"></div></div>
                <div style="margin-top:15px;"><b>Hourly Entry</b><div id="chart-h"></div></div>
                <button class="close-btn" onclick="document.getElementById('modal').style.display='none'">Close</button>
            </div>
        </div>

        <script>
        const STOCK_DATA = {json_data};
        function setTab(id, el) {{
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            el.classList.add('active');
        }}
        function openModal(ticker) {{
            const data = STOCK_DATA[ticker];
            if (!data) return;
            document.getElementById('modal').style.display = 'flex';
            document.getElementById('m-ticker').innerText = ticker;
            document.getElementById('m-deploy').innerHTML = data.deploy;
            document.getElementById('chart-d').innerHTML = data.img_d ? '<img src="'+data.img_d+'">' : 'No Data';
            document.getElementById('chart-h').innerHTML = data.img_h ? '<img src="'+data.img_h+'">' : 'No Data';
        }}
        </script>
    </body>
    </html>
    """
    
    # 無論如何都會寫入檔案，確保不會 404
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated successfully!")

if __name__ == "__main__":
    main()
