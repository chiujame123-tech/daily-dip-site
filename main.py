import os
import requests
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
import json
import time
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# --- 全域變數 ---
API_KEY = os.environ.get("POLYGON_API_KEY")
HTML_CONTENT = "" # 用來確保最後一定有東西寫入

# --- 1. 設定觀察清單 ---
SECTORS = {
    "💎 科技七巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI"],
    "☁️ 軟體與SaaS": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PANW", "CRWD", "SHOP"],
    "🏦 金融與消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT"],
}

# --- 2. 獲取成交量前 100 ---
def get_top_volume_tickers(limit=100):
    if not API_KEY: return []
    print("🔍 Scanning Top Volume...")
    for i in range(1, 5): # 回推找最近交易日
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{target_date}?adjusted=true&apiKey={API_KEY}"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if data.get('status') == 'OK' and data.get('resultsCount', 0) > 0:
                print(f"✅ Found data for {target_date}")
                df = pd.DataFrame(data['results'])
                df = df[df['c'] > 5] # 過濾股價 > 5
                df = df.sort_values(by='v', ascending=False)
                return df['T'].head(limit).tolist()
        except: continue
    return []

# --- 3. 獲取個股數據 ---
def get_polygon_data(ticker, multiplier=1, timespan='day'):
    if not API_KEY: return None
    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
        
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429: time.sleep(1); resp = requests.get(url, timeout=10)
        
        data = resp.json()
        if data.get('results'):
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except: pass
    return None

def get_polygon_news():
    if not API_KEY: return "<div>API Key Missing</div>"
    news_html = ""
    try:
        url = f"https://api.polygon.io/v2/reference/news?limit=15&order=desc&sort=published_utc&apiKey={API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('results'):
            for item in data['results']:
                title = item.get('title')
                url = item.get('article_url')
                pub = item.get('publisher', {}).get('name', 'Unknown')
                news_html += f"<div class='news-item'><div class='news-meta'>{pub}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else: news_html = "<div style='padding:20px'>暫無新聞</div>"
    except: news_html = "News Error"
    return news_html

# --- 4. SMC ---
def calculate_smc(df):
    try:
        recent = df.tail(50)
        bsl, ssl = float(recent['High'].max()), float(recent['Low'].min())
        eq = (bsl + ssl) / 2
        best_entry, found_fvg = eq, False
        for i in range(len(recent)-1, 2, -1):
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
                fvg = float(recent['Low'].iloc[i])
                if fvg < eq: best_entry, found_fvg = fvg, True; break
        return bsl, ssl, eq, best_entry, ssl*0.99, found_fvg
    except: return 0,0,0,0,0,False

def generate_chart(df, ticker, title, entry, sl, tp, is_wait):
    try:
        plot_df = df.tail(60)
        if len(plot_df)<10: return None
        swing_high, swing_low = plot_df['High'].max(), plot_df['Low'].min()
        eq = (swing_high+swing_low)/2
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        line_style, line_alpha = (':', 0.3) if is_wait else ('--', 0.9)
        hlines = dict(hlines=[tp, entry, sl], colors=['#10b981', '#3b82f6', '#ef4444'], linewidths=[1,1,1], linestyle=['-', line_style, '-'], alpha=line_alpha)
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False, title=dict(title=f"{ticker}-{title}", color='white', size=10), hlines=hlines, figsize=(5, 3), returnfig=True)
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        ax.text(x_min, tp, f" TP ${tp:.2f}", color='#10b981', fontsize=8, va='bottom')
        ax.text(x_min, entry, f" REF ${entry:.2f}", color='#3b82f6', fontsize=8, va='bottom')
        
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.05)
        ax.add_patch(rect_prem)
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return None

# --- 5. 處理單一股票 ---
def process_ticker(t, app_data_dict):
    try:
        time.sleep(0.1)
        df_d = get_polygon_data(t, 1, 'day')
        if df_d is None or len(df_d)<50: return None
        df_h = get_polygon_data(t, 1, 'hour')
        if df_h is None: df_h = df_d
        
        curr_price = df_d['Close'].iloc[-1]
        sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
        if pd.isna(sma200): sma200 = curr_price
        
        bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
        tp = bsl
        
        is_bullish, in_discount = (curr_price > sma200), (curr_price < eq)
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "D1", entry, sl, tp, is_wait)
        img_h = generate_chart(df_h, t, "H1", entry, sl, tp, is_wait)
        
        # AI 文案
        rr = (tp-entry)/(entry-sl) if (entry-sl)>0 else 0
        if signal == "LONG":
            ai_html = f"<div class='deploy-box long'><div class='deploy-title'>✅ LONG SETUP</div><ul class='deploy-list'><li>Entry: ${entry:.2f}</li><li>SL: ${sl:.2f}</li><li>TP: ${tp:.2f}</li><li>RR: {rr:.1f}R</li></ul><div style='margin-top:5px;font-size:0.8rem'>AI: 趨勢向上+折價區+FVG確認。</div></div>"
        else:
            reason = "無FVG" if not found_fvg else ("逆勢" if not is_bullish else "溢價區")
            ai_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><ul class='deploy-list'><li>狀態: {reason}</li><li>參考入場: ${entry:.2f}</li></ul></div>"
            
        app_data_dict[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}
        return {"ticker": t, "price": curr_price, "signal": signal, "cls": "b-long" if signal=="LONG" else "b-wait"}
    except: return None

# --- 6. 主程式 ---
def main():
    print("🚀 Starting Volume Scanner...")
    
    # 防呆：如果沒有 Key，產生一個錯誤頁面，但不要 Crash
    if not API_KEY:
        final_html = "<html><body><h1>Error: API Key Missing</h1><p>Please check GitHub Secrets.</p></body></html>"
        with open("index.html", "w", encoding="utf-8") as f: f.write(final_html)
        return

    weekly_news_html = get_polygon_news()
    top_100_tickers = get_top_volume_tickers(limit=100)
    
    APP_DATA, sector_html_blocks, screener_rows = {}, "", ""
    
    # 板塊
    for sector, tickers in SECTORS.items():
        cards = ""
        for t in tickers:
            res = process_ticker(t, APP_DATA)
            if res: cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><span class='badge {res['cls']}'>{res['signal']}</span></div><div class='hint'>Tap for Analysis ↗</div></div>"
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    # Top 100
    processed = set([t for sec in SECTORS.values() for t in sec])
    for t in top_100_tickers:
        if t in processed: continue
        res = process_ticker(t, APP_DATA)
        if res and res['signal'] == "LONG":
            screener_rows += f"<tr><td>{t}</td><td>${res['price']:.2f}</td><td class='g'>🔥 Vol</td><td><span class='badge {res['cls']}'>{res['signal']}</span></td></tr>"

    json_data = json.dumps(APP_DATA)
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DailyDip Pro</title>
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
    .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:6px; font-weight:bold; margin-top:10px; cursor:pointer; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 Market</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 Screener (Top 100)</div>
            <div class="tab" onclick="setTab('news', this)">📰 News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content">
            <div style="padding:10px;background:rgba(16,185,129,0.1);margin-bottom:15px;border-radius:6px;font-size:0.9rem">
            🎯 <b>SMC Screener:</b> Filtering top 100 volume stocks for Perfect LONG setups.
            </div>
            <table><thead><tr><th>Ticker</th><th>Price</th><th>Tag</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table>
        </div>
        <div id="news" class="content">{weekly_news_html}</div>
        
        <div style="text-align:center;color:#666;margin-top:30px;font-size:0.7rem">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><b>Daily Structure</b><div id="chart-d"></div></div>
                <div><b>Hourly Execution</b><div id="chart-h"></div></div>
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
    </body></html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated!")

if __name__ == "__main__":
    main()
