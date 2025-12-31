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

# --- 0. 讀取 API KEY ---
API_KEY = os.environ.get("POLYGON_API_KEY")

# --- 1. 固定觀察清單 (板塊概覽用) ---
SECTORS = {
    "💎 科技七巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI"],
    "☁️ 軟體與SaaS": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PANW", "CRWD", "SHOP"],
    "🏦 金融與消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT"],
}

# --- 2. 核心功能：獲取全市場成交量前 100 名 ---
def get_top_volume_tickers(limit=100):
    if not API_KEY: return []
    print("🔍 Scanning Market for Top Volume...")
    
    # 嘗試回推最近 3 天，找到有數據的交易日 (避開週末)
    for i in range(1, 5):
        target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{target_date}?adjusted=true&apiKey={API_KEY}"
        
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            
            if data.get('status') == 'OK' and data.get('resultsCount', 0) > 0:
                print(f"✅ Found data for {target_date}. Processing...")
                results = data['results']
                
                # 轉換為 DataFrame 方便排序
                df = pd.DataFrame(results)
                
                # 簡單過濾：
                # 1. 價格 > $5 (過濾垃圾股)
                # 2. 成交量排序 (由大到小)
                df = df[df['c'] > 5] 
                df = df.sort_values(by='v', ascending=False)
                
                # 取前 N 名的代號
                top_tickers = df['T'].head(limit).tolist()
                print(f"🔥 Top 5 Volume: {top_tickers[:5]}")
                return top_tickers
                
        except Exception as e:
            print(f"⚠️ Error scanning {target_date}: {e}")
            continue
            
    print("❌ Failed to find market data in last 4 days.")
    return []

# --- 3. Polygon 個股數據請求 ---
def get_polygon_data(ticker, multiplier=1, timespan='day'):
    if not API_KEY: return None
    try:
        # 抓取昨天以前的數據
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
        
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429:
            time.sleep(1)
            resp = requests.get(url, timeout=10)

        data = resp.json()
        if data.get('status') == 'OK' and data.get('results'):
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            return None
    except:
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
                pub_time = item.get('published_utc', '')
                try:
                    dt = datetime.strptime(pub_time, "%Y-%m-%dT%H:%M:%SZ")
                    date_str = dt.strftime('%m/%d %H:%M')
                except: date_str = ""
                news_html += f"<div class='news-item'><div class='news-meta'>{pub} • {date_str}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else:
            news_html = "<div style='padding:20px'>暫無新聞</div>"
    except:
        news_html = "<div style='padding:20px'>新聞載入失敗</div>"
    return news_html

# --- 4. SMC 分析邏輯 ---
def calculate_smc(df):
    try:
        window = 50
        recent = df.tail(window)
        bsl = float(recent['High'].max())
        ssl = float(recent['Low'].min())
        eq = (bsl + ssl) / 2
        best_entry = eq # 預設入場點
        found_fvg = False
        
        for i in range(len(recent)-1, 2, -1):
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]: # Bullish FVG
                fvg_top = float(recent['Low'].iloc[i])
                if fvg_top < eq:
                    best_entry = fvg_top
                    found_fvg = True
                    break
        sl_price = ssl * 0.99
        return bsl, ssl, eq, best_entry, sl_price, found_fvg
    except:
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False

def generate_chart(df, ticker, title, entry, sl, tp, is_wait):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 10: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2

        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        line_alpha = 0.3 if is_wait else 0.9
        line_style = ':' if is_wait else '--'
        hlines = dict(hlines=[tp, entry, sl], colors=['#10b981', '#3b82f6', '#ef4444'], linewidths=[1, 1, 1], linestyle=['-', line_style, '-'], alpha=line_alpha)
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False, title=dict(title=f"{ticker} - {title}", color='white', size=10), hlines=hlines, figsize=(5, 3), returnfig=True)
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        ax.text(x_min, tp, f" TP ${tp:.2f}", color='#10b981', fontsize=8, va='bottom', alpha=0.8)
        ax.text(x_min, entry, f" REF ${entry:.2f}", color='#3b82f6', fontsize=8, va='bottom', alpha=0.8)
        ax.text(x_min, sl, f" SL ${sl:.2f}", color='#ef4444', fontsize=8, va='top', alpha=0.8)
        
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.05)
        ax.add_patch(rect_prem)
        ax.text(x_min, swing_high, " Premium", color='#ef4444', fontsize=6, va='top', alpha=0.5)
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.05)
        ax.add_patch(rect_disc)
        ax.text(x_min, swing_low, " Discount", color='#10b981', fontsize=6, va='bottom', alpha=0.5)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return None

# --- 5. 處理單一股票的函式 (封裝以供重複使用) ---
def process_ticker(t, app_data_dict):
    try:
        time.sleep(0.1)
        
        # 獲取日線
        df_d = get_polygon_data(t, 1, 'day')
        if df_d is None or len(df_d) < 50: return None
        
        # 獲取小時線
        df_h = get_polygon_data(t, 1, 'hour')
        if df_h is None: df_h = df_d

        curr_price = df_d['Close'].iloc[-1]
        sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
        if pd.isna(sma200): sma200 = curr_price

        # SMC
        bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
        tp = bsl

        # 訊號
        is_bullish = curr_price > sma200
        in_discount = curr_price < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # 繪圖
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "Daily Structure", entry, sl, tp, is_wait)
        img_h = generate_chart(df_h, t, "Hourly Execution", entry, sl, tp, is_wait)
        
        # AI 文案
        trend_str = "多頭 (Bullish)" if is_bullish else "空頭 (Bearish)"
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        if signal == "LONG":
            ai_html = f"""
            <div class='deploy-box long'>
                <div class='deploy-title'>✅ LONG SETUP (做多建議)</div>
                <ul class='deploy-list'>
                    <li><b>入場 (FVG):</b> ${entry:.2f}</li>
                    <li><b>止損 (SL):</b> ${sl:.2f}</li>
                    <li><b>止盈 (TP):</b> ${tp:.2f}</li>
                    <li><b>盈虧比:</b> {rr:.1f}R</li>
                </ul>
                <div style='margin-top:10px; font-size:0.85rem'>
                    🤖 <b>AI 分析:</b> 高交易量熱門股！股價位於 200MA 之上，回調至折價區，SMC 結構完整。
                </div>
            </div>"""
        else:
            reason = "無明顯 FVG" if not found_fvg else ("趨勢偏空" if not is_bullish else "位於溢價區")
            ai_html = f"""
            <div class='deploy-box wait'>
                <div class='deploy-title'>⏳ WAIT (觀望)</div>
                <ul class='deploy-list'>
                    <li><b>趨勢:</b> {trend_str}</li>
                    <li><b>位置:</b> {"溢價區" if curr_price >= eq else "折價區"}</li>
                    <li><b>原因:</b> {reason}</li>
                </ul>
                <div style='margin-top:10px; font-size:0.85rem; color:#cbd5e1;'>
                    🤖 雖然條件未滿足，但已畫出參考結構。
                </div>
            </div>"""

        # 存入字典
        app_data_dict[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}
        
        return {
            "ticker": t,
            "price": curr_price,
            "signal": signal,
            "cls": "b-long" if signal == "LONG" else "b-wait",
            "is_bullish": is_bullish,
            "found_fvg": found_fvg
        }
    except Exception as e:
        print(f"Error {t}: {e}")
        return None

# --- 6. 主程式 ---
def main():
    print("🚀 Starting Top 100 Volume Scanner...")
    
    if not API_KEY:
        print("❌ FATAL: API Key missing")
        return

    # 1. 抓新聞
    weekly_news_html = get_polygon_news()
    
    # 2. 抓 Top 100 熱門股
    top_100_tickers = get_top_volume_tickers(limit=100)
    
    APP_DATA = {}
    sector_html_blocks = ""
    screener_rows = ""
    
    # 3. 處理固定板塊 (Overview Tab)
    print("📊 Processing Fixed Sectors...")
    for sector, tickers in SECTORS.items():
        cards = ""
        for t in tickers:
            res = process_ticker(t, APP_DATA)
            if res:
                cards += f"""
                <div class="card" onclick="openModal('{t}')">
                    <div class="head"><div><div class="code">{t}</div><div class="price">${res['price']:.2f}</div></div><span class="badge {res['cls']}">{res['signal']}</span></div>
                    <div class="hint">Tap for Analysis ↗</div>
                </div>"""
        if cards:
            sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    # 4. 處理 Top 100 (Screener Tab)
    print("🔥 Processing Top 100 Volume...")
    # 為了避免重複處理，先過濾掉已經在 SECTORS 裡跑過的
    processed_set = set([t for sec in SECTORS.values() for t in sec])
    
    for t in top_100_tickers:
        if t in processed_set: continue # 已經跑過就跳過，節省時間
        
        res = process_ticker(t, APP_DATA)
        if res:
            # 只有 LONG 訊號才加入 Screener 表格
            if res['signal'] == "LONG":
                screener_rows += f"<tr><td>{t}</td><td>${res['price']:.2f}</td><td class='g'>🔥 Volume Leader</td><td><span class='badge {res['cls']}'>{res['signal']}</span></td></tr>"

    json_data = json.dumps(APP_DATA)

    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DailyDip Pro (Volume Scanner)</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
        body {{ background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding:10px; }}
        .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; border-bottom:1px solid #333; overflow-x:auto; }}
        .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; font-weight:bold; font-size:0.9rem; white-space:nowrap; }}
        .tab.active {{ background:var(--acc); color:white; }}
        .content {{ display:none; }} .content.active {{ display:block; }}
        .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
        .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:10px; cursor:pointer; transition:0.2s; }}
        .card:hover {{ border-color:var(--acc); transform:translateY(-2px); }}
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
        .news-title:hover {{ color:var(--acc); }}
        .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:99; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
        .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; margin-top:20px; border:1px solid #555; }}
        .m-content img {{ width:100%; border-radius:6px; margin-bottom:10px; }}
        .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
        .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
        .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
        .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:6px; font-weight:bold; margin-top:10px; cursor:pointer; }}
        .time {{ text-align:center; color:#666; font-size:0.7rem; margin-top:30px; }}
        .chart-lbl {{ color:var(--acc); font-weight:bold; display:block; margin-bottom:5px; font-size:0.9rem; margin-top:10px; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選 (Top 100)</div>
            <div class="tab" onclick="setTab('news', this)">📰 熱門新聞</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content">
            <div style="padding:10px; background:rgba(16,185,129,0.1); margin-bottom:15px; border-radius:6px; font-size:0.9rem;">
                🎯 <b>全市場掃描：</b> 已掃描成交量最大的 100 隻股票，以下是符合 <b>SMC 做多條件</b> 的強勢股。
            </div>
            <table><thead><tr><th>Ticker</th><th>Price</th><th>Source</th><th>Signal</th></tr></thead><tbody>{screener_rows if screener_rows else "<tr><td colspan='4' style='text-align:center;padding:20px'>Top 100 中暫無符合完美條件的標的</td></tr>"}</tbody></table>
        </div>
        <div id="news" class="content"><h3 class="sector-title">Polygon Hot News</h3>{weekly_news_html}</div>
        
        <div class="time">Powered by Polygon.io | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><span class="chart-lbl">📅 Daily Structure</span><div id="chart-d"></div></div>
                <div><span class="chart-lbl">⏱️ Hourly Execution</span><div id="chart-h"></div></div>
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
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated successfully!")

if __name__ == "__main__":
    main()
