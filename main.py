import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta

# --- 0. 設定 ---
API_KEY = os.environ.get("POLYGON_API_KEY")

# --- 1. 觀察清單 ---
SECTORS = {
    "🔥 熱門交易": ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "SOFI", "MARA", "MSTR", "SMCI"],
    "💎 科技巨頭": ["MSFT", "AMZN", "GOOGL", "META", "NFLX", "CRM", "ADBE"],
    "⚡ 半導體": ["TSM", "AVGO", "MU", "INTC", "ARM", "QCOM", "TXN", "AMAT"],
    "🚀 成長股": ["COIN", "HOOD", "DKNG", "RBLX", "U", "CVNA", "OPEN", "SHOP", "NET"],
    "🏦 金融與消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT", "DIS", "SBUX"],
    "📉 指數 ETF": ["SPY", "QQQ", "IWM", "TQQQ", "SQQQ"]
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 新聞 ---
def get_polygon_news():
    if not API_KEY: return "<div style='padding:20px'>API Key Missing</div>"
    news_html = ""
    try:
        url = f"https://api.polygon.io/v2/reference/news?limit=12&order=desc&sort=published_utc&apiKey={API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('results'):
            for item in data['results']:
                title = item.get('title')
                url = item.get('article_url')
                pub = item.get('publisher', {}).get('name', 'Unknown')
                dt = item.get('published_utc', '')[:10]
                news_html += f"<div class='news-item'><div class='news-meta'>{pub} • {dt}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else: news_html = "<div style='padding:20px'>暫無新聞</div>"
    except: news_html = "News Error"
    return news_html

# --- 3. 數據獲取 (只抓數據，不畫圖) ---
def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

# --- 4. 數據轉換 (轉成前端能讀的 JSON) ---
def df_to_chart_data(df):
    # 只取最後 60 根 K 線，減少檔案大小
    df_slice = df.tail(60).copy()
    chart_data = []
    
    for index, row in df_slice.iterrows():
        # 處理時間格式
        t = int(index.timestamp()) # UNIX timestamp for robustness
        
        chart_data.append({
            "time": t,
            "open": round(row['Open'], 2),
            "high": round(row['High'], 2),
            "low": round(row['Low'], 2),
            "close": round(row['Close'], 2)
        })
    return chart_data

# --- 5. SMC 運算 ---
def calculate_smc(df):
    try:
        window = 50
        recent = df.tail(window)
        bsl = float(recent['High'].max())
        ssl = float(recent['Low'].min())
        eq = (bsl + ssl) / 2
        best_entry = eq
        found_fvg = False
        fvg_list = [] # 前端繪圖用
        
        for i in range(2, len(recent)):
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]: # Bullish FVG
                top, bot = float(recent['Low'].iloc[i]), float(recent['High'].iloc[i-2])
                # 記錄 FVG 時間與價格範圍
                fvg_time = int(recent.index[i-1].timestamp())
                fvg_list.append({"time": fvg_time, "top": top, "bot": bot, "type": "bull"})
                
                if top < eq:
                    best_entry = top
                    found_fvg = True
        
        return bsl, ssl, eq, best_entry, ssl*0.99, found_fvg
    except:
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False

# --- 6. 評分 ---
def calculate_score(df, entry, sl, tp, is_bullish):
    try:
        score = 60
        close = df['Close'].iloc[-1]
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        if rr >= 3: score += 15
        elif rr >= 2: score += 10
        elif rr < 1: score -= 20
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain/loss
        rsi = 100 - (100/(1+rs)).iloc[-1]
        
        if 40 <= rsi <= 55: score += 15
        elif rsi > 70: score -= 15
        
        # Trend
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        if close > sma50 > sma200: score += 10
        if close < sma50: score -= 5
        
        # Distance
        dist = abs(close - entry)/entry
        if dist < 0.01: score += 20
        elif dist < 0.03: score += 10
        
        return min(max(int(score), 0), 99), rr
    except: return 50, 0

# --- 7. 單一股票處理 ---
def process_ticker(t, app_data_dict):
    try:
        time.sleep(0.3)
        df_d = fetch_data(t, "1y", "1d")
        if df_d is None or len(df_d) < 50: return None
        
        df_h = fetch_data(t, "1mo", "1h")
        if df_h is None: df_h = df_d
        
        curr = float(df_d['Close'].iloc[-1])
        sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
        if pd.isna(sma200): sma200 = curr
        
        # SMC
        bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
        tp = bsl
        
        # Signal
        is_bullish = curr > sma200
        in_discount = curr < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # Score
        score, rr = calculate_score(df_d, entry, sl, tp, is_bullish)
        
        # 準備前端需要的數據
        chart_data_d = df_to_chart_data(df_d)
        chart_data_h = df_to_chart_data(df_h)
        
        levels = {
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": round(rr, 2)
        }
        
        # 文案
        score_color = "#10b981" if score >= 80 else ("#fbbf24" if score >= 60 else "#ef4444")
        if signal == "LONG":
            deploy_html = f"""
            <div class='deploy-box long'>
                <div class='deploy-title'>✅ LONG SETUP</div>
                <div style='display:flex;justify-content:space-between;border-bottom:1px solid #333;padding-bottom:5px;margin-bottom:5px;'>
                    <span>評分: <b style='color:{score_color};font-size:1.1em'>{score}</b></span>
                    <span>RR: <b style='color:#10b981'>{rr:.1f}R</b></span>
                </div>
                <ul class='deploy-list'>
                    <li>TP: ${tp:.2f}</li><li>Entry: ${entry:.2f}</li><li>SL: ${sl:.2f}</li>
                </ul>
            </div>"""
        else:
            reason = "無FVG" if not found_fvg else ("逆勢" if not is_bullish else "溢價區")
            deploy_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><div>評分: <b style='color:#94a3b8'>{score}</b></div><ul class='deploy-list'><li>狀態: {reason}</li><li>參考入場: ${entry:.2f}</li></ul></div>"
        
        app_data_dict[t] = {
            "signal": signal, 
            "deploy": deploy_html, 
            "score": score,
            "chart_d": chart_data_d, # K線數據
            "chart_h": chart_data_h,
            "levels": levels # 關鍵位
        }
        
        return {"ticker": t, "price": curr, "signal": signal, "cls": "b-long" if signal=="LONG" else "b-wait", "score": score}
    except Exception as e:
        print(f"Err {t}: {e}")
        return None

# --- 8. 主程式 ---
def main():
    print("🚀 Starting Analysis (Client-Side Rendering Mode)...")
    weekly_news_html = get_polygon_news()
    
    APP_DATA, sector_html_blocks, screener_rows = {}, "", []
    
    for sector, tickers in SECTORS.items():
        cards = ""
        results = []
        for t in tickers:
            res = process_ticker(t, APP_DATA)
            if res:
                results.append(res)
                if res['signal'] == "LONG": screener_rows.append(res)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        for res in results:
            t = res['ticker']
            s_c = "#10b981" if res['score']>=80 else "#fbbf24"
            cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><div style='text-align:right'><span class='badge {res['cls']}'>{res['signal']}</span><div style='font-size:0.7rem;color:{s_c}'>{res['score']}</div></div></div></div>"
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    screener_rows.sort(key=lambda x: x['score'], reverse=True)
    screener_html = "".join([f"<tr><td>{r['ticker']}</td><td>${r['price']:.2f}</td><td><b>{r['score']}</b></td><td><span class='badge {r['cls']}'>{r['signal']}</span></td></tr>" for r in screener_rows])

    # 將數據轉為 JSON 字串，嵌入 HTML
    json_data = json.dumps(APP_DATA)
    
    # HTML 包含 TradingView Lightweight Charts 邏輯
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DailyDip Pro</title>
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
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
    table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
    th, td {{ padding:8px; text-align:left; border-bottom:1px solid #333; }}
    .g {{ color:var(--g); font-weight:bold; }}
    .news-item {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:15px; margin-bottom:10px; }}
    .news-title {{ color:var(--text); text-decoration:none; font-weight:bold; display:block; }}
    .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:99; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
    .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; margin-top:20px; border:1px solid #555; }}
    .chart-container {{ height: 250px; width: 100%; margin-bottom: 20px; border: 1px solid #333; border-radius: 6px; }}
    .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
    .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
    .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
    .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:6px; font-weight:bold; margin-top:10px; cursor:pointer; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選</div>
            <div class="tab" onclick="setTab('news', this)">📰 News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Score</th><th>Signal</th></tr></thead><tbody>{screener_html}</tbody></table></div>
        <div id="news" class="content">{weekly_news_html}</div>
        
        <div style="text-align:center;color:#666;margin-top:20px;font-size:0.7rem">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="closeModal()">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                
                <b>Daily Structure</b>
                <div id="chart-d" class="chart-container"></div>
                
                <b>Hourly Execution</b>
                <div id="chart-h" class="chart-container"></div>
                
                <button class="close-btn" onclick="closeModal()">Close</button>
            </div>
        </div>

        <script>
        const STOCK_DATA = {json_data};
        let chartD, chartH; // Chart instances

        function setTab(id, el) {{
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            el.classList.add('active');
        }}

        function createChart(containerId, data, levels) {{
            document.getElementById(containerId).innerHTML = ''; // Clear old chart
            
            const chart = LightweightCharts.createChart(document.getElementById(containerId), {{
                layout: {{ background: {{ type: 'solid', color: '#0f172a' }}, textColor: '#d1d5db' }},
                grid: {{ vertLines: {{ color: '#334155' }}, horzLines: {{ color: '#334155' }} }},
                timeScale: {{ timeVisible: true, borderColor: '#475569' }},
                rightPriceScale: {{ borderColor: '#475569' }},
            }});

            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#10b981', downColor: '#ef4444', borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#ef4444'
            }});
            
            candleSeries.setData(data);

            // Add SMC Lines
            if(levels) {{
                candleSeries.createPriceLine({{ price: levels.tp, color: '#10b981', lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: 'TP' }});
                candleSeries.createPriceLine({{ price: levels.entry, color: '#3b82f6', lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: 'ENTRY' }});
                candleSeries.createPriceLine({{ price: levels.sl, color: '#ef4444', lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: 'SL' }});
            }}
            
            chart.timeScale().fitContent();
            return chart;
        }}

        function openModal(ticker) {{
            const data = STOCK_DATA[ticker];
            if (!data) return;
            
            document.getElementById('modal').style.display = 'flex';
            document.getElementById('m-ticker').innerText = ticker;
            document.getElementById('m-deploy').innerHTML = data.deploy;
            
            // Render Charts using JS
            if (chartD) chartD.remove();
            if (chartH) chartH.remove();
            
            // Daily Chart
            if (data.chart_d && data.chart_d.length > 0) {{
                chartD = createChart('chart-d', data.chart_d, data.levels);
            }} else {{
                document.getElementById('chart-d').innerHTML = '<div style="padding:20px;text-align:center">No Data</div>';
            }}

            // Hourly Chart
            if (data.chart_h && data.chart_h.length > 0) {{
                chartH = createChart('chart-h', data.chart_h, data.levels);
            }} else {{
                document.getElementById('chart-h').innerHTML = '<div style="padding:20px;text-align:center">No Data</div>';
            }}
        }}

        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}
        </script>
    </body></html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated!")

if __name__ == "__main__":
    main()
