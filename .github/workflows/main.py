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

# --- 全域診斷日誌 ---
DIAGNOSTIC_LOG = []

def log_msg(msg):
    print(msg)
    DIAGNOSTIC_LOG.append(str(msg))

# --- 1. 設定觀察清單 (先縮減數量，專注測試) ---
SECTORS = {
    "💎 測試清單": ["NVDA", "AAPL", "TSLA"], 
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 測試 API 連線 (關鍵步驟) ---
def test_api_connection():
    if not API_KEY:
        log_msg("❌ 致命錯誤: GitHub Secrets 內沒有找到 POLYGON_API_KEY")
        return False
    
    # 測試 1: 檢查 Key 是否有效 (查詢 AAPL 詳情)
    url = f"https://api.polygon.io/v3/reference/tickers/AAPL?apiKey={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('status') == 'OK':
            log_msg(f"✅ API Key 驗證成功! (連線正常)")
            return True
        else:
            log_msg(f"❌ API Key 驗證失敗: {data}")
            return False
    except Exception as e:
        log_msg(f"❌ 無法連線到 Polygon: {e}")
        return False

# --- 3. 數據獲取 (診斷模式) ---
def get_polygon_data(ticker):
    # 策略：抓取過去 5 天的數據 (避開假日或未收盤問題)
    # 你的 Starter Plan 權限通常是 "End of Day" (EOD)
    
    # 將結束日期設為 2 天前，確保數據絕對已經入庫
    end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
    
    # URL (印出來檢查用，但隱藏 Key)
    url_mask = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey=******"
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
    
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        # --- 診斷重點：印出 API 回傳什麼 ---
        if data.get('status') != 'OK' or data.get('resultsCount', 0) == 0:
            log_msg(f"⚠️ {ticker} 請求失敗/無數據:")
            log_msg(f"   - URL: {url_mask}")
            log_msg(f"   - Response: {json.dumps(data)}") # 印出錯誤代碼
            return None
            
        log_msg(f"✅ {ticker} 成功抓到 {data.get('resultsCount')} 筆數據")
        
        df = pd.DataFrame(data['results'])
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        log_msg(f"❌ 程式錯誤 ({ticker}): {e}")
        return None

# --- 4. 繪圖與 SMC (簡化版以確保運行) ---
def generate_chart(df, ticker):
    try:
        if len(df) < 20: return None
        plot_df = df.tail(50)
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        buf = BytesIO()
        mpf.plot(plot_df, type='candle', style=s, volume=False, title=ticker, figsize=(5, 3), savefig=buf)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        log_msg(f"Plot Error {ticker}: {e}")
        return None

# --- 5. 主程式 ---
def main():
    log_msg("🚀 啟動診斷模式 (Diagnostic Mode)...")
    
    # 1. 測試連線
    if not test_api_connection():
        log_msg("⚠️ 停止執行：API Key 無效或無法連線")
    
    sector_html_blocks = ""
    screener_rows = ""
    APP_DATA = {}

    # 2. 開始抓取 (只抓測試清單)
    for sector, tickers in SECTORS.items():
        cards_in_sector = ""
        for t in tickers:
            df = get_polygon_data(t)
            
            if df is None: continue
            
            # 簡單計算
            curr_price = df['Close'].iloc[-1]
            img = generate_chart(df, t)
            
            APP_DATA[t] = {"price": f"${curr_price:.2f}", "img": img}
            
            cards_in_sector += f"""
            <div class="card" onclick="openModal('{t}')">
                <div class="head"><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                <div class="hint">Tap for Chart</div>
            </div>"""
            
            screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td>OK</td></tr>"
            
        if cards_in_sector:
            sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

    # 生成 HTML
    log_html = "<br>".join(DIAGNOSTIC_LOG)
    json_data = json.dumps(APP_DATA)
    
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DailyDip Diagnostic</title>
    <style>
        body {{ background:#0f172a; color:white; font-family:monospace; padding:20px; }}
        .log-box {{ background:#330000; border:1px solid #ff4444; color:#ffcccc; padding:15px; border-radius:8px; white-space:pre-wrap; margin-bottom:20px; font-size:0.8rem; }}
        .card {{ background:#1e293b; padding:15px; border-radius:8px; margin-bottom:10px; border:1px solid #334155; cursor:pointer; }}
        .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:99; justify-content:center; align-items:center; }}
        .m-content {{ background:#1e293b; padding:20px; border-radius:10px; max-width:90%; }}
    </style>
    </head>
    <body>
        <h2>🛠️ 系統診斷報告</h2>
        <div class="log-box">{log_html}</div>
        
        <h3>測試結果：</h3>
        {sector_html_blocks if sector_html_blocks else "<div>無數據可顯示，請查看上方紅框內的錯誤訊息。</div>"}
        
        <div id="modal" class="modal" onclick="this.style.display='none'">
            <div class="m-content">
                <h2 id="m-ticker"></h2>
                <div id="m-chart"></div>
            </div>
        </div>

        <script>
        const DATA = {json_data};
        function openModal(t) {{
            const d = DATA[t];
            if(!d) return;
            document.getElementById('modal').style.display = 'flex';
            document.getElementById('m-ticker').innerText = t;
            document.getElementById('m-chart').innerHTML = d.img ? '<img src="'+d.img+'" style="width:100%">' : 'No Image';
        }}
        </script>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ Diagnostic index.html generated!")

if __name__ == "__main__":
    main()
