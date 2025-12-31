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

# --- 全域錯誤收集 ---
DEBUG_LOG = []

def log_msg(msg):
    print(msg)
    DEBUG_LOG.append(msg)

# --- 1. 設定觀察清單 (擴充版) ---
SECTORS = {
    "💎 科技七巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI"],
    "☁️ 軟體與SaaS": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PANW", "CRWD", "PLTR"],
    "🏦 金融": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. Polygon 核心請求函式 (加入重試機制) ---
def fetch_polygon(url, retries=3):
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429: # Too Many Requests
                log_msg(f"⚠️ Rate Limit Hit. Sleeping 1s... (Attempt {i+1}/{retries})")
                time.sleep(1)
                continue
            else:
                log_msg(f"❌ API Error: {resp.status_code} | {resp.text[:100]}")
                return None
        except Exception as e:
            log_msg(f"❌ Connection Error: {e}")
            time.sleep(1)
    return None

def get_polygon_data(ticker, multiplier=1, timespan='day'):
    if not API_KEY: return None
    
    # 計算日期 (抓取過去 200 天以計算 MA)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    
    # 建構 URL
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
    
    data = fetch_polygon(url)
    
    if data and data.get('status') == 'OK' and data.get('results'):
        df = pd.DataFrame(data['results'])
        # 轉換時間戳
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('Date', inplace=True)
        # 重新命名欄位
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        log_msg(f"⚠️ No Data returned for {ticker} ({timespan})")
        return None

def get_polygon_news():
    if not API_KEY: return "<div>API Key Missing</div>"
    
    # 抓取熱門股新聞
    tickers = "SPY,QQQ,NVDA,TSLA,AAPL,AMD"
    url = f"https://api.polygon.io/v2/reference/news?ticker={tickers}&limit=15&sort=published_utc&order=desc&apiKey={API_KEY}"
    
    data = fetch_polygon(url)
    news_html = ""
    
    if data and data.get('results'):
        for item in data['results']:
            title = item.get('title')
            url = item.get('article_url')
            publisher = item.get('publisher', {}).get('name', 'Unknown')
            pub_time = item.get('published_utc', '')
            
            try:
                dt = datetime.strptime(pub_time, "%Y-%m-%dT%H:%M:%SZ")
                date_str = dt.strftime('%m/%d %H:%M')
            except: date_str = ""
            
            news_html += f"""
            <div class="news-item">
                <div class="news-meta">{publisher} • {date_str}</div>
                <a href="{url}" target="_blank" class="news-title">{title}</a>
            </div>
            """
    else:
        news_html = "<div style='padding:20px'>本週暫無熱門新聞</div>"
        
    return news_html

# --- 3. SMC 分析邏輯 ---
def calculate_smc(df):
    """計算 SMC 關鍵位：尋找折價區內的 FVG 作為入場"""
    window = 50
    recent = df.tail(window)
    
    bsl = recent['High'].max() # TP
    ssl = recent['Low'].min()  # SL
    eq = (bsl + ssl) / 2       # 平衡點
    
    best_entry = eq
    # 倒序尋找最近的 Bullish FVG
    for i in range(len(recent)-1, 2, -1):
        # FVG 條件: Low[i] > High[i-2]
        if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
            fvg_top = recent['Low'].iloc[i]
            # 只有當 FVG 在折價區 (低於 EQ) 才考慮
            if fvg_top < eq:
                best_entry = fvg_top
                break
                
    return bsl, ssl, eq, best_entry, ssl * 0.99

def generate_chart(df, ticker, title, entry, sl, tp):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 30: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2

        # 樣式設定
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        # 繪製 SMC 線
        hlines = dict(
            hlines=[tp, entry, sl],
            colors=['#10b981', '#3b82f6', '#ef4444'],
            linewidths=[1, 1, 1],
            linestyle=['-', '--', '-']
        )
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 文字標註
        ax.text(x_min, tp, f" TP (BSL) ${tp:.2f}", color='#10b981', fontsize=7, va='bottom', fontweight='bold')
        ax.text(x_min, entry, f" ENTRY (FVG) ${entry:.2f}", color='#3b82f6', fontsize=7, va='bottom', fontweight='bold')
        ax.text(x_min, sl, f" SL (SSL) ${sl:.2f}", color='#ef4444', fontsize=7, va='top', fontweight='bold')
        
        # 區域標示
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.05)
        ax.add_patch(rect_prem)
        ax.text(x_min, swing_high, " Premium Zone (Sell)", color='#ef4444', fontsize=6, va='top', alpha=0.6)
        
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.05)
        ax.add_patch(rect_disc)
        ax.text(x_min, swing_low, " Discount Zone (Buy)", color='#10b981', fontsize=6, va='bottom', alpha=0.6)

        # 轉換圖片
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        log_msg(f"Plot Error {ticker}: {e}")
        return None

# --- 4. 主程式 ---
def main():
    print("🚀 Starting Polygon Pro Analysis...")
    
    if not API_KEY:
        log_msg("❌ FATAL: POLYGON_API_KEY is missing!")
        # 這裡不 return，繼續執行以生成錯誤頁面
    
    # 1. 抓新聞
    weekly_news_html = get_polygon_news()

    sector_html_blocks = ""
    screener_rows = ""
    APP_DATA = {}
    
    if API_KEY:
        for sector, tickers in SECTORS.items():
            cards_in_sector = ""
            for t in tickers:
                try:
                    # 避免頻率限制
                    time.sleep(0.1) 
                    
                    # 2. 獲取數據
                    df_d = get_polygon_data(t, 1, 'day')
                    if df_d is None or len(df_d) < 50: continue
                    
                    # 嘗試獲取小時線，失敗則用日線代替
                    df_h = get_polygon_data(t, 1, 'hour')
                    if df_h is None: df_h = df_d

                    curr_price = df_d['Close'].iloc[-1]
                    sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                    if pd.isna(sma200): sma200 = curr_price

                    # 3. SMC 計算
                    bsl, ssl, eq, entry, sl = calculate_smc(df_d)
                    tp = bsl

                    # 4. 繪圖
                    img_d = generate_chart(df_d, t, "Daily", entry, sl, tp)
                    if not img_d: continue
                    
                    img_h = generate_chart(df_h, t, "Hourly Execution", entry, sl, tp)
                    if not img_h: img_h = ""

                    # 5. 訊號與AI文案
                    is_bullish = curr_price > sma200
                    signal = "LONG" if is_bullish and curr_price < eq else "WAIT"
                    cls = "b-long" if signal == "LONG" else "b-wait"
                    
                    risk = entry - sl
                    reward = tp - entry
                    rr = reward / risk if risk > 0 else 0
                    
                    if signal == "LONG":
                        ai_html = f"""
                        <div class='deploy-box long'>
                            <div class='deploy-title'>✅ LONG SETUP (做多)</div>
                            <ul class='deploy-list'>
                                <li><b>入場 (Entry):</b> ${entry:.2f}</li>
                                <li><b>止損 (SL):</b> ${sl:.2f}</li>
                                <li><b>止盈 (TP):</b> ${tp:.2f}</li>
                                <li><b>盈虧比:</b> {rr:.1f}R</li>
                            </ul>
                            <div style='margin-top:10px; font-size:0.85rem'>
                                🤖 <b>AI 分析:</b> 趨勢向上(>200MA)且回調至折價區，SMC結構完整。
                            </div>
                        </div>
                        """
                    else:
                        ai_html = f"""
                        <div class='deploy-box wait'>
                            <div class='deploy-title'>⏳ WAIT (觀望)</div>
                            <ul class='deploy-list'>
                                <li>價格位於 <b>Premium (溢價區)</b></li>
                                <li>等待回調至: <b>${entry:.2f}</b></li>
                                <li>目前趨勢: {"多頭" if is_bullish else "空頭"}</li>
                            </ul>
                        </div>
                        """
                    
                    APP_DATA[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}

                    cards_in_sector += f"""
                    <div class="card" onclick="openModal('{t}')">
                        <div class="head"><div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div><span class="badge {cls}">{signal}</span></div>
                        <div class="hint">Tap for SMC ↗</div>
                    </div>"""
                    
                    if is_bullish and curr_price < eq:
                        screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>Buy Zone</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

                except Exception as e:
                    log_msg(f"Error processing {t}: {e}")
                    continue
            
            if cards_in_sector:
                sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

    # JSON 數據
    json_data = json.dumps(APP_DATA)
    # 錯誤日誌顯示 (如果有)
    error_html = "<br>".join(DEBUG_LOG[-5:]) if DEBUG_LOG else "" # 只顯示最後5條錯誤以免太長

    # HTML
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DailyDip Pro (Polygon Paid)</title>
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
        .debug {{ background:#330000; color:#ff9999; padding:10px; font-size:0.7rem; margin-bottom:10px; border-radius:4px; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 Market</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 Screener</div>
            <div class="tab" onclick="setTab('news', this)">📰 Polygon News</div>
        </div>
        
        {f'<div class="debug"><b>System Log:</b><br>{error_html}</div>' if error_html else ''}

        <div id="overview" class="content active">{sector_html_blocks if sector_html_blocks else '<div style="text-align:center;padding:50px">No Data Loaded. Check System Log above.</div>'}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Trend</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table></div>
        <div id="news" class="content"><h3 class="sector-title">Polygon Hot News</h3>{weekly_news_html}</div>
        
        <div style="text-align:center; color:#666; margin-top:30px; font-size:0.7rem;">
            Powered by Polygon.io Paid Tier | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        </div>

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
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated successfully!")

if __name__ == "__main__":
    main()
