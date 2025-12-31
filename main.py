import os
import requests
import yfinance as yf
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

# --- 0. 讀取 API KEY (只用於新聞) ---
API_KEY = os.environ.get("POLYGON_API_KEY")

# --- 1. 設定觀察清單 (擴充熱門成交量股) ---
# 這些是美股成交量最大的 40+ 隻股票
SECTORS = {
    "🔥 熱門交易 (Top Volume)": ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "SOFI", "MARA", "F", "BAC"],
    "💎 科技巨頭 (Mag 7)": ["MSFT", "AMZN", "GOOGL", "META", "NFLX"],
    "⚡ 半導體 (Semis)": ["TSM", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI", "SOXL"],
    "🚀 成長與加密 (Growth)": ["COIN", "MSTR", "HOOD", "DKNG", "RBLX", "U", "CVNA", "OPEN"],
    "🏦 金融與消費 (Value)": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT", "DIS"],
    "📉 指數 ETF": ["SPY", "QQQ", "IWM", "TQQQ", "SQQQ"]
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 獲取 Polygon 新聞 (保留你的付費優勢) ---
def get_polygon_news():
    if not API_KEY: return "<div style='padding:20px'>API Key Missing</div>"
    news_html = ""
    try:
        # 抓取最近的熱門新聞
        url = f"https://api.polygon.io/v2/reference/news?limit=15&order=desc&sort=published_utc&apiKey={API_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if data.get('results'):
            for item in data['results']:
                title = item.get('title')
                article_url = item.get('article_url')
                publisher = item.get('publisher', {}).get('name', 'Unknown')
                pub_time = item.get('published_utc', '')
                
                try:
                    dt = datetime.strptime(pub_time, "%Y-%m-%dT%H:%M:%SZ")
                    date_str = dt.strftime('%m/%d %H:%M')
                except: date_str = ""
                
                news_html += f"""
                <div class="news-item">
                    <div class="news-meta">{publisher} • {date_str}</div>
                    <a href="{article_url}" target="_blank" class="news-title">{title}</a>
                </div>
                """
        else:
            news_html = "<div style='padding:20px'>暫無新聞</div>"
    except Exception as e:
        news_html = f"<div style='padding:20px'>新聞載入錯誤: {e}</div>"
    return news_html

# --- 3. SMC 分析邏輯 (Yahoo 數據版) ---
def calculate_smc(df):
    """
    計算 SMC 關鍵位。
    如果找不到 FVG，強制使用 EQ 作為 Entry，保證畫出線條。
    """
    try:
        window = 50
        recent = df.tail(window)
        
        bsl = float(recent['High'].max()) # TP
        ssl = float(recent['Low'].min())  # SL
        eq = (bsl + ssl) / 2       # 平衡點
        
        best_entry = eq # 預設入場點
        found_fvg = False
        
        # 尋找折價區內的最近 Bullish FVG
        for i in range(len(recent)-1, 2, -1):
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]: # Bullish FVG
                fvg_top = float(recent['Low'].iloc[i])
                if fvg_top < eq: # 必須在折價區
                    best_entry = fvg_top
                    found_fvg = True
                    break
        
        sl_price = ssl * 0.99
        return bsl, ssl, eq, best_entry, sl_price, found_fvg
    except:
        # 安全回傳
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False

# --- 4. 繪圖函式 (保證出圖) ---
def generate_chart(df, ticker, title, entry, sl, tp, is_wait):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 10: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2

        # 確保數值安全
        if np.isnan(entry): entry = eq
        if np.isnan(sl): sl = swing_low
        if np.isnan(tp): tp = swing_high

        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        # 線條樣式：WAIT 狀態用虛線，LONG 狀態用實線
        line_style = ':' if is_wait else '--'
        line_alpha = 0.5 if is_wait else 0.9
        
        hlines = dict(
            hlines=[tp, entry, sl],
            colors=['#10b981', '#3b82f6', '#ef4444'],
            linewidths=[1.5, 1.5, 1.5],
            linestyle=['-', line_style, '-'],
            alpha=line_alpha
        )
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 標註文字
        ax.text(x_min, tp, f" TP {tp:.2f}", color='#10b981', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, entry, f" ENTRY {entry:.2f}", color='#3b82f6', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, sl, f" SL {sl:.2f}", color='#ef4444', fontsize=8, va='top', fontweight='bold')
        
        # 區域標示
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
    except Exception as e:
        print(f"Chart Error {ticker}: {e}")
        return None

# --- 5. 主程式 ---
def main():
    print("🚀 Starting Analysis (Yahoo Finance Mode)...")
    
    # 1. 抓新聞 (用 Polygon)
    weekly_news_html = get_polygon_news()

    # 2. 抓股價 (用 Yahoo Finance - 一次下載全部，速度快)
    print("📊 Downloading Market Data from Yahoo...")
    try:
        # 下載日線
        data_daily = yf.download(ALL_TICKERS, period="1y", interval="1d", group_by='ticker', progress=False)
        # 下載小時線 (Yahoo 限制小時線最多 730 天，我們抓 1 個月即可)
        data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    except Exception as e:
        print(f"Yahoo Download Error: {e}")
        return

    APP_DATA = {}
    sector_html_blocks = ""
    screener_rows = ""
    
    for sector, tickers in SECTORS.items():
        cards_in_sector = ""
        for t in tickers:
            try:
                # 提取個股數據
                if len(tickers) == 1: # 單隻股票的情況
                    df_d = data_daily
                    df_h = data_hourly
                else:
                    try:
                        df_d = data_daily[t]
                        df_h = data_hourly[t]
                    except: continue # 該股票無數據
                
                # 清洗數據
                df_d = df_d.dropna()
                df_h = df_h.dropna()
                
                if len(df_d) < 50: continue

                curr_price = float(df_d['Close'].iloc[-1])
                sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
                if pd.isna(sma200): sma200 = curr_price

                # SMC 計算
                bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
                tp = bsl

                # 訊號判斷
                is_bullish = curr_price > sma200
                in_discount = curr_price < eq
                
                # LONG: 多頭 + 折價區 + (有FVG 或 強制策略)
                signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
                
                # 繪圖 (WAIT 狀態也要畫圖)
                is_wait = (signal == "WAIT")
                img_d = generate_chart(df_d, t, "Daily Structure", entry, sl, tp, is_wait)
                img_h = generate_chart(df_h, t, "Hourly Execution", entry, sl, tp, is_wait)
                
                if not img_d: img_d = ""
                if not img_h: img_h = ""

                # AI 文案
                cls = "b-long" if signal == "LONG" else "b-wait"
                trend_str = "多頭 (Bullish)" if is_bullish else "空頭 (Bearish)"
                risk = entry - sl
                reward = tp - entry
                rr = reward / risk if risk > 0 else 0
                
                if signal == "LONG":
                    ai_html = f"""
                    <div class='deploy-box long'>
                        <div class='deploy-title'>✅ LONG SETUP (做多)</div>
                        <ul class='deploy-list'>
                            <li><b>Entry:</b> ${entry:.2f}</li>
                            <li><b>SL:</b> ${sl:.2f}</li>
                            <li><b>TP:</b> ${tp:.2f}</li>
                            <li><b>RR:</b> {rr:.1f}R</li>
                        </ul>
                        <div style='margin-top:10px; font-size:0.85rem'>
                            🤖 <b>AI:</b> 趨勢向上且回調至折價區，發現 FVG 缺口，建議進場。
                        </div>
                    </div>"""
                else:
                    reason = "無明顯 FVG" if not found_fvg else ("逆勢" if not is_bullish else "價格過高")
                    ai_html = f"""
                    <div class='deploy-box wait'>
                        <div class='deploy-title'>⏳ WAIT (觀望)</div>
                        <ul class='deploy-list'>
                            <li><b>趨勢:</b> {trend_str}</li>
                            <li><b>位置:</b> {"溢價區" if curr_price >= eq else "折價區"}</li>
                            <li><b>原因:</b> {reason}</li>
                        </ul>
                        <div style='margin-top:10px; font-size:0.85rem; color:#aaa'>
                            🤖 條件未滿足，圖表僅供結構參考 (虛線)。
                        </div>
                    </div>"""
                
                APP_DATA[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}

                cards_in_sector += f"""
                <div class="card" onclick="openModal('{t}')">
                    <div class="head"><div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div><span class="badge {cls}">{signal}</span></div>
                    <div class="hint">Tap for Chart ↗</div>
                </div>"""
                
                if signal == "LONG":
                    screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>LONG</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

            except Exception as e:
                print(f"Error {t}: {e}")
                continue
        
        if cards_in_sector:
            sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

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
    .time {{ text-align:center; color:#666; font-size:0.7rem; margin-top:30px; }}
    .chart-lbl {{ color:var(--acc); font-weight:bold; display:block; margin-bottom:5px; font-size:0.9rem; margin-top:10px; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選 (LONG)</div>
            <div class="tab" onclick="setTab('news', this)">📰 Polygon News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content">
            <div style="padding:10px;background:rgba(16,185,129,0.1);margin-bottom:15px;border-radius:6px;font-size:0.9rem">
            🎯 <b>SMC Screener:</b> 顯示符合「多頭 + 折價 + FVG」的股票。
            </div>
            <table><thead><tr><th>Ticker</th><th>Price</th><th>Signal</th><th>Action</th></tr></thead><tbody>{screener_rows if screener_rows else "<tr><td colspan='4' style='text-align:center;padding:20px'>目前無符合條件的股票</td></tr>"}</tbody></table>
        </div>
        <div id="news" class="content">{weekly_news_html}</div>
        
        <div style="text-align:center;color:#666;margin-top:30px;font-size:0.7rem">Market Data by Yahoo | News by Polygon | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

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

