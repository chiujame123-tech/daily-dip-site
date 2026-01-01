import os
import matplotlib
matplotlib.use('Agg') # 強制後台繪圖，修復 GitHub Actions 無圖問題
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

# --- 2. 新聞 (Polygon) ---
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
                pub_time = item.get('published_utc', '')
                try:
                    dt = datetime.strptime(pub_time, "%Y-%m-%dT%H:%M:%SZ")
                    date_str = dt.strftime('%m/%d')
                except: date_str = ""
                news_html += f"<div class='news-item'><div class='news-meta'>{pub} • {date_str}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else: news_html = "<div style='padding:20px'>暫無新聞</div>"
    except: news_html = "News Error"
    return news_html

# --- 3. 技術指標計算 ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 4. 嚴格評分系統 (Strict Scoring) ---
def calculate_quality_score(df, entry, sl, tp, is_bullish):
    """
    滿分 100，起步 60 分。
    根據 RR、RSI、均線乖離率進行加減分，拉開分數差距。
    """
    try:
        score = 60 # 基礎分
        close = df['Close'].iloc[-1]
        
        # 1. 盈虧比 (RR) 權重: 30%
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        if rr >= 3.0: score += 15    # 極佳
        elif rr >= 2.0: score += 10  # 優秀
        elif rr >= 1.5: score += 5   # 普通
        elif rr < 1.0: score -= 20   # 爛賭局 (扣分)

        # 2. RSI 位置 權重: 20%
        rsi = calculate_rsi(df['Close']).iloc[-1]
        if 40 <= rsi <= 55: score += 15 # 完美回調區
        elif rsi > 70: score -= 15      # 過熱 (扣分)
        elif rsi < 30: score -= 5       # 過冷 (可能接刀)

        # 3. 趨勢強度 權重: 20%
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        if close > sma50 > sma200: score += 10 # 多頭排列
        if close < sma50: score -= 5 # 短期轉弱

        # 4. 距離入場點 權重: 30% (越接近 Entry 越好)
        dist_pct = abs(close - entry) / entry
        if dist_pct < 0.01: score += 20   # 就在入場點附近 (狙擊)
        elif dist_pct < 0.03: score += 10 # 稍微偏離
        elif dist_pct > 0.05: score -= 10 # 已經跑掉了

        return min(max(int(score), 0), 99) # 限制 0-99 分
    except:
        return 50

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
        
        for i in range(len(recent)-1, 2, -1):
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
                fvg = float(recent['Low'].iloc[i])
                if fvg < eq:
                    best_entry = fvg
                    found_fvg = True
                    break
        
        return bsl, ssl, eq, best_entry, ssl*0.99, found_fvg
    except:
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False

# --- 6. 繪圖核心 (修復版 - 確保出圖) ---
def generate_chart(df, ticker, title, entry, sl, tp, is_wait):
    try:
        # 清除舊圖
        plt.close('all')
        
        plot_df = df.tail(60)
        if len(plot_df) < 10: return None
        
        # 確保數值 (防止 NaN 崩潰)
        entry = entry if not np.isnan(entry) else plot_df['Close'].iloc[-1]
        sl = sl if not np.isnan(sl) else plot_df['Low'].min()
        tp = tp if not np.isnan(tp) else plot_df['High'].max()

        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#1e293b', facecolor='#0f172a')
        
        # 線條樣式
        line_style = ':' if is_wait else '-'
        alpha_val = 0.6 if is_wait else 0.9
        
        hlines = dict(
            hlines=[tp, entry, sl],
            colors=['#10b981', '#3b82f6', '#ef4444'],
            linewidths=[1, 1, 1],
            linestyle=['-', line_style, '-'],
            alpha=alpha_val
        )
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 簡化文字標註 (防止重疊)
        ax.text(x_min, tp, " TP", color='#10b981', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, entry, " ENTRY", color='#3b82f6', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, sl, " SL", color='#ef4444', fontsize=8, va='top', fontweight='bold')

        # 簡單區域 (不畫複雜 FVG 以免崩潰)
        if not is_wait:
            rect_profit = patches.Rectangle((x_min, entry), x_max-x_min, tp-entry, linewidth=0, facecolor='#10b981', alpha=0.1)
            ax.add_patch(rect_profit)
            rect_loss = patches.Rectangle((x_min, sl), x_max-x_min, entry-sl, linewidth=0, facecolor='#ef4444', alpha=0.1)
            ax.add_patch(rect_loss)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        
    except Exception as e:
        print(f"Chart Error {ticker}: {e}")
        # 回傳一個空的 Base64 圖片 (1x1 像素) 防止前端破圖
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# --- 7. 處理單一股票 ---
def process_ticker(t, app_data_dict, data_d, data_h):
    try:
        try:
            df_d = data_d if isinstance(data_d, pd.DataFrame) else data_d[t]
            df_h = data_h if isinstance(data_h, pd.DataFrame) else data_h[t]
        except: return None
        
        df_d = df_d.dropna()
        df_h = df_h.dropna()
        if len(df_d) < 50: return None

        curr = float(df_d['Close'].iloc[-1])
        sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
        if pd.isna(sma200): sma200 = curr

        # SMC
        bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
        tp = bsl

        # 訊號
        is_bullish = curr > sma200
        in_discount = curr < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # 嚴格評分
        score = calculate_quality_score(df_d, entry, sl, tp, is_bullish)
        
        # 繪圖
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "Daily", entry, sl, tp, is_wait)
        img_h = generate_chart(df_h, t, "Hourly", entry, sl, tp, is_wait)

        # 文案
        cls = "b-long" if signal == "LONG" else "b-wait"
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        score_color = "#10b981" if score >= 80 else ("#fbbf24" if score >= 60 else "#ef4444")
        
        if signal == "LONG":
            ai_html = f"<div class='deploy-box long'><div class='deploy-title'>✅ LONG SETUP</div><div style='display:flex;justify-content:space-between;border-bottom:1px solid #333;padding-bottom:5px;margin-bottom:5px;'><span>🏆 評分: <b style='color:{score_color};font-size:1.1em'>{score}</b></span><span>💰 RR: <b style='color:#10b981'>{rr:.1f}R</b></span></div><ul class='deploy-list'><li>TP: ${tp:.2f}</li><li>Entry: ${entry:.2f}</li><li>SL: ${sl:.2f}</li></ul></div>"
        else:
            reason = "無FVG" if not found_fvg else ("逆勢" if not is_bullish else "溢價區")
            ai_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><div>評分: <b style='color:#94a3b8'>{score}</b></div><ul class='deploy-list'><li>狀態: {reason}</li><li>參考入場: ${entry:.2f}</li></ul></div>"
            
        app_data_dict[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h, "score": score}
        return {"ticker": t, "price": curr, "signal": signal, "cls": cls, "score": score}
    except Exception as e:
        print(f"Err {t}: {e}")
        return None

# --- 8. 主程式 ---
def main():
    print("🚀 Starting Analysis (Strict Score Fix)...")
    
    weekly_news_html = get_polygon_news()

    print("📊 Downloading Data (Yahoo)...")
    try:
        data_d = yf.download(ALL_TICKERS, period="1y", interval="1d", group_by='ticker', progress=False)
        data_h = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    except: return

    APP_DATA, sector_html_blocks, screener_rows_list = {}, "", []
    
    # 處理數據
    for sector, tickers in SECTORS.items():
        cards = ""
        # 這一區塊的股票列表
        sector_results = []
        
        for t in tickers:
            res = process_ticker(t, APP_DATA, data_d, data_h)
            if res:
                sector_results.append(res)
                if res['signal'] == "LONG":
                    screener_rows_list.append(res)
        
        # 排序：分數高的排前面
        sector_results.sort(key=lambda x: x['score'], reverse=True)
        
        for res in sector_results:
            t = res['ticker']
            s_color = "#10b981" if res['score'] >= 80 else "#fbbf24"
            cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><div style='text-align:right'><span class='badge {res['cls']}'>{res['signal']}</span><div style='font-size:0.7rem;color:{s_color};margin-top:2px'>Score: {res['score']}</div></div></div></div>"
            
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    # 處理 Screener 排序 (由高分到低分)
    screener_rows_list.sort(key=lambda x: x['score'], reverse=True)
    screener_html = ""
    for res in screener_rows_list:
        score_cls = "g" if res['score'] >= 80 else ""
        screener_html += f"<tr><td>{res['ticker']}</td><td>${res['price']:.2f}</td><td class='{score_cls}'><b>{res['score']}</b></td><td><span class='badge {res['cls']}'>{res['signal']}</span></td></tr>"

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
    table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
    th, td {{ padding:8px; text-align:left; border-bottom:1px solid #333; }}
    .g {{ color:var(--g); font-weight:bold; }}
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
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選 (Score)</div>
            <div class="tab" onclick="setTab('news', this)">📰 News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Score</th><th>Signal</th></tr></thead><tbody>{screener_html}</tbody></table></div>
        <div id="news" class="content">{weekly_news_html}</div>
        
        <div class="time">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

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
