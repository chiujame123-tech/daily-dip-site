import os
import matplotlib
# ⚠️ 關鍵修正：設定為非互動模式，專門給 GitHub Actions 用
matplotlib.use('Agg') 
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

# --- 0. 讀取 API KEY ---
API_KEY = os.environ.get("POLYGON_API_KEY")

# --- 1. 設定觀察清單 ---
SECTORS = {
    "🔥 熱門交易": ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "SOFI", "MARA"],
    "💎 科技巨頭": ["MSFT", "AMZN", "GOOGL", "META", "NFLX"],
    "⚡ 半導體": ["TSM", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI", "SOXL"],
    "🚀 成長股": ["COIN", "MSTR", "HOOD", "DKNG", "RBLX", "U", "CVNA"],
    "🏦 金融消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT"],
    "📉 指數": ["SPY", "QQQ", "IWM", "TQQQ", "SQQQ"]
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 新聞 (Polygon) ---
def get_polygon_news():
    if not API_KEY: return "<div style='padding:20px'>API Key Missing</div>"
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
                    date_str = dt.strftime('%m/%d')
                except: date_str = ""
                news_html += f"<div class='news-item'><div class='news-meta'>{pub} • {date_str}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else: news_html = "<div style='padding:20px'>暫無新聞</div>"
    except: news_html = "News Error"
    return news_html

# --- 3. SMC 運算 ---
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

# --- 4. 繪圖核心 (修復版) ---
def generate_chart(df, ticker, title, entry, sl, tp, is_wait):
    try:
        # 確保清理舊圖表，釋放記憶體
        plt.close('all')
        
        plot_df = df.tail(60)
        if len(plot_df) < 10: return None
        
        # 確保數值安全 (避免 NaN)
        entry = entry if not np.isnan(entry) else plot_df['Close'].iloc[-1]
        sl = sl if not np.isnan(sl) else plot_df['Low'].min()
        tp = tp if not np.isnan(tp) else plot_df['High'].max()
        eq = (plot_df['High'].max() + plot_df['Low'].min()) / 2

        # 設定風格
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        # 線條設定
        line_style = ':' if is_wait else '--'
        alpha_val = 0.5 if is_wait else 0.9
        
        hlines = dict(
            hlines=[tp, entry, sl],
            colors=['#10b981', '#3b82f6', '#ef4444'],
            linewidths=[1.5, 1.5, 1.5],
            linestyle=['-', line_style, '-'],
            alpha=alpha_val
        )
        
        # 繪圖
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 標註
        ax.text(x_min, tp, f" TP {tp:.2f}", color='#10b981', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, entry, f" ENTRY {entry:.2f}", color='#3b82f6', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, sl, f" SL {sl:.2f}", color='#ef4444', fontsize=8, va='top', fontweight='bold')
        
        # 區域
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, tp-eq, linewidth=0, facecolor='#ef4444', alpha=0.05)
        ax.add_patch(rect_prem)
        rect_disc = patches.Rectangle((x_min, sl), x_max-x_min, eq-sl, linewidth=0, facecolor='#10b981', alpha=0.05)
        ax.add_patch(rect_disc)

        # 存檔轉碼
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig) # 畫完馬上關閉
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Plot Error {ticker}: {e}")
        return None

# --- 5. 處理單一股票 ---
def process_ticker(t, app_data_dict, data_d, data_h):
    try:
        # 提取數據
        try:
            df_d = data_d if isinstance(data_d, pd.DataFrame) and 'Close' in data_d.columns else data_d[t]
            df_h = data_h if isinstance(data_h, pd.DataFrame) and 'Close' in data_h.columns else data_h[t]
        except: return None
        
        df_d = df_d.dropna()
        df_h = df_h.dropna()
        if len(df_d) < 50: return None

        curr_price = float(df_d['Close'].iloc[-1])
        sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
        if pd.isna(sma200): sma200 = curr_price

        # SMC
        bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
        tp = bsl

        # 訊號
        is_bullish = curr_price > sma200
        in_discount = curr_price < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # 畫圖
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "Daily", entry, sl, tp, is_wait)
        img_h = generate_chart(df_h, t, "Hourly", entry, sl, tp, is_wait)
        
        # 確保有圖片字串 (如果是 None 則給空字串)
        img_d = img_d if img_d else ""
        img_h = img_h if img_h else ""

        # AI
        cls = "b-long" if signal == "LONG" else "b-wait"
        rr = (tp-entry)/(entry-sl) if (entry-sl)>0 else 0
        
        if signal == "LONG":
            ai_html = f"<div class='deploy-box long'><div class='deploy-title'>✅ LONG SETUP</div><ul class='deploy-list'><li>Entry: ${entry:.2f}</li><li>SL: ${sl:.2f}</li><li>TP: ${tp:.2f}</li><li>RR: {rr:.1f}R</li></ul><div style='margin-top:5px;font-size:0.8rem'>AI: 趨勢向上 + FVG 確認。</div></div>"
        else:
            reason = "無FVG" if not found_fvg else ("逆勢" if not is_bullish else "溢價區")
            ai_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><ul class='deploy-list'><li>狀態: {reason}</li><li>參考入場: ${entry:.2f}</li></ul></div>"
            
        app_data_dict[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}
        return {"ticker": t, "price": curr_price, "signal": signal, "cls": cls}
    except Exception as e:
        print(f"Err {t}: {e}")
        return None

# --- 6. 主程式 ---
def main():
    print("🚀 Starting Analysis (Headless Mode)...")
    
    weekly_news_html = get_polygon_news()

    print("📊 Downloading Data (Yahoo)...")
    try:
        data_d = yf.download(ALL_TICKERS, period="1y", interval="1d", group_by='ticker', progress=False)
        data_h = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    except: return

    APP_DATA, sector_html_blocks, screener_rows = {}, "", ""
    
    for sector, tickers in SECTORS.items():
        cards = ""
        for t in tickers:
            res = process_ticker(t, APP_DATA, data_d, data_h)
            if res:
                cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><span class='badge {res['cls']}'>{res['signal']}</span></div><div class='hint'>Tap for Chart ↗</div></div>"
                if res['signal'] == "LONG":
                    screener_rows += f"<tr><td>{t}</td><td>${res['price']:.2f}</td><td class='g'>LONG</td><td><span class='badge {res['cls']}'>{res['signal']}</span></td></tr>"
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

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
            <div class="tab" onclick="setTab('screener', this)">🔍 Screener</div>
            <div class="tab" onclick="setTab('news', this)">📰 News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Signal</th><th>Action</th></tr></thead><tbody>{screener_rows}</tbody></table></div>
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
            
            // 這裡做了雙重保險：如果圖片字串存在才顯示，否則顯示 No Chart Data
            document.getElementById('chart-d').innerHTML = data.img_d ? '<img src="'+data.img_d+'">' : '<div style="padding:20px;text-align:center;color:#666">No Chart Available</div>';
            document.getElementById('chart-h').innerHTML = data.img_h ? '<img src="'+data.img_h+'">' : '<div style="padding:20px;text-align:center;color:#666">No Chart Available</div>';
        }}
        </script>
    </body></html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated!")

if __name__ == "__main__":
    main()
