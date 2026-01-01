import os
import matplotlib
matplotlib.use('Agg') # 1. 強制後台繪圖 (修復無圖問題的關鍵)
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
                dt = item.get('published_utc', '')[:10]
                news_html += f"<div class='news-item'><div class='news-meta'>{pub} • {dt}</div><a href='{url}' target='_blank' class='news-title'>{title}</a></div>"
        else: news_html = "<div style='padding:20px'>暫無新聞</div>"
    except: news_html = "News Error"
    return news_html

# --- 3. 數據獲取 (核心修復：強制清洗 MultiIndex) ---
def fetch_data_safe(ticker, period, interval):
    try:
        # 下載數據
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if df is None or df.empty: return None

        # ⚠️ 關鍵修復：如果欄位是 ('Close', 'AAPL')，強制轉為 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 檢查必要欄位
        required = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required): return None
            
        return df
    except Exception as e:
        print(f"Download Error {ticker}: {e}")
        return None

# --- 4. 技術指標與評分 (含詳解生成) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_quality_score(df, entry, sl, tp, is_bullish):
    """
    回傳: (分數, 理由列表)
    """
    try:
        score = 60
        reasons = [] # 這裡儲存加分理由
        close = df['Close'].iloc[-1]
        
        # RR (盈虧比)
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        if rr >= 3.0: 
            score += 15
            reasons.append(f"💰 盈虧比極佳 ({rr:.1f}R)")
        elif rr >= 2.0: 
            score += 10
            reasons.append(f"💰 盈虧比優秀 ({rr:.1f}R)")
        elif rr < 1.0: 
            score -= 20
            reasons.append("⚠️ 盈虧比過低 (<1R)")

        # RSI (相對強弱)
        rsi = calculate_rsi(df['Close']).iloc[-1]
        if 40 <= rsi <= 55: 
            score += 15
            reasons.append(f"📉 RSI 黃金回調位 ({int(rsi)})")
        elif rsi > 70: 
            score -= 15
            reasons.append("⚠️ RSI 過熱 (>70)")

        # 趨勢
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        if close > sma50 > sma200: 
            score += 10
            reasons.append("📈 多頭排列強勢 (價格 > 50MA > 200MA)")
        if close < sma50: 
            score -= 5

        # 距離
        dist_pct = abs(close - entry) / entry
        if dist_pct < 0.01: 
            score += 20
            reasons.append("🎯 價格位於狙擊入場區 (Spread < 1%)")
        elif dist_pct < 0.03: 
            score += 10

        return min(max(int(score), 0), 99), reasons
    except: return 50, []

# --- 5. SMC 運算 (含 FVG 列表) ---
def calculate_smc(df):
    try:
        window = 50
        recent = df.tail(window)
        bsl = float(recent['High'].max())
        ssl = float(recent['Low'].min())
        eq = (bsl + ssl) / 2
        best_entry = eq
        found_fvg = False
        fvg_list = []
        
        for i in range(2, len(recent)):
            # Bullish FVG
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
                gap_top = float(recent['Low'].iloc[i])
                gap_bot = float(recent['High'].iloc[i-2])
                fvg_list.append({'idx': i-1, 'top': gap_top, 'bot': gap_bot, 'type': 'bull'})
                if gap_top < eq:
                    best_entry = gap_top
                    found_fvg = True
            # Bearish FVG
            elif recent['High'].iloc[i] < recent['Low'].iloc[i-2]:
                gap_top = float(recent['Low'].iloc[i-2])
                gap_bot = float(recent['High'].iloc[i])
                fvg_list.append({'idx': i-1, 'top': gap_top, 'bot': gap_bot, 'type': 'bear'})

        return bsl, ssl, eq, best_entry, ssl*0.99, found_fvg, fvg_list
    except:
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False, []

# --- 6. 繪圖核心 (Matplotlib PNG 模式) ---
def generate_chart(df, ticker, title, entry, sl, tp, fvg_list, is_wait):
    try:
        plt.close('all') # 重要：清除舊圖
        plot_df = df.tail(60)
        if len(plot_df) < 10: return None
        
        # 確保數值
        entry = entry if not np.isnan(entry) else plot_df['Close'].iloc[-1]
        sl = sl if not np.isnan(sl) else plot_df['Low'].min()
        tp = tp if not np.isnan(tp) else plot_df['High'].max()

        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#1e293b', facecolor='#0f172a')
        
        # 繪製 K 線
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 繪製 FVG 色塊 (這就是你想要的 SMC 指標)
        for fvg in fvg_list:
            plot_idx_start = len(df) - 60 
            rel_idx = fvg['idx'] - plot_idx_start
            if 0 <= rel_idx < 60:
                color = '#10b981' if fvg['type'] == 'bull' else '#ef4444'
                rect = patches.Rectangle((rel_idx, fvg['bot']), x_max - rel_idx, fvg['top'] - fvg['bot'],
                                         linewidth=0, facecolor=color, alpha=0.25)
                ax.add_patch(rect)

        # 繪製 Entry/SL/TP 線條
        line_style = ':' if is_wait else '-'
        ax.axhline(tp, color='#10b981', linestyle=line_style, linewidth=1)
        ax.axhline(entry, color='#3b82f6', linestyle=line_style, linewidth=1)
        ax.axhline(sl, color='#ef4444', linestyle=line_style, linewidth=1)
        
        # 文字標註
        ax.text(x_min, tp, " TP", color='#10b981', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, entry, " ENTRY", color='#3b82f6', fontsize=8, va='bottom', fontweight='bold')
        ax.text(x_min, sl, " SL", color='#ef4444', fontsize=8, va='top', fontweight='bold')

        # 轉成 Base64 圖片
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=80)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        print(f"Chart fail {ticker}: {e}")
        return None

# --- 7. 單一股票處理 (含 >90分 解釋邏輯) ---
def process_ticker(t, app_data_dict):
    try:
        time.sleep(0.5) # 避免 Rate Limit
        
        # 下載
        df_d = fetch_data_safe(t, "1y", "1d")
        if df_d is None or len(df_d) < 50: return None

        df_h = fetch_data_safe(t, "1mo", "1h")
        if df_h is None or df_h.empty: df_h = df_d

        curr = float(df_d['Close'].iloc[-1])
        sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
        if pd.isna(sma200): sma200 = curr

        # SMC
        bsl, ssl, eq, entry, sl, found_fvg, fvg_list_d = calculate_smc(df_d)
        _, _, _, _, _, _, fvg_list_h = calculate_smc(df_h)
        tp = bsl

        # 訊號
        is_bullish = curr > sma200
        in_discount = curr < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # 評分與詳細理由 (reasons)
        score, reasons = calculate_quality_score(df_d, entry, sl, tp, is_bullish)
        
        # 繪圖
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "Daily SMC", entry, sl, tp, fvg_list_d, is_wait)
        img_h = generate_chart(df_h, t, "Hourly Entry", entry, sl, tp, fvg_list_h, is_wait)
        
        img_d = img_d if img_d else ""
        img_h = img_h if img_h else ""

        # 文案生成
        cls = "b-long" if signal == "LONG" else "b-wait"
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        score_color = "#10b981" if score >= 90 else ("#3b82f6" if score >= 80 else "#fbbf24")
        
        # 🌟 90分以上詳解功能 (恢復)
        elite_html = ""
        if score >= 90:
            reasons_html = "".join([f"<li>✅ {r}</li>" for r in reasons])
            elite_html = f"""
            <div style='background:rgba(16,185,129,0.1); border:1px solid #10b981; padding:10px; border-radius:6px; margin:10px 0;'>
                <div style='font-weight:bold; color:#10b981; margin-bottom:5px;'>💎 為什麼值得入手？</div>
                <ul style='margin:0; padding-left:20px; font-size:0.85rem; color:#d1d5db;'>
                    {reasons_html}
                </ul>
            </div>
            """
        
        if signal == "LONG":
            ai_html = f"""
            <div class='deploy-box long'>
                <div class='deploy-title'>✅ LONG SETUP</div>
                <div style='display:flex;justify-content:space-between;border-bottom:1px solid #333;padding-bottom:5px;margin-bottom:5px;'>
                    <span>🏆 評分: <b style='color:{score_color};font-size:1.1em'>{score}</b></span>
                    <span>💰 RR: <b style='color:#10b981'>{rr:.1f}R</b></span>
                </div>
                {elite_html}
                <ul class='deploy-list' style='margin-top:10px'>
                    <li>TP: ${tp:.2f}</li><li>Entry: ${entry:.2f}</li><li>SL: ${sl:.2f}</li>
                </ul>
            </div>"""
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
    print("🚀 Starting Analysis (Stable Image + 90+ Analysis)...")
    weekly_news_html = get_polygon_news()
    
    APP_DATA, sector_html_blocks, screener_rows_list = {}, "", []
    
    for sector, tickers in SECTORS.items():
        cards = ""
        sector_results = []
        for t in tickers:
            res = process_ticker(t, APP_DATA)
            if res:
                sector_results.append(res)
                if res['signal'] == "LONG":
                    screener_rows_list.append(res)
        
        # 排序
        sector_results.sort(key=lambda x: x['score'], reverse=True)
        
        for res in sector_results:
            t = res['ticker']
            s_color = "#10b981" if res['score'] >= 90 else ("#3b82f6" if res['score'] >= 80 else "#fbbf24")
            cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><div style='text-align:right'><span class='badge {res['cls']}'>{res['signal']}</span><div style='font-size:0.7rem;color:{s_color};margin-top:2px'>Score: {res['score']}</div></div></div></div>"
            
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    # Screener 表格
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
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選 (LONG)</div>
            <div class="tab" onclick="setTab('news', this)">📰 News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks if sector_html_blocks else '<div style="text-align:center;padding:50px">載入中...</div>'}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Score</th><th>Signal</th></tr></thead><tbody>{screener_html}</tbody></table></div>
        <div id="news" class="content">{weekly_news_html}</div>
        
        <div class="time">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><b>Daily SMC (Green Box=FVG)</b><div id="chart-d"></div></div>
                <div><b>Hourly Entry</b><div id="chart-h"></div></div>
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
