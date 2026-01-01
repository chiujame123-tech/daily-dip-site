import os
import matplotlib
matplotlib.use('Agg') # 強制後台繪圖，防止 GitHub Actions 報錯
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
    "🔥 熱門交易": ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "SOFI", "MARA", "MSTR"],
    "💎 科技巨頭": ["MSFT", "AMZN", "GOOGL", "META", "NFLX"],
    "⚡ 半導體": ["TSM", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI", "SOXL"],
    "🚀 成長股": ["COIN", "HOOD", "DKNG", "RBLX", "U", "CVNA", "OPEN", "SHOP"],
    "🏦 金融與消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT", "DIS"],
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

# --- 3. SMC 核心運算 (含 FVG 列表) ---
def calculate_smc_details(df):
    """
    計算 Entry, SL, TP 並回傳所有發現的 FVG 列表供繪圖使用。
    """
    try:
        window = 50
        recent = df.tail(window)
        
        bsl = float(recent['High'].max()) # TP
        ssl = float(recent['Low'].min())  # SL
        eq = (bsl + ssl) / 2       # 平衡點
        
        best_entry = eq
        found_fvg = False
        
        # 儲存所有 FVG 用於繪圖 [{'idx': 10, 'top': 100, 'bot': 90, 'type': 'bull'}]
        fvg_list = []
        
        # 遍歷尋找 FVG
        # 這裡使用相對索引，因為 mplfinance 繪圖是用 0,1,2...
        for i in range(2, len(recent)):
            # Bullish FVG
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
                gap_top = float(recent['Low'].iloc[i])
                gap_bot = float(recent['High'].iloc[i-2])
                fvg_list.append({'idx': i-1, 'top': gap_top, 'bot': gap_bot, 'type': 'bull'})
                
                # 如果這個缺口在折價區，選它做 Entry
                if gap_top < eq:
                    best_entry = gap_top
                    found_fvg = True
            
            # Bearish FVG (僅供繪圖參考)
            elif recent['High'].iloc[i] < recent['Low'].iloc[i-2]:
                gap_top = float(recent['Low'].iloc[i-2])
                gap_bot = float(recent['High'].iloc[i])
                fvg_list.append({'idx': i-1, 'top': gap_top, 'bot': gap_bot, 'type': 'bear'})

        sl_price = ssl * 0.99
        return bsl, ssl, eq, best_entry, sl_price, found_fvg, fvg_list
    except:
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False, []

# --- 4. 計算勝率評分 (Quant Score) ---
def calculate_win_score(df, is_bullish, in_discount, has_fvg):
    score = 50 # 基礎分
    
    # 趨勢加分
    close = df['Close'].iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    sma200 = df['Close'].rolling(200).mean().iloc[-1]
    
    if close > sma200: score += 15 # 長期多頭
    if close > sma50: score += 10  # 中期多頭
    if is_bullish: score += 5
    
    # 位置加分
    if in_discount: score += 10
    if has_fvg: score += 10
    
    # 動能加分 (簡單 RSI 模擬)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    if 40 < rsi < 65: score += 5 # 健康回調區間
    
    return min(score, 95) # 上限 95

# --- 5. 繪圖核心 (增強版) ---
def generate_chart(df, ticker, title, entry, sl, tp, fvg_list, is_wait):
    try:
        plt.close('all')
        plot_df = df.tail(50) # 只畫最後 50 根
        if len(plot_df) < 10: return None
        
        # 確保數值
        entry = entry if not np.isnan(entry) else plot_df['Close'].iloc[-1]
        sl = sl if not np.isnan(sl) else plot_df['Low'].min()
        tp = tp if not np.isnan(tp) else plot_df['High'].max()
        
        # 風格設定
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#1e293b', facecolor='#0f172a')
        
        # 1. 繪製 K 線
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 2. 繪製 FVG 矩形 (最重要的新功能)
        # mplfinance 的 X 軸是 0 到 len(df)，我們需要轉換 index
        for fvg in fvg_list:
            # 確保 FVG 索引在當前繪圖範圍內
            # 我們畫的是 tail(50)，所以原始 df 的 index 要轉換為 0-49
            plot_idx_start = len(df) - 50
            rel_idx = fvg['idx'] - plot_idx_start
            
            if 0 <= rel_idx < 50:
                color = '#10b981' if fvg['type'] == 'bull' else '#ef4444'
                # 畫出延伸到右邊的矩形
                rect = patches.Rectangle((rel_idx, fvg['bot']), x_max - rel_idx, fvg['top'] - fvg['bot'],
                                         linewidth=0, facecolor=color, alpha=0.25)
                ax.add_patch(rect)

        # 3. 繪製 Entry/SL/TP 線與背景色 (RR 可視化)
        if not is_wait:
            # 獲利區間 (綠色背景)
            rect_profit = patches.Rectangle((x_min, entry), x_max-x_min, tp-entry, linewidth=0, facecolor='#10b981', alpha=0.1)
            ax.add_patch(rect_profit)
            # 虧損區間 (紅色背景)
            rect_loss = patches.Rectangle((x_min, sl), x_max-x_min, entry-sl, linewidth=0, facecolor='#ef4444', alpha=0.1)
            ax.add_patch(rect_loss)

        # 畫線
        line_style = ':' if is_wait else '-'
        ax.axhline(tp, color='#10b981', linestyle=line_style, linewidth=1)
        ax.axhline(entry, color='#3b82f6', linestyle=line_style, linewidth=1)
        ax.axhline(sl, color='#ef4444', linestyle=line_style, linewidth=1)

        # 文字標籤
        ax.text(x_min, tp, f" TP: {tp:.2f}", color='#10b981', fontsize=7, va='bottom', fontweight='bold')
        ax.text(x_min, entry, f" ENTRY: {entry:.2f}", color='#3b82f6', fontsize=7, va='bottom', fontweight='bold')
        ax.text(x_min, sl, f" SL: {sl:.2f}", color='#ef4444', fontsize=7, va='top', fontweight='bold')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=80)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        print(f"Chart Error {ticker}: {e}")
        return None

# --- 6. 處理邏輯 ---
def process_ticker(t, app_data_dict, data_d, data_h):
    try:
        # 提取
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

        # SMC 計算 (含 FVG)
        bsl, ssl, eq, entry, sl, found_fvg, fvg_list_d = calculate_smc_details(df_d)
        _, _, _, _, _, _, fvg_list_h = calculate_smc_details(df_h) # 也要算小時線的 FVG
        tp = bsl

        # 訊號
        is_bullish = curr > sma200
        in_discount = curr < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # 分數計算
        win_score = calculate_win_score(df_d, is_bullish, in_discount, found_fvg)
        
        # 繪圖
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "Daily SMC", entry, sl, tp, fvg_list_d, is_wait)
        img_h = generate_chart(df_h, t, "Hourly Entry", entry, sl, tp, fvg_list_h, is_wait)
        
        if not img_d: img_d = ""
        if not img_h: img_h = ""

        # AI 文案
        cls = "b-long" if signal == "LONG" else "b-wait"
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        
        # 顏色設定
        score_color = "#10b981" if win_score >= 70 else "#fbbf24"
        
        if signal == "LONG":
            ai_html = f"""
            <div class='deploy-box long'>
                <div class='deploy-title'>✅ LONG SETUP (做多建議)</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                    <span>🏆 勝率評分: <b style="color:{score_color}">{win_score}</b>/100</span>
                    <span>💰 盈虧比: <b style="color:#10b981">{rr:.2f}R</b></span>
                </div>
                <ul class='deploy-list'>
                    <li><b>🎯 目標 (TP):</b> ${tp:.2f} (BSL)</li>
                    <li><b>🔵 入場 (Entry):</b> ${entry:.2f} (FVG)</li>
                    <li><b>🛑 止損 (SL):</b> ${sl:.2f}</li>
                </ul>
                <div style='margin-top:10px; font-size:0.85rem; line-height:1.4;'>
                    <b>SMC 分析:</b> 股價回調至折價區，並在支撐位出現機構 FVG 缺口 (圖中綠色區塊)，具備高勝率反轉條件。
                </div>
            </div>"""
        else:
            reason = "無FVG" if not found_fvg else ("逆勢" if not is_bullish else "溢價區")
            ai_html = f"""
            <div class='deploy-box wait'>
                <div class='deploy-title'>⏳ WAIT (觀望)</div>
                <div style="margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                    <span>趨勢評分: <b style="color:#94a3b8">{win_score}</b>/100</span>
                </div>
                <ul class='deploy-list'>
                    <li><b>目前狀態:</b> {reason}</li>
                    <li><b>參考入場:</b> ${entry:.2f}</li>
                </ul>
                <div style='margin-top:10px; font-size:0.85rem; color:#aaa'>
                    圖中綠色/紅色區塊為 FVG 缺口。目前條件未滿足，請耐心等待。
                </div>
            </div>"""
            
        app_data_dict[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}
        return {"ticker": t, "price": curr, "signal": signal, "cls": cls}
    except Exception as e:
        print(f"Err {t}: {e}")
        return None

# --- 7. 主程式 ---
def main():
    print("🚀 Starting SMC Visual Pro...")
    
    weekly_news_html = get_polygon_news()

    print("📊 Downloading Data...")
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
                cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><span class='badge {res['cls']}'>{res['signal']}</span></div><div class='hint'>Tap for SMC Chart ↗</div></div>"
                if res['signal'] == "LONG":
                    screener_rows += f"<tr><td>{t}</td><td>${res['price']:.2f}</td><td class='g'>LONG</td><td><span class='badge {res['cls']}'>{res['signal']}</span></td></tr>"
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    json_data = json.dumps(APP_DATA)
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DailyDip SMC Pro</title>
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
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選 (LONG)</div>
            <div class="tab" onclick="setTab('news', this)">📰 Polygon News</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Signal</th><th>Action</th></tr></thead><tbody>{screener_rows}</tbody></table></div>
        <div id="news" class="content">{weekly_news_html}</div>
        
        <div style="text-align:center;color:#666;margin-top:30px;font-size:0.7rem">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><b>Daily Structure (Green Box = FVG)</b><div id="chart-d"></div></div>
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
