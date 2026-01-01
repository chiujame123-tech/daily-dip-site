import os
import matplotlib
# 1. 強制設定後台繪圖 (最優先執行)
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

# --- 3. 市場大盤分析 (Market Filter) ---
def get_market_condition():
    """分析 SPY 和 QQQ 的趨勢，決定市場紅綠燈"""
    try:
        print("🔍 Analyzing Market Sentiment (SPY/QQQ)...")
        tickers = ["SPY", "QQQ"]
        df = yf.download(tickers, period="6mo", interval="1d", progress=False)
        
        # 處理 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            spy_close = df['Close']['SPY']
            qqq_close = df['Close']['QQQ']
        else:
            return "NEUTRAL", "數據格式錯誤", 0

        # 計算 SPY 50MA
        spy_50 = spy_close.rolling(50).mean().iloc[-1]
        spy_curr = spy_close.iloc[-1]
        
        # 計算 QQQ 50MA
        qqq_50 = qqq_close.rolling(50).mean().iloc[-1]
        qqq_curr = qqq_close.iloc[-1]
        
        is_bullish = (spy_curr > spy_50) and (qqq_curr > qqq_50)
        is_bearish = (spy_curr < spy_50) and (qqq_curr < qqq_50)
        
        if is_bullish:
            return "BULLISH", "🟢 市場順風 (大盤 > 50MA)", 5 # 加分
        elif is_bearish:
            return "BEARISH", "🔴 市場逆風 (大盤 < 50MA)", -10 # 扣分
        else:
            return "NEUTRAL", "🟡 市場震盪", 0
            
    except Exception as e:
        print(f"Market analysis failed: {e}")
        return "NEUTRAL", "市場數據獲取失敗", 0

# --- 4. 數據獲取 (單一股票) ---
def fetch_data_safe(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required): return None
        return df
    except: return None

# --- 5. 技術指標 ---
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # RVOL (相對成交量)
    vol_ma = df['Volume'].rolling(10).mean()
    rvol = df['Volume'] / vol_ma
    
    return rsi, rvol

# --- 6. 評分系統 (加入 RVOL 和市場因子) ---
def calculate_quality_score(df, entry, sl, tp, is_bullish, market_bonus):
    try:
        score = 60 + market_bonus # 基礎分 + 市場加權
        reasons = []
        close = df['Close'].iloc[-1]
        
        # 1. 盈虧比 (RR)
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
            reasons.append("⚠️ 盈虧比過低")

        # 2. RSI (回調程度)
        rsi = calculate_indicators(df)[0].iloc[-1]
        if 40 <= rsi <= 55: 
            score += 15
            reasons.append(f"📉 RSI 完美回調 ({int(rsi)})")
        elif rsi > 70: 
            score -= 15
            reasons.append("⚠️ RSI 過熱")

        # 3. RVOL (成交量確認) - 新功能
        rvol = calculate_indicators(df)[1].iloc[-1]
        if rvol > 1.5:
            score += 10
            reasons.append(f"🔥 爆量確認 (RVOL {rvol:.1f}x)")
        elif rvol > 1.1:
            score += 5
            reasons.append("📊 量能溫和放大")

        # 4. 趨勢
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        if close > sma50 > sma200: 
            score += 10
            reasons.append("📈 強力多頭排列")
        if close < sma50: score -= 5

        # 5. 距離
        dist_pct = abs(close - entry) / entry
        if dist_pct < 0.01: 
            score += 15
            reasons.append("🎯 狙擊入場區")

        # 市場理由
        if market_bonus > 0: reasons.append("🌍 大盤順風車 (+5)")
        if market_bonus < 0: reasons.append("🌪️ 逆大盤風險 (-10)")

        return min(max(int(score), 0), 99), reasons, rr, rvol
    except: return 50, [], 0, 0

# --- 7. SMC 運算 ---
def calculate_smc(df):
    try:
        window = 50
        recent = df.tail(window)
        bsl = float(recent['High'].max())
        ssl = float(recent['Low'].min())
        eq = (bsl + ssl) / 2
        best_entry = eq
        found_fvg = False
        
        for i in range(2, len(recent)):
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

# --- 8. 繪圖核心 ---
def create_error_image(msg):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.text(0.5, 0.5, msg, color='white', ha='center', va='center')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def generate_chart(df, ticker, title, entry, sl, tp, is_wait):
    try:
        plt.close('all')
        if df is None or len(df) < 5: return create_error_image("No Data")
        plot_df = df.tail(60).copy()
        
        entry = float(entry) if not np.isnan(entry) else plot_df['Close'].iloc[-1]
        sl = float(sl) if not np.isnan(sl) else plot_df['Low'].min()
        tp = float(tp) if not np.isnan(tp) else plot_df['High'].max()

        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#1e293b', facecolor='#0f172a')
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        for i in range(2, len(plot_df)):
            idx = i - 1
            if plot_df['Low'].iloc[i] > plot_df['High'].iloc[i-2]: # Bullish
                bot, top = plot_df['High'].iloc[i-2], plot_df['Low'].iloc[i]
                rect = patches.Rectangle((idx, bot), x_max - idx, top - bot, linewidth=0, facecolor='#10b981', alpha=0.25)
                ax.add_patch(rect)
            elif plot_df['High'].iloc[i] < plot_df['Low'].iloc[i-2]: # Bearish
                bot, top = plot_df['High'].iloc[i], plot_df['Low'].iloc[i-2]
                rect = patches.Rectangle((idx, bot), x_max - idx, top - bot, linewidth=0, facecolor='#ef4444', alpha=0.25)
                ax.add_patch(rect)

        line_style = ':' if is_wait else '-'
        ax.axhline(tp, color='#10b981', linestyle=line_style, linewidth=1)
        ax.axhline(entry, color='#3b82f6', linestyle=line_style, linewidth=1)
        ax.axhline(sl, color='#ef4444', linestyle=line_style, linewidth=1)
        
        if not is_wait:
            ax.add_patch(patches.Rectangle((x_min, entry), x_max-x_min, tp-entry, linewidth=0, facecolor='#10b981', alpha=0.1))
            ax.add_patch(patches.Rectangle((x_min, sl), x_max-x_min, entry-sl, linewidth=0, facecolor='#ef4444', alpha=0.1))

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=80)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        return create_error_image(f"Plot Error")

# --- 9. 單一股票處理 ---
def process_ticker(t, app_data_dict, market_bonus):
    try:
        time.sleep(0.5)
        df_d = fetch_data_safe(t, "1y", "1d")
        if df_d is None or len(df_d) < 50: return None
        df_h = fetch_data_safe(t, "1mo", "1h")
        if df_h is None or df_h.empty: df_h = df_d

        curr = float(df_d['Close'].iloc[-1])
        sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
        if pd.isna(sma200): sma200 = curr

        bsl, ssl, eq, entry, sl, found_fvg = calculate_smc(df_d)
        tp = bsl

        is_bullish = curr > sma200
        in_discount = curr < eq
        signal = "LONG" if (is_bullish and in_discount and found_fvg) else "WAIT"
        
        # 傳入市場分數
        score, reasons, rr, rvol = calculate_quality_score(df_d, entry, sl, tp, is_bullish, market_bonus)
        
        is_wait = (signal == "WAIT")
        img_d = generate_chart(df_d, t, "Daily SMC", entry, sl, tp, is_wait)
        img_h = generate_chart(df_h, t, "Hourly Entry", entry, sl, tp, is_wait)

        cls = "b-long" if signal == "LONG" else "b-wait"
        score_color = "#10b981" if score >= 85 else ("#3b82f6" if score >= 70 else "#fbbf24")
        
        # 💎 85分以上詳解邏輯 (更新)
        elite_html = ""
        if score >= 85:
            reasons_html = "".join([f"<li>{r}</li>" for r in reasons])
            
            # 根據 RVOL 和市場狀況生成的動態評語
            ai_comment = "此股表現強勁，"
            if rvol > 1.2: ai_comment += "且今日成交量明顯放大 (有大資金)，"
            if market_bonus > 0: ai_comment += "加上大盤順風，"
            ai_comment += "建議優先關注。"
            
            elite_html = f"""
            <div style='background:rgba(16,185,129,0.1); border:1px solid #10b981; padding:12px; border-radius:8px; margin:10px 0;'>
                <div style='font-weight:bold; color:#10b981; margin-bottom:5px; display:flex; align-items:center;'>
                    💎 AI 戰略分析 (Score {score})
                </div>
                <div style='color:#e2e8f0; font-size:0.9rem; margin-bottom:8px;'>
                    {ai_comment}
                </div>
                <div style='font-size:0.8rem; color:#94a3b8; font-weight:bold;'>得分關鍵：</div>
                <ul style='margin:0; padding-left:20px; font-size:0.8rem; color:#d1d5db;'>
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
        return {"ticker": t, "price": curr, "signal": signal, "cls": cls, "score": score, "rvol": rvol}
    except Exception as e:
        print(f"Err {t}: {e}")
        return None

# --- 10. 主程式 ---
def main():
    print("🚀 Starting Analysis (Market Filter + RVOL)...")
    weekly_news_html = get_polygon_news()
    
    # 1. 先抓大盤狀態
    market_status, market_text, market_bonus = get_market_condition()
    market_color = "#10b981" if market_status == "BULLISH" else ("#ef4444" if market_status == "BEARISH" else "#fbbf24")
    
    print(f"🌍 Market: {market_status} ({market_bonus})")
    
    APP_DATA, sector_html_blocks, screener_rows_list = {}, "", []
    
    for sector, tickers in SECTORS.items():
        cards = ""
        sector_results = []
        for t in tickers:
            res = process_ticker(t, APP_DATA, market_bonus)
            if res:
                sector_results.append(res)
                if res['signal'] == "LONG":
                    screener_rows_list.append(res)
        
        sector_results.sort(key=lambda x: x['score'], reverse=True)
        
        for res in sector_results:
            t = res['ticker']
            s_color = "#10b981" if res['score'] >= 85 else ("#3b82f6" if res['score'] >= 70 else "#fbbf24")
            # 在卡片上也顯示 RVOL
            rvol_tag = f"<span style='font-size:0.7rem;color:#f472b6;margin-right:5px'>Vol {res['rvol']:.1f}x</span>" if res['rvol'] > 1.2 else ""
            
            cards += f"<div class='card' onclick=\"openModal('{t}')\"><div class='head'><div><div class='code'>{t}</div><div class='price'>${res['price']:.2f}</div></div><div style='text-align:right'><span class='badge {res['cls']}'>{res['signal']}</span><div style='margin-top:2px'>{rvol_tag}<span style='font-size:0.7rem;color:{s_color}'>{res['score']}</span></div></div></div></div>"
            
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    screener_rows_list.sort(key=lambda x: x['score'], reverse=True)
    screener_html = ""
    for res in screener_rows_list:
        score_cls = "g" if res['score'] >= 85 else ""
        vol_fire = "🔥" if res['rvol'] > 1.5 else ""
        screener_html += f"<tr><td>{res['ticker']}</td><td>${res['price']:.2f}</td><td class='{score_cls}'><b>{res['score']}</b> {vol_fire}</td><td><span class='badge {res['cls']}'>{res['signal']}</span></td></tr>"

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
    .market-bar {{ background: #1e293b; padding: 10px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #333; display: flex; align-items: center; gap: 10px; }}
    </style>
    </head>
    <body>
        <div class="market-bar" style="border-left: 4px solid {market_color}">
            <div style="font-size:1.2rem;">{ "🟢" if market_status=="BULLISH" else ("🔴" if market_status=="BEARISH" else "🟡") }</div>
            <div>
                <div style="font-weight:bold; color:{market_color}">Market: {market_status}</div>
                <div style="font-size:0.8rem; color:#94a3b8">{market_text}</div>
            </div>
        </div>

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
