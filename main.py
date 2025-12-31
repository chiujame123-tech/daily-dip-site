import os
import requests
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
import json
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta

# --- 0. 讀取 API KEY ---
API_KEY = os.environ.get("POLYGON_API_KEY")

if not API_KEY:
    print("❌ 錯誤：找不到 POLYGON_API_KEY。請確認 GitHub Secrets 已設定。")
    # 本地測試時可暫時取消下方註解填入 Key，上傳時請務必刪除
    # API_KEY = "你的KEY"

# --- 1. 設定觀察清單 ---
SECTORS = {
    "💎 科技巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM"],
    "☁️ 軟體與SaaS": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PANW"],
    "🏦 金融": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. Polygon 數據獲取 ---
def get_polygon_data(ticker, multiplier=1, timespan='day', limit=100):
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d') # 抓多一點確保 MA 計算
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
        
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if data.get('status') != 'OK' or not data.get('results'):
            return None
            
        df = pd.DataFrame(data['results'])
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Data Error {ticker}: {e}")
        return None

def get_weekly_hot_news():
    """獲取過去 7 天的熱門股票新聞"""
    news_html = ""
    try:
        # 設定日期範圍：過去 7 天
        today = datetime.now().strftime('%Y-%m-%d')
        last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # 針對大盤 (SPY, QQQ) 和熱門股 (NVDA, TSLA) 抓新聞
        tickers = "SPY,QQQ,NVDA,TSLA,AAPL"
        url = f"https://api.polygon.io/v2/reference/news?ticker={tickers}&published_utc.gte={last_week}&limit=15&sort=published_utc&order=desc&apiKey={API_KEY}"
        
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if data.get('results'):
            for item in data['results']:
                title = item.get('title')
                article_url = item.get('article_url')
                publisher = item.get('publisher', {}).get('name', 'Unknown')
                published_utc = item.get('published_utc', '')
                description = item.get('description', '')
                
                try:
                    dt = datetime.strptime(published_utc, "%Y-%m-%dT%H:%M:%SZ")
                    date_str = dt.strftime('%Y/%m/%d')
                except:
                    date_str = ""
                
                # 簡單過濾掉太短或無意義的新聞
                if len(description) < 20: continue

                news_html += f"""
                <div class="news-item">
                    <div class="news-meta"><span style="color:#fbbf24">{date_str}</span> • {publisher}</div>
                    <a href="{article_url}" target="_blank" class="news-title">{title}</a>
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:5px;">{description[:100]}...</div>
                </div>
                """
        else:
            news_html = "<div style='padding:20px'>本週暫無重大熱門新聞。</div>"
            
    except Exception as e:
        news_html = f"<div style='padding:20px'>新聞載入錯誤: {e}</div>"
        
    return news_html

# --- 3. SMC 戰術分析邏輯 (核心) ---
def calculate_smc_levels(df):
    """計算 SMC 關鍵點位：Entry, SL, TP"""
    # 尋找最近 50 根 K 線的高低點 (Swing High/Low)
    window = 50
    recent_df = df.tail(window)
    
    bsl = recent_df['High'].max() # Buy Side Liquidity (TP)
    ssl = recent_df['Low'].min()  # Sell Side Liquidity (SL)
    eq = (bsl + ssl) / 2          # Equilibrium
    
    current_price = recent_df['Close'].iloc[-1]
    
    # 尋找最近的 Bullish FVG (看漲缺口) 作為最佳入場點
    best_entry = eq # 預設入場點為平衡點
    
    # 從最新往回找 FVG
    for i in range(len(recent_df)-1, 2, -1):
        # 條件: Low[i] > High[i-2] (中間有缺口) 且 缺口在折價區 ( < EQ )
        candle_low = recent_df['Low'].iloc[i]
        prev_high = recent_df['High'].iloc[i-2]
        
        if candle_low > prev_high:
            fvg_top = candle_low
            # 如果這個 FVG 在折價區，這就是最佳入場點
            if fvg_top < eq:
                best_entry = fvg_top
                break # 找到最近的一個就停止
    
    # SL 設定在 SSL 下方 1% 作為緩衝
    stop_loss = ssl * 0.99 
    
    return bsl, ssl, eq, best_entry, stop_loss

def identify_fvgs(df):
    features = {"FVG": []}
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({'type': 'Bullish', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': df.index[i-1]})
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({'type': 'Bearish', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': df.index[i-1]})
    return features

def generate_chart_image(df, ticker, timeframe, entry, sl, tp):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 30: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        smc_features = identify_fvgs(plot_df)
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        # 設定 SMC 戰術線 (Entry, SL, TP)
        hlines = dict(
            hlines=[tp, entry, sl],
            colors=['#10b981', '#3b82f6', '#ef4444'], # 綠(TP), 藍(Entry), 紅(SL)
            linewidths=[1.5, 1.5, 1.5],
            linestyle=['-', '--', '-']
        )

        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe}", color='white', size=10),
            hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # --- 文字標註 ---
        # TP
        ax.text(x_min, tp, f" TP (BSL): ${tp:.2f}", color='#10b981', fontsize=7, fontweight='bold', va='bottom')
        # Entry
        ax.text(x_min, entry, f" ENTRY: ${entry:.2f}", color='#3b82f6', fontsize=7, fontweight='bold', va='bottom')
        # SL
        ax.text(x_min, sl, f" SL (SSL): ${sl:.2f}", color='#ef4444', fontsize=7, fontweight='bold', va='top')

        # 區域底色
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.05)
        ax.add_patch(rect_prem)
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.05)
        ax.add_patch(rect_disc)

        # FVG 區塊
        for fvg in smc_features['FVG']:
            try:
                idx = plot_df.index.get_loc(fvg['index'])
                color = '#10b981' if fvg['type'] == 'Bullish' else '#ef4444'
                rect = patches.Rectangle((idx, fvg['bottom']), x_max-idx, fvg['top']-fvg['bottom'], linewidth=0, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            except: pass

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except Exception as e:
        print(f"Plot Error {ticker}: {e}")
        return None

def generate_ai_analysis_text(ticker, price, sma200, bsl, ssl, entry, sl, tp):
    # 趨勢分析
    trend = "多頭 (Bullish)" if price > sma200 else "空頭 (Bearish)"
    trend_color = "#10b981" if price > sma200 else "#ef4444"
    
    # 盈虧比計算 (RR)
    risk = entry - sl
    reward = tp - entry
    rr = reward / risk if risk > 0 else 0
    
    # 建議邏輯
    if price < entry * 1.02 and price > sl:
        action = "✅ **現價接近入場點，可考慮部署**"
        reason = "股價位於折價區且接近 FVG/支撐位。"
    elif price > entry * 1.05:
        action = "⏳ **價格已跑，建議等待回調**"
        reason = "目前價格偏離最佳入場點，追高風險大。"
    else:
        action = "👀 **觀察中**"
        reason = "價格結構尚未明確。"

    analysis = f"""
    <div class="ai-report">
        <div style="border-bottom:1px solid #333; padding-bottom:5px; margin-bottom:8px;">
            <b style="color:#fbbf24;">🤖 SMC 戰術面板 ({ticker})</b>
        </div>
        <ul style="padding-left:15px; margin:0; line-height:1.6;">
            <li><b>趨勢判定：</b> <span style="color:{trend_color}">{trend}</span> (vs 200MA)</li>
            <li><b>流動性目標 (TP)：</b> <span style="color:#10b981">${tp:.2f}</span> (BSL)</li>
            <li><b>最佳入場 (Entry)：</b> <span style="color:#3b82f6">${entry:.2f}</span> (FVG/EQ)</li>
            <li><b>防守位置 (SL)：</b> <span style="color:#ef4444">${sl:.2f}</span> (SSL)</li>
            <li><b>潛在盈虧比 (RR)：</b> {rr:.2f}R</li>
        </ul>
        <div style="margin-top:10px; padding:8px; background:rgba(255,255,255,0.05); border-radius:4px;">
            {action}<br>
            <span style="font-size:0.85em; color:#94a3b8;">理由: {reason}</span>
        </div>
    </div>
    """
    return analysis

# --- 4. 主程式 ---
def main():
    print("🚀 Starting SMC Analysis with Polygon...")
    
    if not API_KEY: return

    # 1. 抓取每週熱門新聞
    print("📰 Fetching Weekly Hot News...")
    weekly_news_html = get_weekly_hot_news()

    sector_html_blocks = ""
    screener_rows = ""
    APP_DATA = {}
    passed_count = 0

    for sector, tickers in SECTORS.items():
        cards_in_sector = ""
        for t in tickers:
            try:
                # 2. 抓取數據
                df_d = get_polygon_data(t, 1, 'day')
                if df_d is None or len(df_d) < 60: continue
                
                df_h = get_polygon_data(t, 1, 'hour')
                if df_h is None: df_h = df_d

                curr_price = df_d['Close'].iloc[-1]
                sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                if pd.isna(sma200): sma200 = curr_price

                # 3. 計算 SMC 關鍵位 (BSL, SSL, Entry, SL, TP)
                bsl, ssl, eq, entry, sl = calculate_smc_levels(df_d)
                tp = bsl # TP 設為上方流動性

                # 4. 生成圖表 (帶有 Entry/SL/TP 線)
                img_d = generate_chart_image(df_d, t, "Daily Structure", entry, sl, tp)
                img_h = generate_chart_image(df_h, t, "Hourly Entry", entry, sl, tp)

                # 5. 生成 AI 分析文案
                ai_html = generate_ai_analysis_text(t, curr_price, sma200, bsl, ssl, entry, sl, tp)

                # 訊號
                is_bullish = curr_price > sma200
                signal = "LONG" if is_bullish and curr_price < eq else "WAIT"
                cls = "b-long" if signal == "LONG" else "b-wait"

                # 存儲數據
                APP_DATA[t] = {
                    "signal": signal,
                    "deploy": ai_html,
                    "img_d": img_d,
                    "img_h": img_h
                }

                cards_in_sector += f"""
                <div class="card" onclick="openModal('{t}')">
                    <div class="head">
                        <div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                        <span class="badge {cls}">{signal}</span>
                    </div>
                    <div class="hint">查看 SMC 部署 ↗</div>
                </div>
                """
                
                # 篩選器條件 (價格在 200MA 上 且 回調到平衡點以下)
                if is_bullish:
                    passed_count += 1
                    screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>多頭</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

            except Exception as e:
                print(f"Skipping {t}: {e}")
                continue
        
        if cards_in_sector:
            sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

    json_data = json.dumps(APP_DATA)
    
    final_html = f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DailyDip Pro (SMC Edition)</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
        body {{ background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding:10px; }}
        .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; border-bottom:1px solid #333; overflow-x:auto; }}
        .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; font-weight:bold; font-size:0.9rem; white-space:nowrap; }}
        .tab.active {{ background:var(--acc); color:white; }}
        .content {{ display:none; }} .content.active {{ display:block; }}
        .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; font-size:1.1rem; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
        .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:10px; cursor:pointer; transition:0.2s; }}
        .card:hover {{ border-color:var(--acc); transform:translateY(-2px); }}
        .head {{ display:flex; justify-content:space-between; margin-bottom:5px; }}
        .code {{ font-weight:900; font-size:1rem; }} .price {{ color:#94a3b8; font-family:monospace; }}
        .badge {{ padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; }}
        .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
        .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #555; }}
        .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
        
        /* News Style */
        .news-item {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:15px; margin-bottom:10px; }}
        .news-meta {{ font-size:0.75rem; color:#94a3b8; margin-bottom:5px; }}
        .news-title {{ color:var(--text); text-decoration:none; font-weight:bold; font-size:1rem; display:block; margin-bottom:5px; }}
        .news-title:hover {{ color:var(--acc); }}
        
        /* Modal & AI Box */
        .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:99; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
        .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; margin-top:20px; border:1px solid #555; }}
        .m-content img {{ width:100%; border-radius:6px; margin-bottom:10px; border:1px solid #333; }}
        .ai-box {{ background:rgba(59,130,246,0.1); border-left:4px solid var(--acc); padding:15px; border-radius:4px; margin-bottom:15px; font-size:0.9rem; }}
        .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:6px; cursor:pointer; font-weight:bold; font-size:1rem; }}
        
        table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
        th, td {{ padding:8px; text-align:left; border-bottom:1px solid #333; }}
        .g {{ color:var(--g); }}
        .time {{ text-align:center; color:#666; font-size:0.7rem; margin-top:30px; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 多頭篩選</div>
            <div class="tab" onclick="setTab('news', this)">📰 本週熱點</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        
        <div id="screener" class="content">
            <div style="padding:10px; background:rgba(16,185,129,0.1); margin-bottom:15px; border-radius:6px; font-size:0.9rem;">
                🎯 <b>篩選邏輯：</b> 股價 > 200MA (多頭) + 價格 < EQ (折價區)
            </div>
            <table><thead><tr><th>代號</th><th>價格</th><th>趨勢</th><th>訊號</th></tr></thead><tbody>{screener_rows}</tbody></table>
        </div>

        <div id="news" class="content">
            <h3 class="sector-title">🔥 本週市場熱門 (Weekly Hot)</h3>
            {weekly_news_html}
        </div>
        
        <div class="time">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                
                <div id="m-deploy" class="ai-box"></div>
                
                <div><span style="color:#3b82f6; font-weight:bold; font-size:0.9rem;">📅 日線結構 (Structure)</span><div id="chart-d"></div></div>
                <div style="margin-top:15px;"><span style="color:#3b82f6; font-weight:bold; font-size:0.9rem;">⏱️ 小時入場 (Execution)</span><div id="chart-h"></div></div>
                
                <button class="close-btn" onclick="document.getElementById('modal').style.display='none'">關閉視窗</button>
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
            
            const dImg = data.img_d ? '<img src="' + data.img_d + '">' : '<div style="padding:20px;text-align:center;color:#666">No Data</div>';
            const hImg = data.img_h ? '<img src="' + data.img_h + '">' : '<div style="padding:20px;text-align:center;color:#666">No Data</div>';
            
            document.getElementById('chart-d').innerHTML = dImg;
            document.getElementById('chart-h').innerHTML = hImg;
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

