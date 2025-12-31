import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
import json
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# --- 1. 設定觀察清單 (您可以隨意增加) ---
SECTORS = {
    "💎 科技巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM"],
    "☁️ 軟體與SaaS": ["PLTR", "CRM", "ADBE", "SNOW", "PANW", "COIN", "MSTR"],
    "🏦 金融與消費": ["JPM", "V", "MA", "COST", "MCD", "NKE", "KO"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 抓取中文新聞 (Google News RSS) ---
def get_chinese_news():
    news_html = ""
    try:
        # Google News RSS 針對 "美股" + 熱門關鍵字
        url = "https://news.google.com/rss/search?q=美股+NVDA+TSLA+AAPL+AMD+台積電&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            count = 0
            
            # 解析 XML
            for item in root.findall('./channel/item'):
                if count >= 12: break # 取前 12 篇
                
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                source = item.find('source').text if item.find('source') is not None else "新聞快訊"
                
                # 簡單格式化時間
                try:
                    dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                    date_str = dt.strftime('%m/%d %H:%M')
                except:
                    date_str = ""

                news_html += f"""
                <div class="news-item">
                    <div class="news-meta">{source} • {date_str}</div>
                    <a href="{link}" target="_blank" class="news-title">{title}</a>
                </div>
                """
                count += 1
    except Exception as e:
        print(f"News Error: {e}")
        news_html = f"<div style='padding:20px'>暫時無法載入新聞 ({e})</div>"
        
    return news_html

# --- 3. AI 中文分析邏輯 ---
def generate_ai_analysis(ticker, price, sma200, swing_high, swing_low):
    # 趨勢判斷
    trend = "多頭上升趨勢 (股價在 200MA 之上)" if price > sma200 else "空頭修正趨勢 (股價在 200MA 之下)"
    trend_color = "#10b981" if price > sma200 else "#ef4444"
    
    # 價格位置判斷
    range_len = swing_high - swing_low
    if range_len == 0: pos = 0.5
    else: pos = (price - swing_low) / range_len
    
    zone = ""
    action = ""
    risk = ""
    
    if pos > 0.6:
        zone = "🔴 溢價區 (Premium Zone)"
        action = "目前價格偏貴，不建議追高。"
        risk = "回調風險較大，建議等待拉回至平衡點 (50%) 再觀察。"
    elif pos < 0.4:
        zone = "🟢 折價區 (Discount Zone)"
        action = "進入機構偏好的買入區間。"
        risk = "若趨勢向上，這裡是盈虧比 (RR) 極佳的入場點。"
    else:
        zone = "🔵 平衡區 (Equilibrium)"
        action = "價格位於中間地帶，方向不明。"
        risk = "建議觀望，等待價格進入折價區再行動。"

    # 組合中文分析報告
    analysis = f"""
    <div class="ai-report">
        <div style="margin-bottom:8px; border-bottom:1px solid #333; padding-bottom:5px;">
            <b style="color:#fbbf24;">🤖 AI 智能分析報告 ({ticker})</b>
        </div>
        <ul style="padding-left:15px; margin:0;">
            <li style="margin-bottom:5px;"><b>趨勢狀態：</b> <span style="color:{trend_color}">{trend}</span></li>
            <li style="margin-bottom:5px;"><b>目前位置：</b> <b>{zone}</b></li>
            <li style="margin-bottom:5px;"><b>關鍵壓力：</b> 前波高點 <b>${swing_high:.2f}</b></li>
            <li style="margin-bottom:5px;"><b>關鍵支撐：</b> 前波低點 <b>${swing_low:.2f}</b></li>
            <li style="margin-top:10px; line-height:1.5;">
                <b>💡 部署建議：</b><br>
                {action}<br>
                <span style="font-size:0.85em; color:#94a3b8;">({risk})</span>
            </li>
        </ul>
    </div>
    """
    return analysis

# --- 4. 繪圖核心函式 (加入指標註解) ---
def identify_smc_features(df):
    features = {"FVG": []}
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({'type': 'Bullish', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': df.index[i-1]})
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({'type': 'Bearish', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': df.index[i-1]})
    return features

def generate_chart_image(df, ticker, timeframe):
    try:
        plot_df = df.tail(50)
        if len(plot_df) < 20: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        smc = identify_smc_features(plot_df)
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe}", color='white', size=10),
            figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # --- 1. 加入區域與文字註解 (Chart Annotations) ---
        
        # Premium (賣出區)
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.1)
        ax.add_patch(rect_prem)
        ax.text(x_min, swing_high, " Premium (Sell)", color='#fca5a5', fontsize=6, va='top', fontweight='bold')
        
        # Discount (買入區)
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.1)
        ax.add_patch(rect_disc)
        ax.text(x_min, swing_low, " Discount (Buy)", color='#86efac', fontsize=6, va='bottom', fontweight='bold')
        
        # Equilibrium (平衡線)
        ax.axhline(eq, color='#3b82f6', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(x_max, eq, " EQ (50%)", color='#3b82f6', fontsize=6, ha='right', va='center')

        # FVG (缺口)
        for fvg in smc['FVG']:
            try:
                idx = plot_df.index.get_loc(fvg['index'])
                color = '#10b981' if fvg['type'] == 'Bullish' else '#ef4444'
                rect = patches.Rectangle((idx, fvg['bottom']), x_max-idx, fvg['top']-fvg['bottom'], linewidth=0, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            except: pass

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=60)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}", swing_high, swing_low
    except:
        return None

# --- 5. 主程式 ---
def main():
    print("🚀 正在啟動分析程序 (中文版)...")
    
    # 下載股價
    print("📊 下載股價數據中...")
    data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=False)
    data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    
    # 抓取中文新聞
    print("📰 正在搜尋熱門美股新聞...")
    market_news_block = get_chinese_news()

    if isinstance(data_daily.columns, pd.MultiIndex):
        spy_close = data_daily['SPY']['Close']
    else:
        spy_close = data_daily['Close']
    spy_ret = spy_close.pct_change()

    sector_html_blocks = ""
    screener_rows = ""
    passed_count = 0
    
    APP_DATA = {}

    for sector, tickers in SECTORS.items():
        cards_in_sector = ""
        for t in tickers:
            try:
                # 數據處理
                if isinstance(data_daily.columns, pd.MultiIndex):
                    try:
                        df_d = data_daily[t].dropna()
                        df_h = data_hourly[t].dropna()
                    except: continue
                else: continue

                if len(df_d) < 50: continue
                curr_price = df_d['Close'].iloc[-1]
                if isinstance(curr_price, pd.Series): curr_price = curr_price.iloc[0]

                # 指標計算
                sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                if isinstance(sma200, pd.Series): sma200 = sma200.iloc[0]
                
                vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
                if isinstance(vol, pd.Series): vol = vol.iloc[0]
                
                stock_ret = df_d['Close'].pct_change()
                combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
                beta = 0
                if len(combo) > 30:
                    beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1] if len(combo)>30 else 0

                pass_filter = (curr_price > sma200 and vol > 900000000 and beta >= 1.0)

                # 生成圖表
                res_d = generate_chart_image(df_d, t, "Daily (D1)")
                if not res_d: continue
                img_d_src, tp, sl = res_d
                
                res_h = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly (H1)")
                img_h_src = res_h[0] if res_h else ""

                # 訊號判斷
                range_len = tp - sl
                pos_pct = (curr_price - sl) / range_len if range_len > 0 else 0.5
                signal = "LONG" if pos_pct < 0.45 else "WAIT"
                cls = "b-long" if signal == "LONG" else "b-wait"

                # 生成中文分析文案
                ai_text = generate_ai_analysis(t, curr_price, sma200, tp, sl)

                # 存入數據
                APP_DATA[t] = {
                    "signal": signal,
                    "price": f"${curr_price:.2f}",
                    "deploy": ai_text,
                    "img_d": img_d_src,
                    "img_h": img_h_src
                }

                cards_in_sector += f"""
                <div class="card" onclick="openModal('{t}')">
                    <div class="head">
                        <div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                        <span class="badge {cls}">{signal}</span>
                    </div>
                    <div class="hint">點擊查看分析 ↗</div>
                </div>
                """
                
                if pass_filter:
                    passed_count += 1
                    screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>通過</td><td>{beta:.2f}</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

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
    <title>美股 AI 智能分析</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
        body {{ background:var(--bg); color:var(--text); font-family:-apple-system, BlinkMacSystemFont, "Microsoft JhengHei", sans-serif; margin:0; padding:10px; }}
        
        /* 頁籤樣式 */
        .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; border-bottom:1px solid #333; overflow-x:auto; }}
        .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; font-weight:bold; font-size:0.9rem; white-space:nowrap; transition:0.2s; }}
        .tab.active {{ background:var(--acc); color:white; }}
        
        .content {{ display:none; animation:fadeIn 0.3s; }} .content.active {{ display:block; }}
        @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}

        .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:25px 0 10px; font-size:1.1rem; color:#e2e8f0; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
        
        .card {{ background:var(--card); border:1px solid #333; border-radius:10px; padding:12px; cursor:pointer; transition:0.2s; }}
        .card:hover {{ border-color:var(--acc); transform:translateY(-3px); }}
        
        .head {{ display:flex; justify-content:space-between; margin-bottom:5px; }}
        .code {{ font-weight:900; font-size:1.1rem; }} .price {{ color:#94a3b8; font-family:monospace; }}
        .badge {{ padding:3px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; }}
        .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
        .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #555; }}
        .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
        
        table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
        th, td {{ padding:10px; text-align:left; border-bottom:1px solid #333; }}
        th {{ color:#94a3b8; }}
        .g {{ color:var(--g); }}
        
        /* 新聞樣式 */
        .news-item {{ background:var(--card); border:1px solid #333; border-radius:10px; padding:15px; margin-bottom:12px; transition:0.2s; }}
        .news-item:hover {{ border-color:#64748b; }}
        .news-meta {{ font-size:0.75rem; color:#94a3b8; margin-bottom:6px; }}
        .news-title {{ color:var(--text); text-decoration:none; font-weight:bold; font-size:1rem; line-height:1.4; display:block; }}
        .news-title:hover {{ color:var(--acc); }}

        /* 彈窗樣式 */
        .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:99; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
        .m-content {{ background:var(--card); width:100%; max-width:600px; padding:20px; border-radius:12px; margin-top:20px; border:1px solid #555; position:relative; }}
        .m-content img {{ width:100%; border-radius:6px; margin-bottom:10px; border:1px solid #333; }}
        
        .ai-box {{ background:rgba(59,130,246,0.1); border-left:4px solid var(--acc); padding:15px; border-radius:4px; margin-bottom:20px; line-height:1.6; font-size:0.95rem; color:#e2e8f0; }}
        
        .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:8px; cursor:pointer; font-weight:bold; font-size:1rem; margin-top:10px; }}
        .time {{ text-align:center; color:#666; font-size:0.7rem; margin-top:30px; margin-bottom:20px; }}
        .chart-lbl {{ color:var(--acc); font-weight:bold; display:block; margin-bottom:5px; font-size:0.9rem; margin-top:10px; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 市場概況</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選</div>
            <div class="tab" onclick="setTab('news', this)">📰 熱門新聞</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        
        <div id="screener" class="content">
            <div style="margin-bottom:15px; padding:10px; background:rgba(16,185,129,0.1); border-radius:6px; font-size:0.9rem;">
                🎯 <b>篩選條件：</b> 股價 > 200MA • 交易量大 • 高波動 (Beta > 1)
            </div>
            <table><thead><tr><th>代號</th><th>價格</th><th>狀態</th><th>Beta</th><th>訊號</th></tr></thead><tbody>{screener_rows}</tbody></table>
        </div>

        <div id="news" class="content">
            <h3 class="sector-title">🔥 今日美股熱點 (Google News)</h3>
            {market_news_block}
        </div>
        
        <div class="time">最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                
                <div id="m-deploy"></div>
                
                <div>
                    <span class="chart-lbl">📅 日線圖 (趨勢與區域)</span>
                    <div id="chart-d"></div>
                </div>
                <div>
                    <span class="chart-lbl">⏱️ 小時圖 (SMC入場細節)</span>
                    <div id="chart-h"></div>
                </div>
                
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
            document.getElementById('m-ticker').innerText = ticker + " (" + data.signal + ")";
            document.getElementById('m-deploy').innerHTML = data.deploy;
            
            const dImg = data.img_d ? '<img src="' + data.img_d + '">' : '<div style="padding:20px;text-align:center;color:#666">暫無圖表數據</div>';
            const hImg = data.img_h ? '<img src="' + data.img_h + '">' : '<div style="padding:20px;text-align:center;color:#666">暫無圖表數據</div>';
            
            document.getElementById('chart-d').innerHTML = dImg;
            document.getElementById('chart-h').innerHTML = hImg;
        }}
        </script>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html 生成成功！")

if __name__ == "__main__":
    main()
