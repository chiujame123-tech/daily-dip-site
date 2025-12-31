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

# --- 1. 完整觀察清單 ---
SECTORS = {
    "💎 科技七巨頭": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ 半導體": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM", "QCOM", "SMCI"],
    "☁️ 軟體與SaaS": ["PLTR", "COIN", "MSTR", "CRM", "SNOW", "PANW", "CRWD", "SHOP"],
    "🏦 金融與消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. Polygon 數據請求 (已修正日期問題) ---
def get_polygon_data(ticker, multiplier=1, timespan='day'):
    if not API_KEY: return None
    
    try:
        # 🌟 關鍵修正：強制抓取「昨天」以前的數據，避開 Starter 方案限制
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=true&sort=asc&limit=500&apiKey={API_KEY}"
        
        resp = requests.get(url, timeout=10)
        
        # 簡單重試機制
        if resp.status_code == 429: # Rate Limit
            time.sleep(1)
            resp = requests.get(url, timeout=10)

        data = resp.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def get_polygon_news():
    if not API_KEY: return "<div style='padding:20px'>API Key Missing</div>"
    news_html = ""
    try:
        # 抓取本週熱門新聞
        last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        tickers = "SPY,QQQ,NVDA,TSLA,AAPL,AMD"
        url = f"https://api.polygon.io/v2/reference/news?ticker={tickers}&published_utc.gte={last_week}&limit=12&sort=published_utc&order=desc&apiKey={API_KEY}"
        
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
            news_html = "<div style='padding:20px'>暫無熱門新聞</div>"
    except:
        news_html = "<div style='padding:20px'>新聞載入失敗</div>"
    return news_html

# --- 3. SMC 戰術分析邏輯 ---
def calculate_smc(df):
    """計算 SMC 關鍵位"""
    window = 50
    recent = df.tail(window)
    
    bsl = recent['High'].max() # TP (上方流動性)
    ssl = recent['Low'].min()  # SL (下方流動性)
    eq = (bsl + ssl) / 2       # 平衡點
    
    best_entry = eq
    # 尋找折價區內的最近 FVG
    for i in range(len(recent)-1, 2, -1):
        if recent['Low'].iloc[i] > recent['High'].iloc[i-2]: # Bullish FVG
            fvg_top = recent['Low'].iloc[i]
            if fvg_top < eq: # 必須在折價區
                best_entry = fvg_top
                break
                
    return bsl, ssl, eq, best_entry, ssl * 0.99

def generate_chart(df, ticker, title, entry, sl, tp):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 20: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2

        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        # 繪製戰術線
        hlines = dict(
            hlines=[tp, entry, sl],
            colors=['#10b981', '#3b82f6', '#ef4444'],
            linewidths=[1.5, 1.5, 1.5],
            linestyle=['-', '--', '-']
        )
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {title}", color='white', size=10),
            hlines=hlines, figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 文字標註
        ax.text(x_min, tp, f" TP ${tp:.2f}", color='#10b981', fontsize=7, va='bottom', fontweight='bold')
        ax.text(x_min, entry, f" ENTRY ${entry:.2f}", color='#3b82f6', fontsize=7, va='bottom', fontweight='bold')
        ax.text(x_min, sl, f" SL ${sl:.2f}", color='#ef4444', fontsize=7, va='top', fontweight='bold')
        
        # 區域標示
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.05)
        ax.add_patch(rect_prem)
        ax.text(x_min, swing_high, " Premium (Sell)", color='#ef4444', fontsize=6, va='top', alpha=0.6)
        
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.05)
        ax.add_patch(rect_disc)
        ax.text(x_min, swing_low, " Discount (Buy)", color='#10b981', fontsize=6, va='bottom', alpha=0.6)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=70)
        plt.close(fig)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except:
        return None

# --- 4. 主程式 ---
def main():
    print("🚀 Starting Polygon Pro Analysis...")
    
    if not API_KEY:
        print("❌ API Key Missing")
        return # 這裡可以 return 因為我們確定 Key 沒問題了
    
    # 1. 抓新聞
    weekly_news_html = get_polygon_news()

    sector_html_blocks = ""
    screener_rows = ""
    APP_DATA = {}
    
    for sector, tickers in SECTORS.items():
        cards_in_sector = ""
        for t in tickers:
            try:
                # 避免頻率限制
                time.sleep(0.2) 
                
                # 2. 獲取數據
                df_d = get_polygon_data(t, 1, 'day')
                if df_d is None or len(df_d) < 50: continue
                
                # 嘗試抓小時線，失敗就用日線
                df_h = get_polygon_data(t, 1, 'hour')
                if df_h is None: df_h = df_d

                curr_price = df_d['Close'].iloc[-1]
                sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                if pd.isna(sma200): sma200 = curr_price

                # 3. SMC 計算
                bsl, ssl, eq, entry, sl = calculate_smc(df_d)
                tp = bsl

                # 4. 繪圖
                img_d = generate_chart(df_d, t, "Daily Structure", entry, sl, tp)
                if not img_d: continue
                
                img_h = generate_chart(df_h, t, "Hourly Entry", entry, sl, tp)
                if not img_h: img_h = ""

                # 5. 訊號與AI文案
                is_bullish = curr_price > sma200
                signal = "LONG" if is_bullish and curr_price < eq else "WAIT"
                cls = "b-long" if signal == "LONG" else "b-wait"
                
                risk = entry - sl
                reward = tp - entry
                rr = reward / risk if risk > 0 else 0
                
                trend_str = "多頭 (Bullish)" if is_bullish else "空頭 (Bearish)"
                
                if signal == "LONG":
                    ai_html = f"""
                    <div class='deploy-box long'>
                        <div class='deploy-title'>✅ LONG SETUP (做多建議)</div>
                        <ul class='deploy-list'>
                            <li><b>入場 (Entry):</b> ${entry:.2f}</li>
                            <li><b>止損 (SL):</b> ${sl:.2f}</li>
                            <li><b>止盈 (TP):</b> ${tp:.2f}</li>
                            <li><b>盈虧比:</b> {rr:.1f}R</li>
                        </ul>
                        <div style='margin-top:10px; font-size:0.85rem'>
                            🤖 <b>AI 分析:</b> 股價位於 200MA 之上且回調至折價區，SMC 結構完整，建議尋找入場機會。
                        </div>
                    </div>
                    """
                else:
                    ai_html = f"""
                    <div class='deploy-box wait'>
                        <div class='deploy-title'>⏳ WAIT (觀望建議)</div>
                        <ul class='deploy-list'>
                            <li>目前位置: <b>Premium (溢價區)</b></li>
                            <li>趨勢狀態: {trend_str}</li>
                            <li>建議等待回調至: <b>${entry:.2f}</b></li>
                        </ul>
                    </div>
                    """
                
                APP_DATA[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h}

                cards_in_sector += f"""
                <div class="card" onclick="openModal('{t}')">
                    <div class="head"><div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div><span class="badge {cls}">{signal}</span></div>
                    <div class="hint">點擊查看分析 ↗</div>
                </div>"""
                
                if is_bullish and curr_price < eq:
                    screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>多頭回調</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

            except Exception as e:
                print(f"Error processing {t}: {e}")
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
    <title>DailyDip Pro (Polygon)</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
        body {{ background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding:10px; }}
        .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; border-bottom:1px solid #333; overflow-x:auto; }}
        .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; font-weight:bold; font-size:0.9rem; white-space:nowrap; }}
        .tab.active {{ background:var(--acc); color:white; }}
        .content {{ display:none; }} .content.active {{ display:block; }}
        .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
        .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:10px; cursor:pointer; transition:0.2s; }}
        .card:hover {{ border-color:var(--acc); transform:translateY(-2px); }}
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
        .news-title:hover {{ color:var(--acc); }}
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
            <div class="tab" onclick="setTab('screener', this)">🔍 強勢篩選</div>
            <div class="tab" onclick="setTab('news', this)">📰 熱門新聞</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks if sector_html_blocks else '<div style="padding:20px;text-align:center">數據載入中或 API 連線異常，請稍後再試。</div>'}</div>
        <div id="screener" class="content"><table><thead><tr><th>代號</th><th>價格</th><th>狀態</th><th>訊號</th></tr></thead><tbody>{screener_rows}</tbody></table></div>
        <div id="news" class="content"><h3 class="sector-title">本週熱門新聞 (Polygon)</h3>{weekly_news_html}</div>
        
        <div class="time">Powered by Polygon.io | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><b style="color:#3b82f6">日線結構 (Daily Structure)</b><div id="chart-d"></div></div>
                <div style="margin-top:15px;"><b style="color:#3b82f6">小時入場 (Hourly Entry)</b><div id="chart-h"></div></div>
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
