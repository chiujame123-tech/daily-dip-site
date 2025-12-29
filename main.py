import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# --- 1. 設定觀察清單 ---
SECTORS = {
    "💎 Magnificent 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "MU", "INTC", "ARM"],
    "☁️ Software & Crypto": ["PLTR", "COIN", "MSTR", "CRM", "SNOW"],
    "🏦 Finance & Retail": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 核心繪圖函式 ---
def identify_smc_features(df):
    features = {"FVG": [], "DISP": []}
    # 簡單 FVG
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({'type': 'Bullish', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': df.index[i-1]})
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({'type': 'Bearish', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': df.index[i-1]})
    return features

def generate_chart_image(df, ticker, timeframe):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 30: return None
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        
        smc = identify_smc_features(plot_df)
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        
        # 縮小尺寸以優化速度
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe}", color='white', size=10),
            figsize=(4, 2.5), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 繪製區域
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.1)
        ax.add_patch(rect_prem)
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.1)
        ax.add_patch(rect_disc)
        ax.axhline(eq, color='#3b82f6', linestyle='-.', linewidth=1, alpha=0.6)

        # FVG
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
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}", swing_high, swing_low, eq
    except Exception as e:
        return None

# --- 3. 主程式 ---
def main():
    print("🚀 Starting Analysis Cycle...")
    
    # 下載數據
    data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=False)
    data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
    
    if isinstance(data_daily.columns, pd.MultiIndex):
        spy_close = data_daily['SPY']['Close']
    else:
        spy_close = data_daily['Close']
    spy_ret = spy_close.pct_change()

    sector_html_blocks = ""
    screener_rows = ""
    passed_count = 0

    for sector, tickers in SECTORS.items():
        cards_in_sector = ""
        for t in tickers:
            try:
                if isinstance(data_daily.columns, pd.MultiIndex):
                    try:
                        df_d = data_daily[t].dropna()
                        df_h = data_hourly[t].dropna()
                    except: continue
                else: continue

                if len(df_d) < 200: continue
                curr_price = df_d['Close'].iloc[-1]
                if isinstance(curr_price, pd.Series): curr_price = curr_price.iloc[0]

                # 篩選邏輯
                sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                if isinstance(sma200, pd.Series): sma200 = sma200.iloc[0]
                
                vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
                if isinstance(vol, pd.Series): vol = vol.iloc[0]
                
                stock_ret = df_d['Close'].pct_change()
                combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
                beta = 0
                if len(combo) > 30:
                    beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1]

                pass_filter = (curr_price > sma200 and vol > 900000000 and beta >= 1.0)

                # 訊號生成 (先不畫圖)
                window = 20
                swing_high = df_d['High'].tail(window).max()
                swing_low = df_d['Low'].tail(window).min()
                range_len = swing_high - swing_low
                pos_pct = (curr_price - swing_low) / range_len if range_len > 0 else 0.5
                signal = "LONG" if pos_pct < 0.4 else "WAIT"
                cls = "b-long" if signal == "LONG" else "b-wait"

                # 優化：只對重要的股票畫圖
                should_draw = pass_filter or (signal == "LONG")
                img_d_src, img_h_src = "", ""
                
                if should_draw:
                    res_d = generate_chart_image(df_d, t, "D1")
                    if res_d:
                        img_d_src, tp, sl, eq = res_d
                        res_h = generate_chart_image(df_h if not df_h.empty else df_d, t, "H1")
                        if res_h: img_h_src = res_h[0]
                    else:
                        # 如果畫圖失敗，補上預設值防止報錯
                        tp, sl, eq = swing_high, swing_low, (swing_high+swing_low)/2
                else:
                    tp, sl, eq = swing_high, swing_low, (swing_high+swing_low)/2

                # AI 文案
                deploy_html = ""
                if signal == "LONG":
                    entry = curr_price
                    stop_loss = sl * 0.98
                    take_profit = tp
                    rr = (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
                    
                    deploy_html = f"""
                    <div class='deploy-box long'>
                        <div class='deploy-title'>✅ LONG SETUP</div>
                        <ul class='deploy-list'>
                            <li><b>Entry:</b> ${entry:.2f}</li>
                            <li><b>SL:</b> ${stop_loss:.2f}</li>
                            <li><b>TP:</b> ${take_profit:.2f}</li>
                            <li><b>RR:</b> {rr:.1f}R</li>
                        </ul>
                    </div>
                    """
                else:
                    deploy_html = f"""
                    <div class='deploy-box wait'>
                        <div class='deploy-title'>⏳ WAIT</div>
                        <ul class='deploy-list'>
                            <li>Price in Premium.</li>
                            <li>Wait for pullback to ${eq:.2f}.</li>
                        </ul>
                    </div>
                    """
                
                # HTML 轉義
                deploy_clean = deploy_html.replace('"', '&quot;').replace('\n', '')

                cards_in_sector += f"""
                <div class="card" onclick="openModal('{t}', '{img_d_src}', '{img_h_src}', '{signal}', '{deploy_clean}')">
                    <div class="head">
                        <div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                        <span class="badge {cls}">{signal}</span>
                    </div>
                    <div class="hint">Tap for Strategy ↗</div>
                </div>
                """
                
                if pass_filter:
                    passed_count += 1
                    screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>Pass</td><td>{beta:.2f}</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

            except Exception as e:
                continue
        
        if cards_in_sector:
            sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

    # 生成 index.html
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DailyDip AI Report</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
        body {{ background:var(--bg); color:var(--text); font-family:sans-serif; margin:0; padding:10px; }}
        .tabs {{ display:flex; gap:10px; padding-bottom:10px; margin-bottom:15px; border-bottom:1px solid #333; }}
        .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; font-weight:bold; font-size:0.9rem; }}
        .tab.active {{ background:var(--acc); color:white; }}
        .content {{ display:none; }} .content.active {{ display:block; }}
        .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; font-size:1.1rem; }}
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; }}
        .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:12px; cursor:pointer; }}
        .head {{ display:flex; justify-content:space-between; margin-bottom:5px; }}
        .code {{ font-weight:900; font-size:1.1rem; }} .price {{ color:#94a3b8; font-family:monospace; }}
        .badge {{ padding:2px 6px; border-radius:4px; font-size:0.8rem; font-weight:bold; }}
        .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); }}
        .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; }}
        .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
        th, td {{ padding:8px; text-align:left; border-bottom:1px solid #333; }}
        .g {{ color:var(--g); }}
        
        .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:99; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
        .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; margin-top:20px; border:1px solid #555; }}
        .m-content img {{ width:100%; border-radius:6px; margin-bottom:10px; border:1px solid #333; }}
        .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
        .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
        .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
        .deploy-title {{ font-weight:bold; margin-bottom:5px; color:white; }}
        .deploy-list {{ margin:0; padding-left:20px; color:#cbd5e1; font-size:0.9rem; }}
        .close-btn {{ width:100%; padding:10px; background:var(--acc); border:none; color:white; border-radius:6px; cursor:pointer; font-weight:bold; font-size:1rem; }}
        .time {{ text-align:center; color:#666; font-size:0.8rem; margin-top:20px; }}
        .no-chart {{ padding:20px; text-align:center; color:#64748b; border:1px dashed #333; border-radius:6px; margin-bottom:10px; font-size:0.8rem; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">Overview</div>
            <div class="tab" onclick="setTab('screener', this)">Screener ({passed_count})</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        
        <div id="screener" class="content">
            <table><thead><tr><th>Ticker</th><th>Price</th><th>Trend</th><th>Beta</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table>
        </div>
        
        <div class="time">Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div id="chart-d"></div>
                <div id="chart-h"></div>
                <button class="close-btn" onclick="document.getElementById('modal').style.display='none'">Close</button>
            </div>
        </div>

        <script>
        function setTab(id, el) {{
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            el.classList.add('active');
        }}
        function openModal(t, d, h, s, html) {{
            document.getElementById('modal').style.display = 'flex';
            document.getElementById('m-ticker').innerText = t + " (" + s + ")";
            document.getElementById('m-deploy').innerHTML = html;
            
            let cd = document.getElementById('chart-d');
            let ch = document.getElementById('chart-h');
            
            if(d) cd.innerHTML = '<div><b style="color:#3b82f6">Daily Chart</b><br><img src="'+d+'"></div>';
            else cd.innerHTML = '<div class="no-chart">Daily Chart not loaded (Optimized)</div>';
            
            if(h) ch.innerHTML = '<div><b style="color:#3b82f6">Hourly Chart</b><br><img src="'+h+'"></div>';
            else ch.innerHTML = '<div class="no-chart">Hourly Chart not loaded (Optimized)</div>';
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
