import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import base64
import json
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# --- 1. 設定觀察清單 ---
SECTORS = {
    "💎 Mag 7": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "⚡ Semi": ["TSM", "AMD", "AVGO", "MU", "INTC", "ARM"],
    "☁️ Software": ["PLTR", "COIN", "MSTR", "CRM", "SNOW"],
    "🏦 Finance": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. 核心繪圖函式 ---
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
        
        # 設定外觀
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#0f172a')
        
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe}", color='white', size=10),
            figsize=(5, 3), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # 畫區域
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.1)
        ax.add_patch(rect_prem)
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.1)
        ax.add_patch(rect_disc)
        ax.axhline(eq, color='#3b82f6', linestyle='--', linewidth=0.8, alpha=0.7)

        # 畫 FVG
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

# --- 3. 主程式 ---
def main():
    print("🚀 Starting Analysis (JSON Data Mode)...")
    
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
    
    # 🌟 關鍵修改：用來儲存所有圖表數據的字典
    APP_DATA = {}

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

                if len(df_d) < 50: continue
                curr_price = df_d['Close'].iloc[-1]
                if isinstance(curr_price, pd.Series): curr_price = curr_price.iloc[0]

                # 計算指標
                sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                if isinstance(sma200, pd.Series): sma200 = sma200.iloc[0]
                
                vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
                if isinstance(vol, pd.Series): vol = vol.iloc[0]
                
                stock_ret = df_d['Close'].pct_change()
                combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
                beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1] if len(combo)>30 else 0

                pass_filter = (curr_price > sma200 and vol > 900000000 and beta >= 1.0)

                # 強制畫圖
                res_d = generate_chart_image(df_d, t, "Daily")
                if not res_d: continue
                img_d_src, tp, sl = res_d
                
                res_h = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly")
                img_h_src = res_h[0] if res_h else ""

                # 計算訊號
                range_len = tp - sl
                pos_pct = (curr_price - sl) / range_len if range_len > 0 else 0.5
                signal = "LONG" if pos_pct < 0.45 else "WAIT"
                cls = "b-long" if signal == "LONG" else "b-wait"

                # 準備文案
                deploy_html = ""
                if signal == "LONG":
                    entry = curr_price
                    stop_loss = sl * 0.98
                    take_profit = tp
                    rr = (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
                    deploy_html = f"<div class='deploy-box long'><div class='deploy-title'>✅ LONG SETUP</div><ul class='deploy-list'><li><b>Entry:</b> ${entry:.2f}</li><li><b>SL:</b> ${stop_loss:.2f}</li><li><b>TP:</b> ${take_profit:.2f}</li><li><b>RR:</b> {rr:.1f}R</li></ul></div>"
                else:
                    target_buy = sl + (range_len * 0.4)
                    deploy_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><ul class='deploy-list'><li>Price in Premium.</li><li>Wait for drop below: <b>${target_buy:.2f}</b></li></ul></div>"

                # 🌟 關鍵修改：將數據存入字典，而不是塞進 HTML 字串
                APP_DATA[t] = {
                    "signal": signal,
                    "price": f"${curr_price:.2f}",
                    "deploy": deploy_html,
                    "img_d": img_d_src,
                    "img_h": img_h_src
                }

                # 🌟 卡片現在只傳遞 't' (Ticker 名稱)
                cards_in_sector += f"""
                <div class="card" onclick="openModal('{t}')">
                    <div class="head">
                        <div><div class="code">{t}</div><div class="price">${curr_price:.2f}</div></div>
                        <span class="badge {cls}">{signal}</span>
                    </div>
                    <div class="hint">Tap for Details ↗</div>
                </div>
                """
                
                if pass_filter:
                    passed_count += 1
                    screener_rows += f"<tr><td>{t}</td><td>${curr_price:.2f}</td><td class='g'>Pass</td><td>{beta:.2f}</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

            except Exception as e:
                print(f"Skipping {t}: {e}")
                continue
        
        if cards_in_sector:
            sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

    # 生成 HTML (將數據字典轉換為 JSON)
    json_data = json.dumps(APP_DATA)
    
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
        .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }}
        .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:10px; cursor:pointer; }}
        .head {{ display:flex; justify-content:space-between; margin-bottom:5px; }}
        .code {{ font-weight:900; font-size:1rem; }} .price {{ color:#94a3b8; font-family:monospace; }}
        .badge {{ padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:bold; }}
        .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
        .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #555; }}
        .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
        table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
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
        .close-btn {{ width:100%; padding:12px; background:var(--acc); border:none; color:white; border-radius:6px; cursor:pointer; font-weight:bold; font-size:1rem; }}
        .chart-lbl {{ color:var(--acc); font-weight:bold; display:block; margin-bottom:5px; font-size:0.9rem; }}
        .time {{ text-align:center; color:#666; font-size:0.7rem; margin-top:30px; }}
    </style>
    </head>
    <body>
        <div class="tabs">
            <div class="tab active" onclick="setTab('overview', this)">📊 Market</div>
            <div class="tab" onclick="setTab('screener', this)">🔍 Screener ({passed_count})</div>
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks}</div>
        
        <div id="screener" class="content">
            <table><thead><tr><th>Ticker</th><th>Price</th><th>Status</th><th>Beta</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table>
        </div>
        
        <div class="time">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                
                <div>
                    <span class="chart-lbl">📅 Daily Chart (Trend)</span>
                    <div id="chart-d"></div>
                </div>
                <div style="margin-top:15px;">
                    <span class="chart-lbl">⏱️ Hourly Chart (Entry)</span>
                    <div id="chart-h"></div>
                </div>
                
                <button class="close-btn" onclick="document.getElementById('modal').style.display='none'">Close</button>
            </div>
        </div>

        <script>
        // 🌟 這是所有數據的核心資料庫
        const STOCK_DATA = {json_data};

        function setTab(id, el) {{
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            el.classList.add('active');
        }}

        function openModal(ticker) {{
            // 🌟 從資料庫讀取數據，而不是從 HTML 屬性
            const data = STOCK_DATA[ticker];
            if (!data) return;

            document.getElementById('modal').style.display = 'flex';
            document.getElementById('m-ticker').innerText = ticker + " (" + data.signal + ")";
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
