import os
import sys
import threading
import time
import base64
from io import BytesIO
from datetime import datetime

# --- Web Server Framework ---
from flask import Flask, render_template_string

# --- Data Analysis Libraries ---
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)

# Global variable to store the latest HTML content
LATEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="10">
    <style>
        body { background: #0f172a; color: white; font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .loader { text-align: center; }
        .spinner { border: 4px solid #334155; border-top: 4px solid #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="loader">
        <h1>🚀 System Initializing...</h1>
        <p>AI Analyst is scanning the market with SMC Logic.</p>
        <p>This may take 1-2 minutes. Please wait...</p>
        <div class="spinner"></div>
    </div>
</body>
</html>
"""

# --- 1. Settings & Watchlist ---
SECTORS = {
    "💎 Magnificent 7 & AI": ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "AVGO"],
    "⚡ Semiconductor": ["TSM", "ASML", "AMAT", "MU", "INTC", "ARM"],
    "☁️ Software & Crypto": ["PLTR", "COIN", "MSTR", "CRM", "SNOW"],
    "🏦 Finance & Retail": ["JPM", "V", "COST", "MCD", "NKE"],
}
ALL_TICKERS = [t for sector in SECTORS.values() for t in sector]

# --- 2. Filter Parameters ---
FILTER_SMA_PERIOD = 200
FILTER_MIN_MONTHLY_VOL = 900000000
FILTER_MIN_BETA = 1.0

# --- 3. SMC Logic ---
def identify_smc_features(df):
    """Identify FVG, EQH/EQL, Displacement"""
    features = {"FVG": [], "EQH": [], "EQL": [], "DISP": []}
    
    # 1. Displacement
    df['Body'] = abs(df['Close'] - df['Open'])
    avg_body = df['Body'].rolling(20).mean()
    features['DISP'] = df.index[df['Body'] > avg_body * 2.5].tolist()

    # 2. Fair Value Gaps (FVG)
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            features['FVG'].append({
                'type': 'Bullish', 'top': df['Low'].iloc[i], 'bottom': df['High'].iloc[i-2], 'index': df.index[i-1]
            })
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            features['FVG'].append({
                'type': 'Bearish', 'top': df['Low'].iloc[i-2], 'bottom': df['High'].iloc[i], 'index': df.index[i-1]
            })

    # 3. EQH / EQL (Simplified)
    window = 5
    highs = df[df['High'] == df['High'].rolling(window*2+1, center=True).max()]['High']
    lows = df[df['Low'] == df['Low'].rolling(window*2+1, center=True).min()]['Low']
    
    threshold = 0.002
    checked_highs = []
    for date, price in highs.items():
        for date2, price2 in highs.items():
            if date == date2: continue
            if abs(price - price2) / price < threshold:
                if not any(abs(h - price)/price < threshold for h in checked_highs):
                    features['EQH'].append({'price': (price+price2)/2, 'date': max(date, date2)})
                    checked_highs.append(price)
    
    checked_lows = []
    for date, price in lows.items():
        for date2, price2 in lows.items():
            if date == date2: continue
            if abs(price - price2) / price < threshold:
                if not any(abs(l - price)/price < threshold for l in checked_lows):
                    features['EQL'].append({'price': (price+price2)/2, 'date': max(date, date2)})
                    checked_lows.append(price)
                    
    return features

# --- 4. Chart Generation ---
def generate_chart_image(df, ticker, timeframe):
    try:
        plot_df = df.tail(60)
        if len(plot_df) < 30: return None, 0, 0, 0
        
        swing_high = plot_df['High'].max()
        swing_low = plot_df['Low'].min()
        eq = (swing_high + swing_low) / 2
        
        smc = identify_smc_features(plot_df)
        
        mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
        s  = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridcolor='#334155', facecolor='#1e293b')
        
        # Reduced figsize and dpi for memory optimization
        fig, axlist = mpf.plot(plot_df, type='candle', style=s, volume=False,
            title=dict(title=f"{ticker} - {timeframe}", color='white', size=10),
            figsize=(4, 2.5), returnfig=True)
        
        ax = axlist[0]
        x_min, x_max = ax.get_xlim()
        
        # Zones
        rect_prem = patches.Rectangle((x_min, eq), x_max-x_min, swing_high-eq, linewidth=0, facecolor='#ef4444', alpha=0.1)
        ax.add_patch(rect_prem)
        rect_disc = patches.Rectangle((x_min, swing_low), x_max-x_min, eq-swing_low, linewidth=0, facecolor='#10b981', alpha=0.1)
        ax.add_patch(rect_disc)
        ax.axhline(eq, color='#3b82f6', linestyle='-.', linewidth=1, alpha=0.6)

        # FVG Features
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
        print(f"Chart Error {ticker}: {e}")
        return None, 0, 0, 0

# --- 5. Analysis Loop (Background Worker) ---
def run_analysis_loop():
    global LATEST_HTML
    print("Background Worker Started...")
    
    while True:
        try:
            print(f"[{datetime.now()}] Starting Analysis Cycle...")
            
            # Download Data
            data_daily = yf.download(ALL_TICKERS + ["SPY"], period="1y", interval="1d", group_by='ticker', progress=False)
            data_hourly = yf.download(ALL_TICKERS, period="1mo", interval="1h", group_by='ticker', progress=False)
            
            # SPY Returns
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
                        # Handle MultiIndex
                        if isinstance(data_daily.columns, pd.MultiIndex):
                            try:
                                df_d = data_daily[t].dropna()
                                df_h = data_hourly[t].dropna()
                            except KeyError: continue
                        else: continue 
                        
                        if len(df_d) < 200: continue
                        
                        current_price = df_d['Close'].iloc[-1]
                        if isinstance(current_price, pd.Series): current_price = current_price.iloc[0]
                        
                        # Indicators
                        sma200 = df_d['Close'].rolling(200).mean().iloc[-1]
                        if isinstance(sma200, pd.Series): sma200 = sma200.iloc[0]
                        
                        vol = (df_d['Close'] * df_d['Volume']).rolling(21).mean().iloc[-1] * 21
                        if isinstance(vol, pd.Series): vol = vol.iloc[0]
                        
                        stock_ret = df_d['Close'].pct_change()
                        combo = pd.DataFrame({'S': stock_ret, 'M': spy_ret}).dropna()
                        beta = combo['S'].rolling(252).cov(combo['M']).iloc[-1] / combo['M'].rolling(252).var().iloc[-1] if len(combo)>30 else 0
                        
                        pass_filter = (current_price > sma200 and vol > FILTER_MIN_MONTHLY_VOL and beta >= FILTER_MIN_BETA)

                        # Generate Daily Chart first to get Levels
                        chart_res = generate_chart_image(df_d, t, "Daily")
                        if not chart_res or chart_res[0] is None: continue
                        img_d, tp, sl, eq = chart_res
                        
                        # Determine Signal
                        range_len = tp - sl
                        pos_pct = (current_price - sl) / range_len if range_len > 0 else 0.5
                        signal = "LONG" if pos_pct < 0.4 else "WAIT"
                        cls = "b-long" if signal == "LONG" else "b-wait"
                        
                        # --- Optimization: Only generate hourly chart if needed ---
                        should_draw_hourly = pass_filter or (signal == "LONG")
                        img_h = ""
                        if should_draw_hourly:
                            h_res = generate_chart_image(df_h if not df_h.empty else df_d, t, "Hourly")
                            if h_res: img_h = h_res[0]

                        # AI Text
                        deployment_html = ""
                        trend_str = "Bullish (>200MA)" if current_price > sma200 else "Neutral"
                        
                        if signal == "LONG":
                            entry = current_price
                            stop_loss = sl * 0.98
                            take_profit = tp
                            rr = (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
                            
                            deployment_html = f"""
                            <div class="deploy-box long">
                                <div class="deploy-title">✅ LONG SETUP</div>
                                <ul class="deploy-list">
                                    <li><b>Entry:</b> ${entry:.2f} (Discount Zone)</li>
                                    <li><b>Stop Loss:</b> ${stop_loss:.2f} (Below SSL)</li>
                                    <li><b>Target:</b> ${take_profit:.2f} (BSL)</li>
                                    <li><b>R:R:</b> {rr:.1f}R ({trend_str})</li>
                                </ul>
                            </div>"""
                        else:
                            discount_entry = sl + (range_len * 0.4)
                            deployment_html = f"""
                            <div class="deploy-box wait">
                                <div class="deploy-title">⏳ WAIT</div>
                                <ul class="deploy-list">
                                    <li>Price is in Premium zone.</li>
                                    <li>Wait for pullback to <b>${eq:.2f}</b>.</li>
                                    <li>Ideal Entry: Below ${discount_entry:.2f}.</li>
                                </ul>
                            </div>"""
                        
                        deploy_clean = deployment_html.replace('"', '&quot;').replace('\n', '')

                        cards_in_sector += f"""
                        <div class="card" onclick="openModal('{t}', '{img_d}', '{img_h}', '{signal}', '{deploy_clean}')">
                            <div class="head">
                                <div><div class="code">{t}</div><div class="price">${current_price:.2f}</div></div>
                                <span class="badge {cls}">{signal}</span>
                            </div>
                            <div class="hint">Tap for Strategy ↗</div>
                        </div>
                        """
                        
                        if pass_filter:
                            passed_count += 1
                            screener_rows += f"<tr><td><b>{t}</b></td><td>${current_price:.2f}</td><td class='g'>Pass</td><td>{beta:.2f}</td><td><span class='badge {cls}'>{signal}</span></td></tr>"

                    except Exception as e:
                        continue
                        
                if cards_in_sector:
                    sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards_in_sector}</div>"

            # --- Generate Final HTML ---
            LATEST_HTML = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>DailyDip Pro AI</title>
            <style>
                :root {{ --bg:#0f172a; --card:#1e293b; --text:#f8fafc; --acc:#3b82f6; --g:#10b981; --r:#ef4444; --y:#fbbf24; }}
                body {{ background:var(--bg); color:var(--text); font-family:-apple-system, sans-serif; margin:0; padding:10px; }}
                .tabs {{ display:flex; gap:10px; border-bottom:1px solid #334155; padding-bottom:10px; margin-bottom:15px; position:sticky; top:0; background:var(--bg); z-index:10; }}
                .tab {{ padding:8px 16px; background:#334155; border-radius:6px; cursor:pointer; color:#94a3b8; font-weight:bold; font-size:0.9rem; }}
                .tab.active {{ background:var(--acc); color:white; }}
                .content {{ display:none; }} .content.active {{ display:block; }}
                .sector-title {{ border-left:4px solid var(--acc); padding-left:10px; margin:20px 0 10px; color:#e2e8f0; font-size:1.1rem; }}
                .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; }}
                .card {{ background:var(--card); border:1px solid #334155; border-radius:8px; padding:12px; cursor:pointer; }}
                .head {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:5px; }}
                .code {{ font-size:1.1rem; font-weight:900; }} .price {{ color:#94a3b8; font-family:monospace; }}
                .badge {{ padding:3px 6px; border-radius:4px; font-size:0.75rem; font-weight:bold; }}
                .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
                .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #334155; }}
                .hint {{ font-size:0.7rem; color:var(--acc); text-align:right; margin-top:5px; opacity:0.8; }}
                table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; font-size:0.9rem; }}
                th, td {{ padding:10px; text-align:left; border-bottom:1px solid #334155; }}
                th {{ background:#334155; color:#94a3b8; }} .g {{ color:var(--g); }}
                .modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); z-index:999; justify-content:center; align-items:start; overflow-y:auto; padding:10px; }}
                .m-content {{ background:var(--card); width:100%; max-width:600px; padding:15px; border-radius:12px; border:1px solid #475569; margin-top:10px; }}
                .deploy-box {{ padding:15px; border-radius:8px; margin-bottom:15px; border-left:4px solid; }}
                .deploy-box.long {{ background:rgba(16,185,129,0.1); border-color:var(--g); }}
                .deploy-box.wait {{ background:rgba(251,191,36,0.1); border-color:var(--y); }}
                .chart-lbl {{ color:var(--acc); font-size:0.85rem; font-weight:bold; display:block; margin-bottom:5px; }}
                .m-content img {{ width:100%; border-radius:6px; border:1px solid #334155; margin-bottom:15px; }}
                .no-chart {{ padding:20px; text-align:center; color:#64748b; font-size:0.9rem; border:1px dashed #334155; border-radius:6px; margin-bottom:15px; }}
                .close-btn {{ width:100%; padding:12px; background:var(--acc); color:white; border:none; border-radius:6px; font-weight:bold; font-size:1rem; cursor:pointer; }}
            </style>
            </head>
            <body>
            <div class="tabs">
                <div class="tab active" onclick="setTab('overview', this)">📊 Market</div>
                <div class="tab" onclick="setTab('screener', this)">🔍 Screener ({passed_count})</div>
            </div>

            <div id="overview" class="content active">
                {sector_html_blocks}
            </div>

            <div id="screener" class="content">
                <table><thead><tr><th>Ticker</th><th>Price</th><th>Trend</th><th>Beta</th><th>Signal</th></tr></thead><tbody>{screener_rows}</tbody></table>
            </div>

            <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
                <div class="m-content" onclick="event.stopPropagation()">
                    <h2 id="m-ticker" style="margin-top:0; color:white"></h2>
                    <div id="m-deploy"></div>
                    <div class="chart-box"><span class="chart-lbl">📅 Daily Chart</span><div id="chart-d"></div></div>
                    <div class="chart-box"><span class="chart-lbl">⏱️ Hourly Chart</span><div id="chart-h"></div></div>
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
            function openModal(ticker, d_src, h_src, signal, deploy_html) {{
                document.getElementById('modal').style.display = 'flex';
                document.getElementById('m-ticker').innerText = ticker + " (" + signal + ")";
                document.getElementById('m-deploy').innerHTML = deploy_html;
                
                document.getElementById('chart-d').innerHTML = d_src ? '<img src="' + d_src + '">' : '<div class="no-chart">Chart not loaded</div>';
                document.getElementById('chart-h').innerHTML = h_src ? '<img src="' + h_src + '">' : '<div class="no-chart">Hourly chart not needed</div>';
            }}
            </script>
            </body>
            </html>
            """
            
            print(f"[{datetime.now()}] Analysis Complete. Updated HTML.")
            time.sleep(3600) # Update every hour

        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(60)

# --- Start Background Worker ---
analysis_thread = threading.Thread(target=run_analysis_loop, daemon=True)
analysis_thread.start()

# --- Flask Route ---
@app.route('/')
def home():
    return render_template_string(LATEST_HTML)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
