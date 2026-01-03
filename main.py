import os
import matplotlib
# 1. 強制設定後台繪圖 (最優先)
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

# --- 1. 觀察清單設定 ---

# 🔥🔥🔥【每日暫時觀察區 (自動過濾)】🔥🔥🔥
# 系統會掃描這裡的股票，如果是 WAIT 就會自動刪除，只有 LONG 會顯示
TEMP_WATCHLIST = [
    "IRWD", "SKYT", "SLS", "PEPG", "TROO", "CTRN", "BCAR", "ARDX", "RCAT", "MLAC", 
    "SNDK", "ONDS", "VELO", "APLD", "TIGR", "FLNC", "SERV", "ACMR", "FTAI", "ZURA"
]

# 固定板塊
SECTORS = {
    "🔥 熱門交易": ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "SOFI", "MARA", "MSTR", "SMCI", "COIN"],
    "💎 科技巨頭": ["MSFT", "AMZN", "GOOGL", "META", "NFLX", "CRM", "ADBE"],
    "⚡ 半導體": ["TSM", "AVGO", "MU", "INTC", "ARM", "QCOM", "TXN", "AMAT"],
    "🚀 成長股": ["HOOD", "DKNG", "RBLX", "U", "CVNA", "OPEN", "SHOP", "NET"],
    "🏦 金融與消費": ["JPM", "V", "COST", "MCD", "NKE", "LLY", "WMT", "DIS", "SBUX"],
    "📉 指數 ETF": ["SPY", "QQQ", "IWM", "TQQQ", "SQQQ"]
}

# --- 2. 市場大盤分析 ---
def get_market_condition():
    try:
        print("🔍 Checking Market...")
        spy = yf.Ticker("SPY").history(period="6mo")
        qqq = yf.Ticker("QQQ").history(period="6mo")
        
        if spy.empty or qqq.empty: return "NEUTRAL", "數據不足", 0

        spy_50 = spy['Close'].rolling(50).mean().iloc[-1]
        spy_curr = spy['Close'].iloc[-1]
        qqq_50 = qqq['Close'].rolling(50).mean().iloc[-1]
        qqq_curr = qqq['Close'].iloc[-1]
        
        is_bullish = (spy_curr > spy_50) and (qqq_curr > qqq_50)
        is_bearish = (spy_curr < spy_50) and (qqq_curr < qqq_50)
        
        if is_bullish: return "BULLISH", "🟢 市場順風 (大盤 > 50MA)", 5
        elif is_bearish: return "BEARISH", "🔴 市場逆風 (大盤 < 50MA)", -10
        else: return "NEUTRAL", "🟡 市場震盪", 0
    except: return "NEUTRAL", "Check Failed", 0

# --- 3. 數據獲取 ---
def fetch_data_safe(ticker, period, interval):
    try:
        dat = yf.Ticker(ticker).history(period=period, interval=interval)
        if dat is None or dat.empty: return None
        if not isinstance(dat.index, pd.DatetimeIndex): dat.index = pd.to_datetime(dat.index)
        dat = dat.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        return dat
    except: return None

# --- 4. 技術指標 (RSI, RVOL) ---
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
    
    # Golden Cross
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    golden_cross = False
    if len(sma50) > 5:
        if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-5] <= sma200.iloc[-5]:
            golden_cross = True
            
    # Trend
    trend_bullish = sma50.iloc[-1] > sma200.iloc[-1] if len(sma200) > 0 else False
    
    # Perf
    if len(df) > 30:
        perf_30d = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100
    else:
        perf_30d = 0
    
    return rsi, rvol, golden_cross, trend_bullish, perf_30d

# --- 5. 評分系統 ---
def calculate_quality_score(df, entry, sl, tp, is_bullish, market_bonus, found_sweep, indicators):
    try:
        score = 60 + market_bonus
        reasons = []
        rsi, rvol, golden_cross, trend, perf_30d = indicators
        
        strategies = 0
        if found_sweep: strategies += 1
        if golden_cross: strategies += 1
        if 40 <= rsi.iloc[-1] <= 55: strategies += 1
        
        # RR
        risk = entry - sl
        reward = tp - entry
        rr = reward / risk if risk > 0 else 0
        if rr >= 3.0: 
            score += 15
            reasons.append(f"💰 盈虧比極佳 ({rr:.1f}R)")
        elif rr >= 2.0: 
            score += 10
            reasons.append(f"💰 盈虧比優秀 ({rr:.1f}R)")

        # RSI
        curr_rsi = rsi.iloc[-1]
        if 40 <= curr_rsi <= 55: 
            score += 10
            reasons.append(f"📉 RSI 完美回調 ({int(curr_rsi)})")
        elif curr_rsi > 70: score -= 15

        # RVOL
        curr_rvol = rvol.iloc[-1]
        if curr_rvol > 1.5:
            score += 10
            reasons.append(f"🔥 爆量確認 (Vol {curr_rvol:.1f}x)")
        elif curr_rvol > 1.1: score += 5

        # Sweep
        if found_sweep:
            score += 20
            reasons.append("💧 觸發流動性獵殺 (Sweep)")
            
        # Golden Cross
        if golden_cross:
            score += 10
            reasons.append("✨ 出現黃金交叉")

        # Distance
        close = df['Close'].iloc[-1]
        dist_pct = abs(close - entry) / entry
        if dist_pct < 0.01: 
            score += 15
            reasons.append("🎯 狙擊入場區")
            
        if trend: 
            score += 5
            reasons.append("📈 長期趨勢向上")

        if market_bonus > 0: reasons.append("🌍 大盤順風車 (+5)")
        if market_bonus < 0: reasons.append("🌪️ 逆大盤風險 (-10)")

        return min(max(int(score), 0), 99), reasons, rr, rvol.iloc[-1], perf_30d, strategies
    except: return 50, [], 0, 0, 0, 0

# --- 6. SMC 運算 ---
def calculate_smc(df):
    try:
        window = 50
        recent = df.tail(window)
        bsl = float(recent['High'].max())
        ssl_long = float(recent['Low'].min())
        
        eq = (bsl + ssl_long) / 2
        
        best_entry = eq
        found_fvg = False
        found_sweep = False
        
        last_3 = recent.tail(3)
        check_low = recent['Low'].iloc[:-3].tail(10).min() 
        
        for i in range(len(last_3)):
            candle = last_3.iloc[i]
            if candle['Low'] < check_low and candle['Close'] > check_low:
                found_sweep = True
                best_entry = check_low 
                break
        
        for i in range(2, len(recent)):
            if recent['Low'].iloc[i] > recent['High'].iloc[i-2]:
                fvg = float(recent['Low'].iloc[i])
                if fvg < eq:
                    if not found_sweep: best_entry = fvg
                    found_fvg = True
                    break
                    
        return bsl, ssl_long, eq, best_entry, ssl_long*0.99, found_fvg, found_sweep
    except:
        last = float(df['Close'].iloc[-1])
        return last*1.05, last*0.95, last, last, last*0.94, False, False

# --- 7. 繪圖核心 ---
def create_error_image(msg):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.text(0.5, 0.5, msg, color='white', ha='center', va='center')
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def generate_chart(df, ticker, title, entry, sl, tp, is_wait, found_sweep):
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

        if found_sweep:
            lowest = plot_df['Low'].min()
            ax.text(x_min + 2, lowest, "💧 SWEEP", color='#fbbf24', fontsize=12, fontweight='bold', va='bottom')

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
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    except: return create_error_image("Plot Error")

# --- 8. 單一股票處理 ---
def process_ticker(t, app_data_dict, market_bonus):
    try:
        time.sleep(0.3)
        df_d = fetch_data_safe(t, "1y", "1d")
        if df_d is None or len(df_d) < 50: return None
        df_h = fetch_data_safe(t, "1mo", "1h")
        if df_h is None or df_h.empty: df_h = df_d

        curr = float(df_d['Close'].iloc[-1])
        sma200 = float(df_d['Close'].rolling(200).mean().iloc[-1])
        if pd.isna(sma200): sma200 = curr

        bsl, ssl, eq, entry, sl, found_fvg, found_sweep = calculate_smc(df_d)
        tp = bsl

        is_bullish = curr > sma200
        in_discount = curr < eq
        signal = "LONG" if (is_bullish and in_discount and (found_fvg or found_sweep)) else "WAIT"
        
        indicators = calculate_indicators(df_d)
        score, reasons, rr, rvol, perf_30d, strategies = calculate_quality_score(df_d, entry, sl, tp, is_bullish, market_bonus, found_sweep, indicators)
        rvol_val = rvol.iloc[-1]

        is_wait = (signal == "WAIT")
        should_plot = (signal == "LONG") or found_sweep or (score >= 80)
        
        if should_plot:
            img_d = generate_chart(df_d, t, "Daily SMC", entry, sl, tp, is_wait, found_sweep)
            img_h = generate_chart(df_h, t, "Hourly Entry", entry, sl, tp, is_wait, found_sweep)
        else:
            img_d, img_h = "", ""

        cls = "b-long" if signal == "LONG" else "b-wait"
        score_color = "#10b981" if score >= 85 else ("#3b82f6" if score >= 70 else "#fbbf24")
        
        elite_html = ""
        if score >= 75 or found_sweep or rvol_val > 1.2 or signal == "LONG":
            reasons_html = "".join([f"<li>✅ {r}</li>" for r in reasons])
            confluence_text = ""
            if strategies >= 2:
                confluence_text = f"🔥 <b>策略共振：</b> 同時觸發 {strategies} 種訊號，可靠度極高。"
            sweep_text = ""
            if found_sweep:
                sweep_text = "<div style='margin-top:8px; padding:8px; background:rgba(251,191,36,0.1); border-left:3px solid #fbbf24; color:#fcd34d; font-size:0.85rem;'><b>⚠️ 偵測到流動性獵殺 (Sweep)</b></div>"
            elite_html = f"<div style='background:rgba(16,185,129,0.1); border:1px solid #10b981; padding:12px; border-radius:8px; margin:10px 0;'><div style='font-weight:bold; color:#10b981; margin-bottom:5px;'>💎 AI 分析 (Score {score})</div><div style='font-size:0.85rem; color:#e2e8f0; margin-bottom:8px;'>{confluence_text}</div><ul style='margin:0; padding-left:20px; font-size:0.8rem; color:#d1d5db;'>{reasons_html}</ul>{sweep_text}</div>"
        
        if signal == "LONG":
            ai_html = f"<div class='deploy-box long'><div class='deploy-title'>✅ LONG SETUP</div><div style='display:flex;justify-content:space-between;border-bottom:1px solid #333;padding-bottom:5px;margin-bottom:5px;'><span>🏆 評分: <b style='color:{score_color};font-size:1.1em'>{score}</b></span><span>💰 RR: <b style='color:#10b981'>{rr:.1f}R</b></span></div><div style='font-size:0.8rem; color:#94a3b8; margin-bottom:5px;'>📈 近30日績效: {perf_30d:+.1f}%</div>{elite_html}<ul class='deploy-list' style='margin-top:10px'><li>TP: ${tp:.2f}</li><li>Entry: ${entry:.2f}</li><li>SL: ${sl:.2f}</li></ul></div>"
        else:
            reason = "無FVG/Sweep" if (not found_fvg and not found_sweep) else ("逆勢" if not is_bullish else "溢價區")
            ai_html = f"<div class='deploy-box wait'><div class='deploy-title'>⏳ WAIT</div><div>評分: <b style='color:#94a3b8'>{score}</b></div><ul class='deploy-list'><li>狀態: {reason}</li><li>參考入場: ${entry:.2f}</li></ul></div>"
            
        app_data_dict[t] = {"signal": signal, "deploy": ai_html, "img_d": img_d, "img_h": img_h, "score": score, "rvol": rvol_val}
        
        return {"ticker": t, "price": curr, "signal": signal, "cls": cls, "score": score, "rvol": rvol_val, "perf": perf_30d}
    except Exception as e:
        print(f"Err {t}: {e}")
        return None

# --- 9. 主程式 ---
def main():
    print("🚀 啟動分析程式 (封面顯示爆量強化版)...")
    
    market_status, market_text, market_bonus = get_market_condition()
    market_color = "#10b981" if market_status == "BULLISH" else ("#ef4444" if market_status == "BEARISH" else "#fbbf24")
    
    APP_DATA, sector_html_blocks, screener_rows_list = {}, "", []

    # 1. 處理暫時觀察名單
    if TEMP_WATCHLIST:
        print(f"🔎 掃描暫時名單 ({len(TEMP_WATCHLIST)} 隻)...")
        valid_temp_stocks = []
        for t in TEMP_WATCHLIST:
            res = process_ticker(t, APP_DATA, market_bonus)
            if res:
                if res['signal'] == "WAIT":
                    if t in APP_DATA: del APP_DATA[t]
                    print(f"   🗑️ {t} (WAIT) -> 移除")
                else:
                    valid_temp_stocks.append(t)
                    screener_rows_list.append(res)
                    print(f"   ✨ {t} (LONG) -> 保留")
        if valid_temp_stocks:
            SECTORS["👀 每日快篩 (LONG Only)"] = valid_temp_stocks
    
    # 2. 處理固定板塊
    for sector, tickers in SECTORS.items():
        sector_results = []
        for t in tickers:
            if t in APP_DATA:
                data = APP_DATA[t]
                res_obj = {'ticker': t, 'score': data['score'], 'signal': data['signal'], 'rvol': data.get('rvol', 0)}
                sector_results.append(res_obj)
            else:
                res = process_ticker(t, APP_DATA, market_bonus)
                if res:
                    sector_results.append(res)
                    if res['signal'] == "LONG":
                        screener_rows_list.append(res)

        sector_results.sort(key=lambda x: x['score'], reverse=True)
        
        cards = ""
        for item in sector_results:
            t = item['ticker']
            if t not in APP_DATA: continue
            data = APP_DATA[t]
            signal = data['signal']
            score = data['score']
            rvol = data.get('rvol', 0)
            
            # 🔥 這裡處理卡片顯示的爆量
            if rvol > 1.2:
                rvol_tag = f"<div style='color:#f472b6;font-weight:bold;margin-top:2px;font-size:0.8rem'>Vol {rvol:.1f}x 🔥</div>"
            else:
                rvol_tag = f"<div style='color:#64748b;margin-top:2px;font-size:0.75rem'>Vol {rvol:.1f}x</div>"
            
            cls = "b-long" if signal == "LONG" else "b-wait"
            s_color = "#10b981" if score >= 85 else ("#3b82f6" if score >= 70 else "#fbbf24")
            
            cards += f"""
            <div class='card' onclick="openModal('{t}')">
                <div class='head'>
                    <div><div class='code'>{t}</div><div style='font-size:0.7rem;color:#666;margin-top:3px'>Score <span style='color:{s_color}'>{score}</span></div></div>
                    <div style='text-align:right'>
                        <span class='badge {cls}'>{signal}</span>
                        {rvol_tag}
                    </div>
                </div>
            </div>
            """
            
        if cards: sector_html_blocks += f"<h3 class='sector-title'>{sector}</h3><div class='grid'>{cards}</div>"

    # 去重
    seen = set()
    unique_screener = []
    for r in screener_rows_list:
        if r['ticker'] not in seen:
            unique_screener.append(r)
            seen.add(r['ticker'])
    unique_screener.sort(key=lambda x: x['score'], reverse=True)
    
    screener_html = ""
    for res in unique_screener:
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
    .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap:10px; }}
    .card {{ background:var(--card); border:1px solid #333; border-radius:8px; padding:12px; cursor:pointer; }}
    .head {{ display:flex; justify-content:space-between; align-items:start; }}
    .code {{ font-weight:900; font-size:1.1rem; }} 
    .badge {{ padding:2px 6px; border-radius:4px; font-size:0.75rem; font-weight:bold; display:inline-block; }}
    .b-long {{ background:rgba(16,185,129,0.2); color:var(--g); border:1px solid var(--g); }}
    .b-wait {{ background:rgba(148,163,184,0.1); color:#94a3b8; border:1px solid #555; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
    th, td {{ padding:8px; text-align:left; border-bottom:1px solid #333; }}
    .g {{ color:var(--g); font-weight:bold; }}
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
        </div>
        
        <div id="overview" class="content active">{sector_html_blocks if sector_html_blocks else '<div style="text-align:center;padding:50px">載入中...</div>'}</div>
        <div id="screener" class="content"><table><thead><tr><th>Ticker</th><th>Price</th><th>Score</th><th>Signal</th></tr></thead><tbody>{screener_html}</tbody></table></div>
        
        <div class="time">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>

        <div id="modal" class="modal" onclick="document.getElementById('modal').style.display='none'">
            <div class="m-content" onclick="event.stopPropagation()">
                <h2 id="m-ticker" style="margin-top:0"></h2>
                <div id="m-deploy"></div>
                <div><b>Daily SMC</b><div id="chart-d"></div></div>
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
            const imgD = data.img_d ? '<img src="'+data.img_d+'">' : '<div style="padding:20px;text-align:center;color:#666">No Chart Available</div>';
            const imgH = data.img_h ? '<img src="'+data.img_h+'">' : '';
            
            document.getElementById('modal').style.display = 'flex';
            document.getElementById('m-ticker').innerText = ticker;
            document.getElementById('m-deploy').innerHTML = data.deploy;
            document.getElementById('chart-d').innerHTML = imgD;
            document.getElementById('chart-h').innerHTML = imgH;
        }}
        </script>
    </body></html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("✅ index.html generated!")

if __name__ == "__main__":
    main()
