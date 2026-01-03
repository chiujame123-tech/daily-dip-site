import pandas as pd
import yfinance as yf
import time
import os

# --- 設定 ---
CSV_FILE = "nasdaq_mid_large_caps (2).csv"
MIN_VOLUME_MULTIPLIER = 1.5  # 只顯示量大於 1.5x 的
MIN_SCORE = 70               # 只顯示分數及格的

def fetch_data_quick(ticker):
    try:
        # 只抓 50 天數據，速度最快
        df = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if df is None or len(df) < 20: return None
        return df
    except: return None

def analyze_stock(ticker, df):
    # 1. 計算 RVOL
    vol = df['Volume']
    vol_ma = vol.rolling(10).mean()
    rvol = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 0
    
    # 2. 簡單趨勢判斷 (價格 > 50MA)
    close = df['Close']
    sma50 = close.rolling(50).mean()
    is_bullish = close.iloc[-1] > sma50.iloc[-1]
    
    # 3. 簡單 SMC 分數模擬
    score = 60
    if rvol > 1.5: score += 10
    if is_bullish: score += 10
    
    # 計算漲跌幅
    change_pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
    
    return {
        "Ticker": ticker,
        "Price": close.iloc[-1],
        "Change%": change_pct,
        "RVOL": rvol,
        "Trend": "Bull" if is_bullish else "Bear",
        "Score": score
    }

def main():
    print(f"🚀 啟動全市場掃描器 (Target: RVOL > {MIN_VOLUME_MULTIPLIER}x)...")
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到 {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    tickers = df['Stock Ticker'].dropna().astype(str).tolist()
    
    print(f"📦 總共載入 {len(tickers)} 隻股票。開始掃描...")
    print("-" * 60)
    print(f"{'Ticker':<8} {'Price':<10} {'Change%':<10} {'RVOL':<10} {'Trend':<8}")
    print("-" * 60)
    
    found_count = 0
    for i, t in enumerate(tickers):
        # 進度顯示 (每 10 隻更新一次)
        if i % 100 == 0: print(f"🔍 Scanning... [{i}/{len(tickers)}]")
            
        df_stock = fetch_data_quick(t)
        if df_stock is None: continue
        
        res = analyze_stock(t, df_stock)
        
        # 🔥 篩選條件：爆量 且 趨勢向上
        if res['RVOL'] >= MIN_VOLUME_MULTIPLIER and res['Trend'] == "Bull":
            # 亮點顯示：如果漲幅 > 5% 或 RVOL > 2.0，加強顯示
            marker = "🔥" if (res['Change%'] > 5 or res['RVOL'] > 2.0) else ""
            
            print(f"{res['Ticker']:<8} ${res['Price']:<9.2f} {res['Change%']:+.2f}%   {res['RVOL']:.1f}x      {res['Trend']} {marker}")
            found_count += 1
            
    print("-" * 60)
    print(f"✅ 掃描完成！共發現 {found_count} 隻爆量潛力股。")

if __name__ == "__main__":
    main()
