# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v10.0 Ultimate (Full Uncut Edition)
=============================================================

這才是真正的完整版，無任何功能刪減！
✅ 保留 v9.5 全部功能: Qullamaggie 歷史回測、真實 IV、PCR、圖表引擎
✅ 融入 v10.0 新功能: 
    1. AI Paper Trade (自動捕捉今日突破並給出交易筆記)
    2. Minervini 嚴格 VCP 趨勢模板
    3. SPY + QQQ 納指雙重環境監控
    4. 專業側邊欄 (Sidebar) 終端機 UI

Author: Pro Trader AI (Powered by Gemini)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================
# ⚙️ CONFIGURATION & STATE
# ============================================
@dataclass
class Config:
    PAGE_TITLE: str = "Market Radar v10.0"
    PAGE_ICON: str = "🎯"

CONFIG = Config()

st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide", initial_sidebar_state="expanded")

if 'paper_trades' not in st.session_state:
    st.session_state.paper_trades = {}

STOCK_UNIVERSE = {
    'Market Leaders': ['NVDA', 'META', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'LLY', 'AVGO', 'TSLA', 'AMD', 'CRM', 'NOW', 'PANW', 'CRWD', 'NFLX', 'COST', 'ISRG', 'LULU', 'CMG', 'FICO'],
    'Semiconductors': ['NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ARM', 'SMCI', 'INTC', 'ASML', 'SNPS', 'ON', 'NXPI', 'ADI', 'MCHP', 'TXN'],
    'Software & Cloud': ['MSFT', 'CRM', 'ADBE', 'NOW', 'INTU', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'NET', 'MDB', 'PLTR', 'ZS', 'FTNT', 'WDAY', 'TEAM', 'HUBS', 'OKTA', 'BILL', 'DOCU'],
    'High Growth': ['NVDA', 'SMCI', 'ARM', 'PLTR', 'COIN', 'MSTR', 'AFRM', 'SOFI', 'HOOD', 'UPST', 'RBLX', 'DKNG', 'SHOP', 'SQ', 'MELI', 'SE', 'NU', 'GRAB', 'BILL', 'CELH'],
    'Blue Chips (Put)': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'V', 'MA', 'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'MCD', 'DIS', 'NFLX', 'ADBE', 'CRM', 'UNH', 'LLY', 'MRK', 'ABBV', 'TMO', 'ACN', 'CSCO', 'ORCL', 'IBM', 'INTC'],
    'Dividend Stocks': ['JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'HD', 'VZ', 'T', 'XOM', 'CVX', 'IBM', 'CSCO', 'INTC', 'MRK', 'ABBV', 'PFE', 'BMY', 'MMM', 'CAT']
}
ALL_STOCKS = list(set([s for stocks in STOCK_UNIVERSE.values() for s in stocks]))
ALL_STOCKS.sort()

# ============================================
# 📡 BATCH DATA FETCHER 
# ============================================
class BatchDataFetcher:
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def batch_download(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        if not tickers: return {}
        try:
            data = yf.download(tickers, period=period, progress=False, group_by='ticker', threads=True, timeout=30)
            result = {}
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].copy().dropna(how='all')
                            if len(df) > 0: result[ticker] = df
                    except: continue
            else:
                if len(tickers) == 1 and len(data) > 0: result[tickers[0]] = data
            return result
        except: return {}

    @staticmethod
    @st.cache_data(ttl=1800)
    def get_single_stock(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=15)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df if df is not None and len(df) > 0 else None
        except: return None

# ============================================
# 🧮 TECHNICAL ANALYSIS
# ============================================
class TechnicalAnalysis:
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        return dx.rolling(period).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return sma + std * 2, sma - std * 2

    @staticmethod
    def bollinger_band_width(prices: pd.Series, period: int = 20) -> pd.Series:
        upper, lower = TechnicalAnalysis.bollinger_bands(prices, period)
        return (upper - lower) / prices.rolling(period).mean()
        
    @staticmethod
    def macd(prices: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'hist': hist})
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def adr_percent(df: pd.DataFrame, period: int = 20) -> pd.Series:
        daily_range = (df['High'] / df['Low'] - 1) * 100
        return daily_range.rolling(period).mean()
    
    @staticmethod
    def rs_rating(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
        if len(stock_df) < 63 or spy_df is None or len(spy_df) < 63: return 50
        periods, weights, score = [21, 42, 63], [0.4, 0.3, 0.3], 0
        for period, weight in zip(periods, weights):
            try:
                stock_ret = (float(stock_df['Close'].iloc[-1]) / float(stock_df['Close'].iloc[-period]) - 1) * 100
                spy_ret = (float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-period]) - 1) * 100
                score += (stock_ret - spy_ret) * weight
            except: pass
        return max(1, min(99, 50 + (score / 30) * 49))
    
    @staticmethod
    def calculate_beta(stock_df: pd.DataFrame, spy_df: pd.DataFrame, period: int = 252) -> float:
        if len(stock_df) < period or spy_df is None or len(spy_df) < period: return 1.0
        try:
            stock_returns, spy_returns = stock_df['Close'].pct_change().tail(period).dropna(), spy_df['Close'].pct_change().tail(period).dropna()
            common_idx = stock_returns.index.intersection(spy_returns.index)
            covariance, variance = np.cov(stock_returns.loc[common_idx], spy_returns.loc[common_idx])[0][1], np.var(spy_returns.loc[common_idx])
            return round(covariance / variance if variance > 0 else 1.0, 2)
        except: return 1.0
    
    @staticmethod
    def calculate_support_levels(df: pd.DataFrame) -> Dict:
        if len(df) < 100: return {}
        close = float(df['Close'].iloc[-1])
        sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
        sma100 = float(df['Close'].rolling(100).mean().iloc[-1])
        sma200 = float(df['Close'].rolling(200).mean().iloc[-1]) if len(df) >= 200 else sma50
        _, lower_bb = TechnicalAnalysis.bollinger_bands(df['Close'])
        lower_bb_val = float(lower_bb.iloc[-1])
        
        supports = [('SMA50', sma50), ('SMA100', sma100), ('SMA200', sma200), ('布林下軌', lower_bb_val)]
        valid = [(name, price) for name, price in supports if price < close]
        
        if valid:
            nearest = max(valid, key=lambda x: x[1])
            return {
                'nearest_support': nearest[0], 
                'nearest_support_price': nearest[1], 
                'distance_pct': round((close - nearest[1]) / close * 100, 2),
                'sma200_val': sma200
            }
        return {'nearest_support': '無 (跌破所有支撐)', 'distance_pct': 99.9, 'sma200_val': sma200}
    
    @staticmethod
    def find_swing_points(df: pd.DataFrame, lookback: int = 60, window: int = 5) -> Dict:
        if len(df) < lookback: return {'swing_highs': [], 'swing_lows': [], 'contractions': []}
        recent, highs, lows, dates = df.tail(lookback), df['High'].tail(lookback).values, df['Low'].tail(lookback).values, df.tail(lookback).index
        swing_highs = [{'date': dates[i], 'price': highs[i], 'index': i} for i in range(window, len(highs) - window) if highs[i] == max(highs[i-window:i+window+1])]
        swing_lows = [{'date': dates[i], 'price': lows[i], 'index': i} for i in range(window, len(lows) - window) if lows[i] == min(lows[i-window:i+window+1])]
        contractions = []
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            for i in range(min(len(swing_highs), len(swing_lows)) - 1):
                h = swing_highs[i]['price'] if i < len(swing_highs) else swing_highs[-1]['price']
                l = swing_lows[i]['price'] if i < len(swing_lows) else swing_lows[-1]['price']
                contractions.append((h - l) / l * 100)
        return {'swing_highs': swing_highs, 'swing_lows': swing_lows, 'contractions': contractions}
    
    @staticmethod
    def analyze_volume_signature(df: pd.DataFrame, period: int = 20) -> Dict:
        if len(df) < period: return {'up_vol': 0, 'down_vol': 0, 'ratio': 1, 'is_healthy': False, 'dry_up': False}
        recent = df.tail(period)
        up_days, down_days = recent[recent['Close'] >= recent['Open']], recent[recent['Close'] < recent['Open']]
        up_vol = float(up_days['Volume'].mean()) if len(up_days) > 0 else 0
        down_vol = float(down_days['Volume'].mean()) if len(down_days) > 0 else 0
        ratio = up_vol / down_vol if down_vol > 0 else 2.0
        avg_vol_50, recent_5_vol = float(df['Volume'].tail(50).mean()), float(df['Volume'].tail(5).mean())
        return {'up_vol': up_vol, 'down_vol': down_vol, 'ratio': round(ratio, 2), 'is_healthy': ratio > 1.0, 'dry_up': recent_5_vol < avg_vol_50 * 0.6, 'dry_up_ratio': round(recent_5_vol / avg_vol_50, 2) if avg_vol_50 > 0 else 1}
    
    @staticmethod
    def estimate_hv_rank(df: pd.DataFrame, period: int = 252) -> float:
        if len(df) < period: return 50
        try:
            hv_values = (df['Close'].pct_change().dropna().rolling(20).std() * np.sqrt(252) * 100).tail(period).dropna()
            if len(hv_values) < 20 or hv_values.max() == hv_values.min(): return 50
            return round(max(0, min(100, (hv_values.iloc[-1] - hv_values.min()) / (hv_values.max() - hv_values.min()) * 100)), 1)
        except: return 50

# ============================================
# 📊 OPTIONS & SENTIMENT
# ============================================
class RealIVCalculator:
    @staticmethod
    def get_real_iv(ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            exp_dates = stock.options
            if not exp_dates: return {'iv': None, 'status': 'No options'}
            best_exp = next((exp for exp in exp_dates if 20 <= (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days <= 60), exp_dates[0])
            chain, current_price = stock.option_chain(best_exp), float(stock.history(period='1d')['Close'].iloc[-1])
            atm_calls = chain.calls[abs(chain.calls['strike'] - current_price) / current_price < 0.03]
            atm_puts = chain.puts[abs(chain.puts['strike'] - current_price) / current_price < 0.03]
            iv_values = [iv for df in (atm_calls, atm_puts) if 'impliedVolatility' in df.columns for iv in df['impliedVolatility'].dropna()]
            if iv_values: return {'iv': round(np.mean(iv_values) * 100, 1), 'expiry': best_exp, 'status': 'OK'}
            return {'iv': None, 'status': 'No IV data'}
        except Exception as e: return {'iv': None, 'status': f'Error: {str(e)}'}

class PCRCalculator:
    @staticmethod
    def get_pcr(ticker: str) -> Dict:
        try:
            stock, exp_dates = yf.Ticker(ticker), yf.Ticker(ticker).options
            if not exp_dates: return {'pcr': None, 'status': 'No options'}
            chain = stock.option_chain(exp_dates[0])
            call_oi, put_oi = chain.calls['openInterest'].sum(), chain.puts['openInterest'].sum()
            pcr_oi = put_oi / call_oi if call_oi > 0 else 0
            sentiment, score = ("🚀 極度恐慌 (看漲反轉信號)", 80) if pcr_oi > 1.5 else ("📈 高避險 (偏看漲)", 65) if pcr_oi > 1.2 else ("😐 中性", 50) if pcr_oi > 0.9 else ("📉 偏樂觀 (小心)", 35) if pcr_oi > 0.6 else ("⚠️ 極度貪婪 (看跌警告)", 20)
            return {'pcr_oi': round(pcr_oi, 2), 'sentiment': sentiment, 'status': 'OK'}
        except Exception as e: return {'pcr': None, 'status': f'Error: {str(e)}'}

# ============================================
# 🎯 VCP SCREENER (v10 Minervini 趨勢模板)
# ============================================
@dataclass
class VCPCandidate:
    ticker: str; price: float; bb_width: float; contractions: List[float]; volume_healthy: bool
    score: float; pivot_price: float; stop_loss: float; notes: List[str]

class VCPScreener:
    def __init__(self): self.ta = TechnicalAnalysis()
    def screen(self, df: pd.DataFrame, ticker: str) -> Optional[VCPCandidate]:
        if df is None or len(df) < 200: return None
        try:
            close, high, low = df['Close'], df['High'], df['Low']
            curr_price = float(close.iloc[-1])
            
            # --- Minervini 嚴格趨勢模板 ---
            sma50, sma150, sma200 = close.rolling(50).mean().iloc[-1], close.rolling(150).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
            low_52w, high_52w = low.tail(252).min(), high.tail(252).max()
            
            cond1 = curr_price > sma150 and curr_price > sma200 # 現價在長線之上
            cond2 = sma150 > sma200 # 均線多頭
            cond3 = curr_price > sma50 # 現價在短線之上
            cond4 = curr_price >= low_52w * 1.30 # 距底部起漲 30%
            cond5 = curr_price >= high_52w * 0.75 # 距 52週高點不超過 25% (高位強勢)
            
            if not (cond1 and cond2 and cond3 and cond4 and cond5): return None # 嚴格過濾死魚
            
            bb_width = float(self.ta.bollinger_band_width(close).iloc[-1])
            swing_data = self.ta.find_swing_points(df)
            contractions = swing_data.get('contractions', [])
            vol_sig = self.ta.analyze_volume_signature(df)
            pivot = float(df['High'].tail(20).max())
            
            score, notes = 40, ["✅ 通過 Minervini 嚴格趨勢模板 (Stage 2 Uptrend)"]
            
            if bb_width < 0.10: score += 30; notes.append(f"🎯 價格極度緊湊 (BB Width: {bb_width:.3f})")
            elif bb_width < 0.15: score += 15; notes.append(f"✅ 價格收斂中 (BB Width: {bb_width:.3f})")
            else: return None
            
            if len(contractions) >= 2 and contractions[-1] < contractions[0]: score += 15; notes.append("✅ 波浪呈現遞減收縮 (VCP 特徵)")
            if vol_sig['dry_up']: score += 15; notes.append("✅ 右側出現極度量縮 (Dry Up)，賣壓枯竭")
            
            atr = float(self.ta.atr(df).iloc[-1])
            return VCPCandidate(ticker, curr_price, bb_width, contractions, vol_sig['is_healthy'], score, pivot, max(low.tail(10).min(), pivot - atr), notes)
        except: return None
        
    def scan_batch(self, stocks: List[str]) -> List[VCPCandidate]:
        results, all_data = [], BatchDataFetcher.batch_download(stocks, period='1y')
        for ticker in stocks:
            df = all_data.get(ticker)
            if df is not None:
                cand = self.screen(df, ticker)
                if cand and cand.score >= 50: results.append(cand)
        return sorted(results, key=lambda x: x.score, reverse=True)

# ============================================
# 💰 SHORT PUT SCREENER (恐慌支撐反彈)
# ============================================
@dataclass
class ShortPutCandidate:
    ticker: str; price: float; above_sma200: bool; pullback_depth: float; rsi: float
    macd_reversal: bool; hv_rank: float; real_iv: Optional[float]
    nearest_support: str; nearest_support_price: float; distance_to_support: float
    score: float; quality: str; suggested_strike: float; notes: List[str]

class ShortPutScreener:
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.iv_calc = RealIVCalculator()
    
    def screen(self, df: pd.DataFrame, ticker: str, fetch_real_iv: bool = False) -> Optional[ShortPutCandidate]:
        if df is None or len(df) < 200: return None
        try:
            close = df['Close']
            curr_price = float(close.iloc[-1])
            
            support_data = self.ta.calculate_support_levels(df)
            sma200 = support_data.get('sma200_val', 0)
            above_sma200 = curr_price > sma200
            if not above_sma200: return None 

            high_60d = float(df['High'].tail(60).max())
            pullback_depth = (curr_price / high_60d - 1) * 100
            rsi = float(self.ta.rsi(close).iloc[-1])
            
            macd_df = self.ta.macd(close)
            hist_today, hist_ytd = macd_df['hist'].iloc[-1], macd_df['hist'].iloc[-2]
            macd_reversal = (hist_today > hist_ytd) and (hist_today < 0) 
            
            hv_rank = self.ta.estimate_hv_rank(df)
            real_iv = None
            if fetch_real_iv:
                iv_data = self.iv_calc.get_real_iv(ticker)
                if iv_data.get('iv'): real_iv = iv_data['iv']

            dist_to_support = support_data.get('distance_pct', 99)
            support_name = support_data.get('nearest_support', 'N/A')
            support_price = support_data.get('nearest_support_price', 0)

            score, notes = 0, []
            
            if dist_to_support <= 2.0: score += 30; notes.append(f"🎯 完美踩中支撐 ({support_name})，距離僅 {dist_to_support:.1f}%")
            elif dist_to_support <= 4.0: score += 20; notes.append(f"✅ 接近支撐區 ({support_name})，距離 {dist_to_support:.1f}%")
            else: score -= 10; notes.append(f"❌ 懸在半空，距離支撐 {dist_to_support:.1f}% (風險高)")

            if -15 <= pullback_depth <= -5: score += 20; notes.append(f"✅ 健康回調區間 ({pullback_depth:.1f}%)，散戶恐慌、IV上升")
            elif pullback_depth > -5: score += 5; notes.append(f"⚠️ 離高點太近 ({pullback_depth:.1f}%)，期權肉不多")
            else: score -= 10; notes.append(f"❌ 跌幅過深 ({pullback_depth:.1f}%)，趨勢可能已破壞")

            if 30 <= rsi <= 45: score += 20; notes.append(f"✅ RSI {rsi:.0f} - 進入超賣區，Premium 最肥")
            elif rsi < 30: score += 10; notes.append(f"⚠️ RSI {rsi:.0f} - 極度超賣 (注意防範無底洞)")
            elif 45 < rsi <= 60: score += 10; notes.append(f"😐 RSI {rsi:.0f} - 情緒中性，適合保守收租")
            else: score -= 10; notes.append(f"❌ RSI {rsi:.0f} - 動能偏上，隨時可能見頂回調")

            if macd_reversal: score += 20; notes.append("✅ MACD 綠柱縮短 - 下跌動能衰竭，適合進場賣 Put")
            else: score -= 10; notes.append("⚠️ 下跌動能仍在釋放中，可能接飛刀")
                
            if (real_iv and real_iv > 40) or (not real_iv and hv_rank > 40):
                score += 10; notes.append("✅ 隱含波動率足夠，權利金豐厚")

            quality = 'A+' if score >= 80 else 'A' if score >= 60 else 'B' if score >= 40 else 'C'
            suggested_strike = round(support_price * 0.98, 2) if support_price > 0 else round(curr_price * 0.9, 2)
            
            if score >= 40: 
                return ShortPutCandidate(
                    ticker=ticker, price=curr_price, above_sma200=True, pullback_depth=pullback_depth,
                    rsi=rsi, macd_reversal=macd_reversal, hv_rank=hv_rank, real_iv=real_iv,
                    nearest_support=support_name, nearest_support_price=support_price, distance_to_support=dist_to_support,
                    score=score, quality=quality, suggested_strike=suggested_strike, notes=notes
                )
            return None
        except: return None
    
    def scan_batch(self, stocks: List[str], progress_callback=None) -> List[ShortPutCandidate]:
        results = []
        all_data = BatchDataFetcher.batch_download(stocks, period='1y')
        for i, ticker in enumerate(stocks):
            if progress_callback: progress_callback(i, len(stocks), ticker)
            df = all_data.get(ticker)
            if df is not None:
                cand = self.screen(df, ticker, fetch_real_iv=False)
                if cand: results.append(cand)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

# ============================================
# 🦊 QULLAMAGGIE STRATEGY & BACKTEST ENGINE
# ============================================
class QullamaggieStrategy:
    def __init__(self):
        self.ta = TechnicalAnalysis()

    def screen(self, df: pd.DataFrame, ticker: str) -> Dict:
        """v10 AI Paper Trade 實時監測引擎"""
        if df is None or len(df) < 60: return None
        try:
            close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
            curr_price, curr_high, curr_vol = float(close.iloc[-1]), float(high.iloc[-1]), float(vol.iloc[-1])
            
            ema10, ema20, sma50 = self.ta.ema(close, 10).iloc[-1], self.ta.ema(close, 20).iloc[-1], close.rolling(50).mean().iloc[-1]
            ma_stacked = (curr_price > ema10 > ema20 > sma50)
            adr = float(self.ta.adr_percent(df, 20).iloc[-1])
            ret_3m = (curr_price / float(close.iloc[-63]) - 1) * 100 if len(close) > 63 else 0
            
            pivot_price = float(high.shift(1).tail(20).max())
            avg_vol_50 = float(vol.shift(1).tail(50).mean())
            
            is_setup = ma_stacked and adr >= 3.5 and ret_3m >= 25
            triggered_today = False
            if is_setup and (curr_high > pivot_price) and (curr_vol > avg_vol_50 * 1.2):
                triggered_today = True
                
            ai_note = f"🤖 **AI 診斷:** {ticker} 處於完美多頭 (EMA10>20>50)，高波動妖股特質 (ADR {adr:.1f}%)。近3月強勢上漲 {ret_3m:.0f}%。"
            if triggered_today: ai_note += f"\n🔥 **今日異動:** 盤中已帶量突破 Pivot (${pivot_price:.2f})！AI 建議立即市價追入，並將止損設於 EMA20 (${ema20:.2f})。"
            
            return {
                'ticker': ticker, 'price': curr_price, 'adr': adr, 'ret_3m': ret_3m,
                'pivot': pivot_price, 'stop_loss': ema20, 'is_setup': is_setup, 
                'triggered_today': triggered_today, 'ai_note': ai_note
            }
        except: return None

    def backtest_1yr(self, df: pd.DataFrame) -> Dict:
        """機構級回測引擎"""
        if len(df) < 252 + 50: return {'trades': 0}
        
        test_df = df.copy()
        test_df['EMA10'] = self.ta.ema(test_df['Close'], 10)
        test_df['EMA20'] = self.ta.ema(test_df['Close'], 20)
        test_df['SMA50'] = test_df['Close'].rolling(50).mean()
        test_df['ADR'] = self.ta.adr_percent(test_df, 20)
        test_df['High_20'] = test_df['High'].shift(1).rolling(20).max()
        test_df['AvgVol_50'] = test_df['Volume'].shift(1).rolling(50).mean()
        
        test_df = test_df.tail(252).copy()
        
        in_position, entry_price, initial_stop, entry_date = False, 0, 0, None
        trades, buy_signals, sell_signals = [], [], []
        
        for row in test_df.itertuples():
            date = row.Index
            if not in_position:
                vol_surge = False
                if not pd.isna(row.AvgVol_50) and row.AvgVol_50 > 0:
                    vol_surge = row.Volume > (row.AvgVol_50 * 1.5)
                
                if (row.High > row.High_20 and row.EMA10 > row.EMA20 > row.SMA50 and row.ADR > 3.0 and vol_surge):
                    in_position = True
                    entry_price = max(row.High_20, row.Open) 
                    initial_stop = max(row.Low, entry_price * 0.95) 
                    entry_date = date
                    buy_signals.append((date, entry_price))
            else:
                current_stop = max(initial_stop, row.EMA20)
                if row.Low < current_stop:
                    in_position = False
                    exit_price = min(current_stop, row.Open) 
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    trades.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': date, 'exit_price': exit_price, 'pnl_pct': pnl_pct})
                    sell_signals.append((date, exit_price))
                    
        if in_position:
            pnl_pct = (test_df['Close'].iloc[-1] / entry_price - 1) * 100
            trades.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': test_df.index[-1], 'exit_price': test_df['Close'].iloc[-1], 'pnl_pct': pnl_pct, 'open': True})

        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss) if trades else 0
        total_pnl = sum([t['pnl_pct'] for t in trades]) if trades else 0
        
        return {
            'trades': len(trades), 'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'expectancy': expectancy, 'total_pnl': total_pnl, 'history': trades, 
            'buy_marks': buy_signals, 'sell_marks': sell_signals, 'test_df': test_df
        }

# ============================================
# 🌡️ MARKET BREADTH & REGIME
# ============================================
class MarketRegime:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_health() -> Dict:
        try:
            data = yf.download(['SPY', 'QQQ'], period='6mo', progress=False, group_by='ticker', timeout=10)
            spy, qqq = data['SPY']['Close'], data['QQQ']['Close']
            
            spy_curr, spy_50, spy_200 = float(spy.iloc[-1]), float(spy.rolling(50).mean().iloc[-1]), float(spy.rolling(200).mean().iloc[-1])
            qqq_curr, qqq_50 = float(qqq.iloc[-1]), float(qqq.rolling(50).mean().iloc[-1])
            
            score = 0
            if spy_curr > spy_200: score += 40
            if spy_curr > spy_50: score += 20
            if qqq_curr > qqq_50: score += 40
            
            status = "🟢 狂暴牛市 (Risk On)" if score >= 80 else "🟡 震盪結構 (Cautious)" if score >= 40 else "🔴 熊市/深調 (Risk Off)"
            return {'status': status, 'score': score, 'spy': spy_curr, 'qqq_trend': 'UP' if qqq_curr > qqq_50 else 'DOWN'}
        except: return {'status': '🟡 未知', 'score': 50, 'spy': 0, 'qqq_trend': '-'}

# ============================================
# 📊 CHART BUILDER
# ============================================
class ChartBuilder:
    @staticmethod
    def create_chart_with_annotations(df: pd.DataFrame, ticker: str, pivot: float = None, entry: float = None, stop: float = None, support: float = None) -> go.Figure:
        df = df.copy()
        df['SMA20'], df['SMA50'] = df['Close'].rolling(20).mean(), df['Close'].rolling(50).mean()
        df['BB_Upper'], df['BB_Lower'] = df['SMA20'] + df['Close'].rolling(20).std() * 2, df['SMA20'] - df['Close'].rolling(20).std() * 2
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#00CC96', decreasing_line_color='#EF553B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='blue', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB', line=dict(color='gray', width=1, dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
        
        if pivot: fig.add_hline(y=pivot, line_dash="dash", line_color="cyan", line_width=2, annotation_text=f"📍 Pivot ${pivot:.2f}", annotation_position="right", row=1, col=1)
        if entry: fig.add_hline(y=entry, line_dash="dash", line_color="green", line_width=2, annotation_text=f"🎯 Entry ${entry:.2f}", annotation_position="right", row=1, col=1)
        if stop: fig.add_hline(y=stop, line_dash="dash", line_color="red", line_width=2, annotation_text=f"🛑 Stop ${stop:.2f}", annotation_position="right", row=1, col=1)
        if support: fig.add_hline(y=support, line_dash="dot", line_color="yellow", line_width=1, annotation_text=f"📊 Support ${support:.2f}", annotation_position="right", row=1, col=1)
        
        colors = ['#00CC96' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF553B' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(50).mean(), name='Avg Vol', line=dict(color='yellow', width=1, dash='dash')), row=2, col=1)
        
        fig.update_layout(height=600, template='plotly_dark', title=f"{ticker} - 技術分析", xaxis_rangeslider_visible=False)
        return fig

    @staticmethod
    def create_qullamaggie_chart(df: pd.DataFrame, ticker: str, buy_marks=[], sell_marks=[]) -> go.Figure:
        df = df.copy()
        df['EMA10'] = TechnicalAnalysis.ema(df['Close'], 10)
        df['EMA20'] = TechnicalAnalysis.ema(df['Close'], 20)
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#00CC96', decreasing_line_color='#EF553B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA10'], name='EMA 10', line=dict(color='yellow', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA 20', line=dict(color='red', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', line=dict(color='blue', width=2, dash='dot')), row=1, col=1)
        
        for date, price in buy_marks:
            fig.add_annotation(x=date, y=price*0.92, text="⬆️ BUY", showarrow=True, arrowhead=1, arrowcolor="green", font=dict(color="green"), row=1, col=1)
        for date, price in sell_marks:
            fig.add_annotation(x=date, y=price*1.08, text="⬇️ SELL", showarrow=True, arrowhead=1, arrowcolor="red", font=dict(color="red"), row=1, col=1)

        colors = ['#00CC96' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF553B' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume'].rolling(50).mean(), name='Avg Vol', line=dict(color='yellow', width=1, dash='dash')), row=2, col=1)
        
        fig.update_layout(height=650, template='plotly_dark', title=f"{ticker} - Qullamaggie 策略回測圖", xaxis_rangeslider_visible=False)
        return fig

# ============================================
# 📱 MAIN UI (Sidebar + Core Logic Integration)
# ============================================
def main():
    # --- Sidebar ---
    st.sidebar.title(f"{CONFIG.PAGE_ICON} Pro Terminal")
    st.sidebar.markdown("### 📊 Market Regime")
    market = MarketRegime.get_health()
    st.sidebar.metric("大盤狀態", market['status'])
    st.sidebar.metric("SPY 指數", f"${market['spy']:.2f}")
    st.sidebar.metric("QQQ 動能", market['qqq_trend'])
    st.sidebar.progress(market['score'] / 100)
    
    st.sidebar.divider()
    page = st.sidebar.radio("導航選單 (Navigation)", 
                            ["🦊 Qullamaggie 實盤與掃描", "🎯 VCP 嚴格趨勢選股", "💰 Short Put 恐慌收租", "📈 個股深度圖表"])
    
    st.sidebar.divider()
    st.sidebar.markdown("v10.0 | AI 驅動量化系統")

    # --- Main Content ---
    st.header(page)
    
    if page == "🦊 Qullamaggie 實盤與掃描":
        tabs = st.tabs(["🤖 AI 實盤交易版 (Paper Trade)", "🔍 今日潛力 Setup", "⏪ 歷史策略回測"])
        q_strategy = QullamaggieStrategy()
        
        with tabs[0]:
            st.subheader("🤖 AI 自動捕捉今日突破 (Live Tracking)")
            st.write("系統掃描美股，若今日盤中股價**帶量衝破 Pivot**，AI 將自動建倉並給出操作指令。")
            
            if st.button("🔄 掃描今日突破訊號", type="primary"):
                stocks = STOCK_UNIVERSE['High Growth'] + STOCK_UNIVERSE['Semiconductors']
                with st.spinner("AI 正在監控市場盤口..."):
                    all_data = BatchDataFetcher.batch_download(stocks, period='6mo')
                    new_trades = 0
                    for ticker in stocks:
                        df = all_data.get(ticker)
                        if df is not None:
                            res = q_strategy.screen(df, ticker)
                            if res and res['triggered_today'] and ticker not in st.session_state.paper_trades:
                                st.session_state.paper_trades[ticker] = res
                                new_trades += 1
                    if new_trades > 0: st.success(f"🔥 捕捉到 {new_trades} 個今日爆發突破！")
                    else: st.info("目前無新增突破訊號。")
            
            if st.session_state.paper_trades:
                st.markdown("### 💼 你的 AI 實盤持倉 (Active Paper Trades)")
                for ticker, trade in st.session_state.paper_trades.items():
                    with st.container(border=True):
                        col1, col2, col3 = st.columns([1, 1, 2])
                        col1.metric("Ticker", ticker)
                        col2.metric("觸發價 (Entry)", f"${trade['pivot']:.2f}")
                        col3.error(f"🛑 動態止損位 (EMA20): ${trade['stop_loss']:.2f}")
                        st.markdown(trade['ai_note'])
            else:
                st.caption("暫無持倉。點擊上方按鈕掃描今日訊號。")
                
        with tabs[1]:
            st.subheader("盤前準備：蓄勢待發的妖股")
            if st.button("掃描 Setup"):
                stocks = STOCK_UNIVERSE['High Growth'] + STOCK_UNIVERSE['Semiconductors']
                with st.spinner("掃描中..."):
                    all_data = BatchDataFetcher.batch_download(stocks, period='6mo')
                    for ticker in stocks:
                        res = q_strategy.screen(all_data.get(ticker), ticker)
                        if res and res['is_setup'] and not res['triggered_today']:
                            with st.expander(f"⏳ **{ticker}** | 準備突破 Pivot: ${res['pivot']:.2f}"):
                                st.write(f"- ADR: {res['adr']:.1f}% | 3月漲幅: {res['ret_3m']:.0f}%")
                                st.info(f"**交易計畫:** 掛 Buy Stop 單於 ${res['pivot']:.2f}。跌破 ${res['stop_loss']:.2f} 止損。")

        with tabs[2]:
            st.subheader("驗證策略有效性 (1-Year Backtest & Scanner)")
            bt_ticker = st.text_input("輸入要回測的股票代碼 (例如: PLTR)", value="PLTR").upper()
            if st.button("▶️ 執行單股回測"):
                with st.spinner(f"正在以機構級精度計算 {bt_ticker}..."):
                    df = BatchDataFetcher.get_single_stock(bt_ticker, "2y")
                    if df is not None:
                        bt_result = q_strategy.backtest_1yr(df)
                        if bt_result['trades'] > 0:
                            st.success(f"回測完成！共觸發 {bt_result['trades']} 次突破。")
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("交易次數", bt_result['trades'])
                            m2.metric("勝率", f"{bt_result['win_rate']:.1f}%")
                            m3.metric("平均獲利", f"+{bt_result['avg_win']:.1f}%")
                            m4.metric("平均虧損", f"{bt_result['avg_loss']:.1f}%")
                            m5.metric("期望值", f"{bt_result['expectancy']:+.2f}%")
                            fig = ChartBuilder.create_qullamaggie_chart(bt_result['test_df'], bt_ticker, bt_result['buy_marks'], bt_result['sell_marks'])
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.warning(f"{bt_ticker} 無觸發信號。")
            
            st.divider()
            st.subheader("🏆 歷史妖股批量掃描")
            scan_group = st.selectbox("選擇掃描板塊", ["高成長股 (High Growth)", "半導體 (Semiconductors)"])
            if st.button("🚀 啟動歷史掃描"):
                target_stocks = STOCK_UNIVERSE['High Growth'] if "高成長" in scan_group else STOCK_UNIVERSE['Semiconductors']
                with st.spinner("批量回測中..."):
                    all_data = BatchDataFetcher.batch_download(target_stocks, period="2y")
                    leaderboard = []
                    for ticker in target_stocks:
                        df = all_data.get(ticker)
                        if df is not None:
                            res = q_strategy.backtest_1yr(df)
                            if res['trades'] > 0:
                                leaderboard.append({"Ticker": ticker, "交易次數": res['trades'], "勝率(%)": round(res['win_rate'], 1), "期望值(%)": round(res['expectancy'], 2), "總利潤(%)": round(res['total_pnl'], 2)})
                    if leaderboard:
                        lb_df = pd.DataFrame(leaderboard).sort_values(by="期望值(%)", ascending=False).reset_index(drop=True)
                        st.dataframe(lb_df.style.background_gradient(subset=['總利潤(%)', '期望值(%)'], cmap='Greens'), use_container_width=True)

    elif page == "🎯 VCP 嚴格趨勢選股":
        st.info("⚠️ **v10 升級:** 引入 Minervini 趨勢模板，過濾下跌趨勢中的假橫盤，只選真正的 Stage 2 強勢股。")
        if st.button("啟動 VCP 掃描", type="primary"):
            screener = VCPScreener()
            with st.spinner("掃描全市場 VCP 形態..."):
                results = screener.scan_batch(ALL_STOCKS)
                if results:
                    st.success(f"找到 {len(results)} 隻純正 VCP 股票")
                    for r in results[:10]:
                        with st.expander(f"🎯 **{r.ticker}** | 評分: {r.score} | 現價: ${r.price:.2f}"):
                            st.write(f"**Pivot 突破位:** ${r.pivot_price:.2f} | **安全止損:** ${r.stop_loss:.2f}")
                            st.write(f"收縮波段: {[f'{c:.1f}%' for c in r.contractions]}")
                            for note in r.notes: st.write(note)
                            if st.button("查看圖表", key=f"vcp_{r.ticker}"):
                                df = BatchDataFetcher.get_single_stock(r.ticker, "6mo")
                                fig = ChartBuilder.create_chart_with_annotations(df, r.ticker, pivot=r.pivot_price, stop=r.stop_loss)
                                st.plotly_chart(fig, use_container_width=True)
                else: st.warning("目前市場沒有符合嚴格 VCP 模板的股票。")

    elif page == "💰 Short Put 恐慌收租":
        st.info("尋找強勢股的回調錯殺點，精準賣在支撐位與動能衰竭處。")
        if st.button("尋找收租機會", type="primary"):
            screener, pb, st_txt = ShortPutScreener(), st.progress(0), st.empty()
            stocks = STOCK_UNIVERSE['Market Leaders'] + STOCK_UNIVERSE['Blue Chips (Put)']
            def upd(i, t, tic): pb.progress(min((i + 1) / t, 1.0)); st_txt.text(f"掃描 {tic}...")
            with st.spinner("掃描藍籌與熱門股..."):
                results = screener.scan_batch(stocks, upd)
            pb.empty(); st_txt.empty()
            
            if results:
                st.success(f"找到 {len(results)} 個安全收租機會")
                for res in results[:10]:
                    with st.container(border=True):
                        st.markdown(f"### 🛡️ {res.ticker} (現價 ${res.price:.2f})")
                        col1, col2 = st.columns(2)
                        col1.write(f"- 回調幅度: {res.pullback_depth:.1f}%\n- 踩中支撐: {res.nearest_support} (距 {res.distance_to_support:.1f}%)")
                        col2.success(f"**建議 Sell Put Strike:** ${res.suggested_strike:.2f}")
                        with st.expander("查看詳細邏輯"):
                            for n in res.notes: st.write(n)
                            
    elif page == "📈 個股深度圖表":
        ticker = st.text_input("輸入代碼 (如 NVDA)", value="NVDA").upper()
        if st.button("繪製專業圖表"):
            df = BatchDataFetcher.get_single_stock(ticker, "1y")
            if df is not None:
                # 繪製帶有 Qullamaggie 均線系統的專業圖表
                df['EMA10'] = TechnicalAnalysis.ema(df['Close'], 10)
                df['EMA20'] = TechnicalAnalysis.ema(df['Close'], 20)
                df['SMA50'] = df['Close'].rolling(50).mean()
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA10'], line=dict(color='yellow', width=1), name='EMA10'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='red', width=1.5), name='EMA20 (Stop)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='blue', dash='dot'), name='SMA50'), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='gray', name='Volume'), row=2, col=1)
                
                fig.update_layout(template='plotly_dark', height=700, xaxis_rangeslider_visible=False, title=f"{ticker} 專業分析圖 (EMA10/20/50)")
                st.plotly_chart(fig, use_container_width=True)
                
                # 附加期權數據
                pcr_calc, iv_calc = PCRCalculator(), RealIVCalculator()
                st.subheader("📊 期權數據")
                c1, c2 = st.columns(2)
                with c1:
                    pcr_data = pcr_calc.get_pcr(ticker)
                    if pcr_data.get('status') == 'OK':
                        st.metric("PCR (OI)", f"{pcr_data['pcr_oi']:.2f}")
                        st.write(pcr_data['sentiment'])
                with c2:
                    iv_data = iv_calc.get_real_iv(ticker)
                    if iv_data.get('iv'):
                        st.metric("真實 IV", f"{iv_data['iv']:.1f}%")

if __name__ == "__main__":
    main()

