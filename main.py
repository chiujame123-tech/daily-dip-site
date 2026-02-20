# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v9.5 Ultimate Edition
=================================================

核心亮點:
1. 🦊 Qullamaggie 機構級回測 (修復執行價滑點、嚴格初始止損、批量期望值掃描)
2. 💰 Short Put 恐慌支撐反彈邏輯 (MACD動能剎車、精準支撐位、拒絕死魚股)
3. 🎯 VCP 波動收縮形態 (Swing Points, 量能特徵分析)
4. 🚀 批量異步下載引擎 (大幅提升掃描速度)

Author: Pro Trader AI
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================
# ⚙️ CONFIGURATION & UNIVERSE
# ============================================
@dataclass
class Config:
    PAGE_TITLE: str = "Market Radar v9.5 Ultimate"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800

CONFIG = Config()

STOCK_UNIVERSE = {
    'Market Leaders': [
        'NVDA', 'META', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'LLY', 'AVGO', 'TSLA', 'AMD',
        'CRM', 'NOW', 'PANW', 'CRWD', 'NFLX', 'COST', 'ISRG', 'LULU', 'CMG', 'FICO'
    ],
    'Semiconductors': [
        'NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM', 'MU', 'AMAT', 'LRCX', 'KLAC',
        'MRVL', 'ARM', 'SMCI', 'INTC', 'ASML', 'SNPS', 'ON', 'NXPI', 'ADI', 'MCHP', 'TXN'
    ],
    'Software & Cloud': [
        'MSFT', 'CRM', 'ADBE', 'NOW', 'INTU', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'NET',
        'MDB', 'PLTR', 'ZS', 'FTNT', 'WDAY', 'TEAM', 'HUBS', 'OKTA', 'BILL', 'DOCU'
    ],
    'High Growth': [
        'NVDA', 'SMCI', 'ARM', 'PLTR', 'COIN', 'MSTR', 'AFRM', 'SOFI', 'HOOD', 'UPST',
        'RBLX', 'DKNG', 'SHOP', 'SQ', 'MELI', 'SE', 'NU', 'GRAB', 'BILL', 'CELH'
    ],
    'Blue Chips (Short Put)': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'V', 'MA', 'JNJ', 'PG',
        'KO', 'PEP', 'WMT', 'COST', 'HD', 'MCD', 'DIS', 'NFLX', 'ADBE', 'CRM',
        'UNH', 'LLY', 'MRK', 'ABBV', 'TMO', 'ACN', 'CSCO', 'ORCL', 'IBM', 'INTC'
    ],
    'Dividend Stocks': [
        'JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'HD', 'VZ', 'T', 'XOM',
        'CVX', 'IBM', 'CSCO', 'INTC', 'MRK', 'ABBV', 'PFE', 'BMY', 'MMM', 'CAT'
    ],
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
    def bollinger_band_width(prices: pd.Series, period: int = 20) -> pd.Series:
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return ((sma + std * 2) - (sma - std * 2)) / sma
        
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return sma + std * 2, sma - std * 2
        
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
# 🦊 QULLAMAGGIE SCREENER & BACKTESTER
# ============================================
@dataclass
class QullamaggieCandidate:
    ticker: str
    price: float
    adr: float
    momentum_3m: float
    momentum_1m: float
    above_ma_stack: bool 
    consolidation_tightness: float 
    vol_dry_up: bool
    score: float
    notes: List[str]
    pivot_price: float
    stop_loss: float

class QullamaggieStrategy:
    def __init__(self):
        self.ta = TechnicalAnalysis()

    def screen(self, df: pd.DataFrame, ticker: str) -> Optional[QullamaggieCandidate]:
        if df is None or len(df) < 100: return None
        try:
            close = df['Close']
            curr_price = float(close.iloc[-1])
            
            ema10 = float(self.ta.ema(close, 10).iloc[-1])
            ema20 = float(self.ta.ema(close, 20).iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            ma_stacked = (curr_price > ema10) and (ema10 > ema20) and (ema20 > sma50)
            
            ret_1m = (curr_price / float(close.iloc[-21]) - 1) * 100
            ret_3m = (curr_price / float(close.iloc[-63]) - 1) * 100
            
            adr_series = self.ta.adr_percent(df, 20)
            adr = float(adr_series.iloc[-1]) if not pd.isna(adr_series.iloc[-1]) else 0
            
            recent_20 = df.tail(20)
            pivot_price = float(recent_20['High'].max())
            
            high_10 = float(df['High'].tail(10).max())
            low_10 = float(df['Low'].tail(10).min())
            tightness = (high_10 / low_10 - 1) * 100
            
            suggested_stop = max(low_10, pivot_price * 0.95)
            
            vol_50_avg = float(df['Volume'].tail(50).mean())
            vol_last_3_avg = float(df['Volume'].tail(3).mean())
            vol_dry_up = vol_last_3_avg < vol_50_avg * 0.7
            
            score, notes = 0, []
            if ma_stacked: score += 30; notes.append("✅ 完美均線排列 (P > EMA10 > EMA20 > SMA50)")
            else: score -= 20; notes.append("❌ 均線未呈現強多頭排列")
                
            if adr >= 4.0: score += 20; notes.append(f"✅ ADR {adr:.1f}% (> 4% 高波動妖股特質)")
            elif adr >= 3.0: score += 10; notes.append(f"⚠️ ADR {adr:.1f}% (波動中等)")
            else: notes.append(f"❌ ADR {adr:.1f}% (波動太小)")
                
            if ret_3m >= 30 or ret_1m >= 20: score += 20; notes.append(f"✅ 動能強勁 (1M: {ret_1m:.1f}%, 3M: {ret_3m:.1f}%)")
            else: notes.append(f"❌ 動能不足")
                
            if tightness < 10.0: score += 20; notes.append(f"✅ 過去10天收縮極致 (緊縮度 {tightness:.1f}%)")
            elif tightness < 15.0: score += 10; notes.append(f"⚠️ 區間盤整中 (緊縮度 {tightness:.1f}%)")
                
            if vol_dry_up: score += 10; notes.append("✅ 突破前量縮 (Dry Up)")

            if score >= 40:
                return QullamaggieCandidate(
                    ticker=ticker, price=curr_price, adr=adr, momentum_3m=ret_3m, momentum_1m=ret_1m,
                    above_ma_stack=ma_stacked, consolidation_tightness=tightness, vol_dry_up=vol_dry_up,
                    score=score, notes=notes, pivot_price=pivot_price, stop_loss=suggested_stop
                )
            return None
        except: return None

    def backtest_1yr(self, df: pd.DataFrame) -> Dict:
        """機構級回測引擎 (修復滑點、增加初始止損、量能確認)"""
        if len(df) < 252 + 50: return {'trades': 0}
        
        test_df = df.copy()
        test_df['EMA10'] = self.ta.ema(test_df['Close'], 10)
        test_df['EMA20'] = self.ta.ema(test_df['Close'], 20)
        test_df['SMA50'] = test_df['Close'].rolling(50).mean()
        test_df['ADR'] = self.ta.adr_percent(test_df, 20)
        test_df['High_20'] = test_df['High'].shift(1).rolling(20).max()
        test_df['AvgVol_50'] = test_df['Volume'].shift(1).rolling(50).mean()
        
        test_df = test_df.tail(252).copy()
        
        in_position = False
        entry_price = 0
        initial_stop = 0
        entry_date = None
        trades = []
        buy_signals = []
        sell_signals = []
        
        for row in test_df.itertuples():
            date = row.Index
            
            if not in_position:
                vol_surge = False
                if not pd.isna(row.AvgVol_50) and row.AvgVol_50 > 0:
                    vol_surge = row.Volume > (row.AvgVol_50 * 1.5)
                
                if (row.High > row.High_20 and 
                    row.EMA10 > row.EMA20 > row.SMA50 and 
                    row.ADR > 3.0 and 
                    vol_surge):
                    
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
                    trades.append({
                        'entry_date': entry_date, 'entry_price': entry_price,
                        'exit_date': date, 'exit_price': exit_price, 'pnl_pct': pnl_pct
                    })
                    sell_signals.append((date, exit_price))
                    
        if in_position:
            pnl_pct = (test_df['Close'].iloc[-1] / entry_price - 1) * 100
            trades.append({
                'entry_date': entry_date, 'entry_price': entry_price,
                'exit_date': test_df.index[-1], 'exit_price': test_df['Close'].iloc[-1], 'pnl_pct': pnl_pct, 'open': True
            })

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
# 💰 SHORT PUT SCREENER (恐慌支撐反彈)
# ============================================
@dataclass
class ShortPutCandidate:
    ticker: str
    price: float
    above_sma200: bool
    pullback_depth: float
    rsi: float
    macd_reversal: bool
    hv_rank: float
    real_iv: Optional[float]
    nearest_support: str
    nearest_support_price: float
    distance_to_support: float
    score: float
    quality: str
    suggested_strike: float
    notes: List[str]

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
# 🎯 VCP SCREENER
# ============================================
@dataclass
class VCPCandidate:
    ticker: str
    price: float
    above_sma50: bool
    above_sma200: bool
    dist_from_52w_high: float
    bb_width: float
    swing_contractions: List[float]
    contraction_quality: str
    volume_signature: Dict
    dry_up: bool
    rsi: float
    rs_rating: float
    pivot_price: float
    score: float
    quality: str
    entry_price: float
    stop_loss: float
    notes: List[str]

class VCPScreener:
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def screen(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[VCPCandidate]:
        if df is None or len(df) < 100: return None
        try:
            close, high, low = df['Close'], df['High'], df['Low']
            curr_price = float(close.iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else sma50
            above_sma50, above_sma200 = curr_price > sma50, curr_price > sma200
            
            if not above_sma50: return None
            
            high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
            dist_from_high = (curr_price / high_52w - 1) * 100
            bb_width = float(self.ta.bollinger_band_width(close).iloc[-1])
            
            swing_data = self.ta.find_swing_points(df)
            swing_contractions = swing_data.get('contractions', [])
            contraction_quality = "❌ 無明顯收縮"
            if len(swing_contractions) >= 2:
                decreasing = all(swing_contractions[i] > swing_contractions[i+1] * 0.9 for i in range(len(swing_contractions)-1))
                if decreasing and swing_contractions[-1] < 10: contraction_quality = f"✅ 完美遞減收縮 ({len(swing_contractions)}波)"
                elif swing_contractions[-1] < 15: contraction_quality = f"⚠️ 有收縮 ({len(swing_contractions)}波)"
            
            vol_sig = self.ta.analyze_volume_signature(df)
            rsi = float(self.ta.rsi(close).iloc[-1])
            rs_rating = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            
            recent = df.tail(40)
            pivot = float(recent['High'].max())
            
            score, notes = 0, []
            if above_sma50 and above_sma200: score += 25; notes.append("✅ 在 SMA50 和 SMA200 之上")
            elif above_sma50: score += 15; notes.append("⚠️ 在 SMA50 之上")
            
            if bb_width < 0.10: score += 25; notes.append(f"✅ BB Width {bb_width:.3f} - 極度收窄")
            elif bb_width < 0.15: score += 20; notes.append(f"✅ BB Width {bb_width:.3f} - 收窄")
            elif bb_width < 0.20: score += 10; notes.append(f"⚠️ BB Width {bb_width:.3f}")
            
            if len(swing_contractions) >= 3 and swing_contractions[-1] < 8: score += 15; notes.append(f"✅ {len(swing_contractions)} 波遞減收縮")
            elif len(swing_contractions) >= 2: score += 10; notes.append(f"⚠️ {len(swing_contractions)} 波收縮")
            
            if dist_from_high >= -5: score += 15; notes.append(f"✅ 距52週高點 {dist_from_high:.1f}%")
            elif dist_from_high >= -15: score += 10; notes.append(f"⚠️ 距52週高點 {dist_from_high:.1f}%")
            
            if 45 <= rsi <= 65: score += 10; notes.append(f"✅ RSI {rsi:.0f} - 橫盤區間")
            
            if vol_sig['is_healthy']: score += 10; notes.append(f"✅ 上漲放量/下跌縮量 (比率 {vol_sig['ratio']:.2f})")
            if vol_sig['dry_up']: score += 10; notes.append(f"✅ 量能萎縮 Dry Up")
            if rs_rating >= 80: score += 10; notes.append(f"✅ RS {rs_rating:.0f}")
            
            quality = 'A+' if score >= 80 else 'A' if score >= 65 else 'B' if score >= 50 else 'C'
            atr = float(self.ta.atr(df).iloc[-1])
            entry = pivot * 1.001
            stop = max(float(recent['Low'].min()), pivot * 0.95) - atr * 0.2
            
            return VCPCandidate(
                ticker=ticker, price=curr_price, above_sma50=above_sma50, above_sma200=above_sma200,
                dist_from_52w_high=dist_from_high, bb_width=bb_width, swing_contractions=swing_contractions,
                contraction_quality=contraction_quality, volume_signature=vol_sig, dry_up=vol_sig['dry_up'],
                rsi=rsi, rs_rating=rs_rating, pivot_price=pivot, score=score, quality=quality, 
                entry_price=round(entry, 2), stop_loss=round(stop, 2), notes=notes
            )
        except: return None
    
    def scan_batch(self, stocks: List[str], spy_df: pd.DataFrame, progress_callback=None) -> List[VCPCandidate]:
        results = []
        all_data = BatchDataFetcher.batch_download(stocks, period='1y')
        for i, ticker in enumerate(stocks):
            if progress_callback: progress_callback(i, len(stocks), ticker)
            df = all_data.get(ticker)
            if df is not None and len(df) >= 100:
                candidate = self.screen(df, ticker, spy_df)
                if candidate and candidate.score >= 40 and candidate.above_sma50: results.append(candidate)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

# ============================================
# 🌡️ MARKET REGIME
# ============================================
class MarketRegime:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_health() -> Dict:
        default = {'status': '🟡 謹慎', 'score': 60, 'vix': 18.0, 'spy_price': 500.0, 'advice': '正常交易'}
        try:
            spy = yf.download('SPY', period='6mo', progress=False, timeout=15)
            if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
            if spy is None or len(spy) == 0: return default
            spy_close = float(spy['Close'].iloc[-1])
            sma50, sma200 = float(spy['Close'].rolling(50).mean().iloc[-1]), float(spy['Close'].rolling(200).mean().iloc[-1]) if len(spy) >= 200 else 0
            score = 50 + (15 if spy_close > sma200 else 0) + (10 if spy_close > sma50 else 0)
            status, advice = ("🟢 強勢", "全力進攻") if score >= 70 else ("🟡 謹慎", "正常交易") if score >= 50 else ("🔴 弱勢", "防守")
            return {'status': status, 'score': score, 'advice': advice, 'vix': 18.0, 'spy_price': round(spy_close, 2)}
        except: return default

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
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v9.5 Ultimate")
    
    market = MarketRegime.get_health()
    cols = st.columns(4)
    cols[0].metric("市場狀態", market['status'])
    cols[1].metric("健康評分", f"{market['score']}/100")
    cols[2].metric("SPY", f"${market.get('spy_price', 500.0):.2f}")
    cols[3].metric("建議", market['advice'])
    
    st.divider()
    
    tabs = st.tabs(["🦊 Qullamaggie 動能突破", "💰 Short Put 收租", "🎯 VCP 橫盤爆發", "📊 個股分析"])
    
    # ===== TAB 1: Qullamaggie =====
    with tabs[0]:
        st.header("🦊 Kristjan Kullamägi (EP/HTF) 動能突破")
        st.info("""
        **Qullamaggie 核心心法：**
        - **趨勢:** 價格必須在 EMA10 > EMA20 > SMA50 之上。
        - **動能:** 過去 1~3 個月內漲幅巨大 (>30%)，波動 ADR(20) > 4%。
        - **進場:** 盤中突破 Pivot 且伴隨**成交量激增 (>1.5倍)**，**切勿等待收盤！**
        - **止損:** 初始止損設在突破日低點，之後用 EMA20 追蹤止損 (Trailing Stop)。
        """)
        
        sub_tabs = st.tabs(["🔍 今日突破掃描 (Screener)", "⏪ 單股策略回測 (Single)", "🏆 歷史妖股排行榜 (Batch Scan)"])
        q_strategy = QullamaggieStrategy()
        
        with sub_tabs[0]:
            st.subheader("尋找即將爆發的妖股")
            if st.button("掃描熱門股 Setup", type="primary"):
                stocks = STOCK_UNIVERSE['High Growth'] + STOCK_UNIVERSE['Semiconductors']
                with st.spinner("掃描中..."):
                    all_data = BatchDataFetcher.batch_download(stocks, period='6mo')
                    results = []
                    for ticker in stocks:
                        df = all_data.get(ticker)
                        if df is not None:
                            cand = q_strategy.screen(df, ticker)
                            if cand: results.append(cand)
                    results.sort(key=lambda x: x.score, reverse=True)
                    
                    st.success(f"找到 {len(results)} 隻符合 Qullamaggie 盤整型態的股票")
                    for r in results:
                        with st.expander(f"🚀 **{r.ticker}** | ADR: {r.adr:.1f}% | 評分: {r.score}"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(f"- **現價:** ${r.price:.2f}")
                                st.write(f"- **3個月漲幅:** {r.momentum_3m:.1f}%")
                                st.write(f"- **近期緊縮度:** {r.consolidation_tightness:.1f}%")
                            with c2:
                                for note in r.notes: st.write(note)
                            
                            st.divider()
                            st.markdown("### 📋 交易計劃 (Trade Plan)")
                            st.warning(f"**操作指令 (Buy Stop Limit):** 當盤中股價衝破 **${r.pivot_price:.2f}** 且成交量異常放大時，立即追入！")
                            c3, c4 = st.columns(2)
                            with c3: st.metric("🎯 突破買入點 (Pivot)", f"${r.pivot_price:.2f}")
                            with c4: st.metric("🛑 初始止損點 (Stop Loss)", f"${r.stop_loss:.2f}", help="跌破此價位立即停損")
                                
        with sub_tabs[1]:
            st.subheader("驗證策略有效性 (1-Year Backtest)")
            bt_ticker = st.text_input("輸入要回測的股票代碼 (例如: NVDA, SMCI, PLTR)", value="PLTR").upper()
            
            if st.button("▶️ 執行策略回測", key="run_bt"):
                with st.spinner(f"正在以機構級精度計算 {bt_ticker}..."):
                    df = BatchDataFetcher.get_single_stock(bt_ticker, "2y")
                    if df is not None:
                        bt_result = q_strategy.backtest_1yr(df)
                        if bt_result['trades'] > 0:
                            st.success(f"回測完成！過去一年共觸發 {bt_result['trades']} 次精準突破信號。")
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("交易次數", bt_result['trades'])
                            m2.metric("策略勝率", f"{bt_result['win_rate']:.1f}%")
                            m3.metric("平均獲利", f"+{bt_result['avg_win']:.1f}%")
                            m4.metric("平均虧損", f"{bt_result['avg_loss']:.1f}%")
                            m5.metric("每筆期望值", f"{bt_result['expectancy']:+.2f}%", help="數學期望值，正數代表長期賺錢")
                            
                            fig = ChartBuilder.create_qullamaggie_chart(bt_result['test_df'], bt_ticker, bt_result['buy_marks'], bt_result['sell_marks'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📜 交易記錄")
                            history_df = pd.DataFrame(bt_result['history'])
                            history_df['entry_date'] = history_df['entry_date'].dt.strftime('%Y-%m-%d')
                            history_df['exit_date'] = history_df['exit_date'].dt.strftime('%Y-%m-%d')
                            history_df['pnl_pct'] = history_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
                            st.dataframe(history_df, use_container_width=True)
                        else: st.warning(f"{bt_ticker} 過去一年沒有觸發信號。")
                    else: st.error("無法獲取數據。")

        with sub_tabs[2]:
            st.subheader("🏆 歷史妖股掃描 (Batch Backtest)")
            st.write("自動掃描板塊內所有股票，找出過去一年使用該策略賺最多錢的標的。")
            scan_group = st.selectbox("選擇掃描板塊", ["高成長股 (High Growth)", "半導體 (Semiconductors)", "軟件雲端 (Software)"])
            
            if st.button("🚀 啟動歷史掃描"):
                target_stocks = STOCK_UNIVERSE['High Growth'] if "高成長" in scan_group else STOCK_UNIVERSE['Semiconductors'] if "半導體" in scan_group else STOCK_UNIVERSE['Software & Cloud']
                with st.spinner(f"正在批量下載 {len(target_stocks)} 隻股票數據並進行回測..."):
                    all_data = BatchDataFetcher.batch_download(target_stocks, period="2y")
                    leaderboard = []
                    for ticker in target_stocks:
                        df = all_data.get(ticker)
                        if df is not None:
                            res = q_strategy.backtest_1yr(df)
                            if res['trades'] > 0:
                                leaderboard.append({
                                    "Ticker": ticker, "交易次數": res['trades'], "勝率 (%)": round(res['win_rate'], 1),
                                    "平均獲利 (%)": round(res['avg_win'], 1), "期望值 (%)": round(res['expectancy'], 2),
                                    "總利潤 (%)": round(res['total_pnl'], 2)
                                })
                    
                    if leaderboard:
                        lb_df = pd.DataFrame(leaderboard).sort_values(by="總利潤 (%)", ascending=False).reset_index(drop=True)
                        st.success("掃描完成！以下是過去一年的策略表現排行榜：")
                        st.dataframe(lb_df.style.background_gradient(subset=['總利潤 (%)', '期望值 (%)'], cmap='Greens'), use_container_width=True)
                        st.info("💡 **解讀:** 總利潤或期望值最高的股票，代表它的『股性』非常適合突破策略，建議加入重點監控名單。")
                    else: st.warning("該板塊內沒有股票觸發信號。")

    # ===== TAB 2: Short Put Screener =====
    with tabs[1]:
        st.header("💰 Short Put 收租選股器 (恐慌支撐反彈)")
        st.info("""
        **全新邏輯：不買死魚股，只買錯殺的強勢股！**
        - ✅ 長期牛市 (Price > SMA200)
        - ✅ 短期恐慌 (距高點回調 5%-15%，RSI < 45)
        - ✅ 精準踩點 (距離 SMA50/SMA100/布林下軌 < 4%)
        - ✅ 拒絕飛刀 (MACD 綠柱開始縮短，動能衰竭)
        """)
        
        c1, c2 = st.columns(2)
        with c1: sp_scope = st.selectbox("掃描範圍", ["🏦 藍籌股 (30隻)", "💵 高息股 (20隻)", "🔥 熱門股 (20隻)"], key="sp_scope")
        with c2: sp_quality = st.selectbox("最低質量", ["全部", "只看 A+ 和 A", "只看 A+"], key="sp_quality")
        
        if st.button("🔍 批量掃描收租機會", type="primary", key="scan_sp"):
            stocks = STOCK_UNIVERSE['Blue Chips (Short Put)'] if "藍籌" in sp_scope else STOCK_UNIVERSE['Dividend Stocks'] if "高息" in sp_scope else STOCK_UNIVERSE['Market Leaders']
            screener, pb, st_txt = ShortPutScreener(), st.progress(0), st.empty()
            
            def upd(i, t, tic): pb.progress(min((i + 1) / t, 1.0)); st_txt.text(f"掃描 {tic}...")
            with st.spinner("批量下載數據..."): results = screener.scan_batch(stocks, upd)
            pb.empty(); st_txt.empty()
            
            if "只看 A+" in sp_quality: results = [r for r in results if r.quality == 'A+']
            elif "只看 A+ 和 A" in sp_quality: results = [r for r in results if r.quality in ['A+', 'A']]
            st.session_state['sp_results'] = results
            
        if 'sp_results' in st.session_state:
            res = st.session_state['sp_results']
            st.success(f"找到 {len(res)} 個安全收租機會")
            for c in res[:10]:
                em = "⭐" if c.quality == 'A+' else "✅" if c.quality == 'A' else "⚠️"
                with st.expander(f"{em} **{c.ticker}** | {c.quality} | {c.score:.0f}分"):
                    c1, c2, c3 = st.columns(3)
                    with c1: 
                        st.write(f"回調幅度: {c.pullback_depth:.1f}%")
                        st.write(f"RSI: {c.rsi:.0f}")
                        st.write(f"MACD止跌: {'✅' if c.macd_reversal else '❌'}")
                    with c2: 
                        st.write(f"最近支撐: {c.nearest_support}")
                        st.write(f"支撐位: ${c.nearest_support_price:.2f}")
                        st.write(f"距離支撐: {c.distance_to_support:.1f}%")
                    with c3: 
                        st.write(f"HV Rank: {c.hv_rank:.0f}%")
                        st.write(f"SMA200: {'✅ 上方' if c.above_sma200 else '❌ 下方'}")
                    
                    st.divider()
                    for note in c.notes: st.write(note)
                    st.divider()
                    st.markdown("### 💰 收租計畫")
                    st.warning(f"**建議賣出 Put 的行使價 (Strike): ${c.suggested_strike:.2f}** (位於強支撐下方，具備極高安全邊際)")

    # ===== TAB 3: VCP Screener =====
    with tabs[2]:
        st.header("🎯 VCP 橫盤爆發選股器")
        c1, c2 = st.columns(2)
        with c1: vcp_scope = st.selectbox("掃描範圍", ["🔥 熱門領導股", "🔬 半導體", "💻 軟件雲端", "🚀 高成長"], key="vcp_scope")
        with c2: vcp_quality = st.selectbox("最低質量", ["全部", "只看 A+ 和 A", "只看 A+"], key="vcp_quality")
        
        if st.button("🔍 批量掃描 VCP", type="primary"):
            stocks = STOCK_UNIVERSE['Market Leaders'] if "熱門" in vcp_scope else STOCK_UNIVERSE['Semiconductors'] if "半導體" in vcp_scope else STOCK_UNIVERSE['Software & Cloud'] if "軟件" in vcp_scope else STOCK_UNIVERSE['High Growth']
            spy_df = BatchDataFetcher.get_single_stock('SPY', '1y')
            screener, pb, st_txt = VCPScreener(), st.progress(0), st.empty()
            
            def upd(i, t, tic): pb.progress((i + 1) / t); st_txt.text(f"掃描 {tic}...")
            with st.spinner("批量下載數據..."): results = screener.scan_batch(stocks, spy_df, upd)
            pb.empty(); st_txt.empty()
            
            if "只看 A+" in vcp_quality: results = [r for r in results if r.quality == 'A+']
            elif "只看 A+ 和 A" in vcp_quality: results = [r for r in results if r.quality in ['A+', 'A']]
            st.session_state['vcp_results'] = results
            
        if 'vcp_results' in st.session_state:
            res = st.session_state['vcp_results']
            st.success(f"找到 {len(res)} 個 VCP 機會")
            for c in res[:10]:
                em = "⭐" if c.quality == 'A+' else "✅" if c.quality == 'A' else "⚠️"
                with st.expander(f"{em} **{c.ticker}** | {c.quality} | {c.score:.0f}分 | BB {c.bb_width:.3f}"):
                    st.write(f"**Pivot:** ${c.pivot_price:.2f} | **Entry:** ${c.entry_price:.2f} | **Stop:** ${c.stop_loss:.2f}")
                    for note in c.notes: st.write(note)
                    if st.button("查看圖表", key=f"vcp_{c.ticker}"):
                        df = BatchDataFetcher.get_single_stock(c.ticker, "6mo")
                        fig = ChartBuilder.create_chart_with_annotations(df, c.ticker, pivot=c.pivot_price, entry=c.entry_price, stop=c.stop_loss)
                        st.plotly_chart(fig, use_container_width=True)

    # ===== TAB 4: Stock Analysis =====
    with tabs[3]:
        st.header("📊 個股深度分析")
        ticker = st.text_input("股票代碼", value="AAPL").upper()
        if st.button("🔍 分析", type="primary", key="analyze"):
            with st.spinner("分析中..."):
                df = BatchDataFetcher.get_single_stock(ticker, "1y")
                spy_df = BatchDataFetcher.get_single_stock('SPY', '1y')
            if df is not None:
                ta, pcr_calc, iv_calc = TechnicalAnalysis(), PCRCalculator(), RealIVCalculator()
                curr_price = float(df['Close'].iloc[-1])
                rsi, adx_val = float(ta.rsi(df['Close']).iloc[-1]), ta.adx(df).iloc[-1]
                adx = float(adx_val) if not pd.isna(adx_val) else 0
                beta = ta.calculate_beta(df, spy_df)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("價格", f"${curr_price:.2f}"); c2.metric("RSI", f"{rsi:.0f}")
                c3.metric("ADX", f"{adx:.1f}"); c4.metric("Beta", f"{beta:.2f}")
                
                bb_width = float(ta.bollinger_band_width(df['Close']).iloc[-1])
                hv_rank = ta.estimate_hv_rank(df)
                rs = ta.rs_rating(df, spy_df)
                vol_sig = ta.analyze_volume_signature(df)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("BB Width", f"{bb_width:.3f}"); c2.metric("HV Rank", f"{hv_rank:.0f}%")
                c3.metric("RS Rating", f"{rs:.0f}"); c4.metric("量能健康", "✅" if vol_sig['is_healthy'] else "⚠️")
                
                st.subheader("📊 支撐位分析")
                support_data = ta.calculate_support_levels(df)
                c1, c2, c3 = st.columns(3)
                c1.metric("最近支撐", f"${support_data.get('nearest_support_price', 0):.2f}", f"{support_data.get('nearest_support', 'N/A')}")
                c2.metric("距離支撐", f"{support_data.get('distance_pct', 0):.1f}%")
                c3.metric("SMA200", f"${support_data.get('sma200_val', 0):.2f}")
                
                st.subheader("📊 期權情緒數據")
                c1, c2 = st.columns(2)
                with c1:
                    pcr_data = pcr_calc.get_pcr(ticker)
                    if pcr_data.get('status') == 'OK':
                        st.metric("PCR (OI)", f"{pcr_data['pcr_oi']:.2f}")
                        st.write(pcr_data['sentiment'])
                    else: st.write("PCR 數據不可用")
                with c2:
                    iv_data = iv_calc.get_real_iv(ticker)
                    if iv_data.get('iv'):
                        st.metric("真實 IV", f"{iv_data['iv']:.1f}%")
                        st.write(f"到期日: {iv_data.get('expiry', 'N/A')}")
                    else: st.write(f"IV: 使用 HV Rank {hv_rank:.0f}% 估算")
                
                st.subheader("📈 技術圖表")
                fig = ChartBuilder.create_chart_with_annotations(df, ticker, support=support_data.get('nearest_support_price'))
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
