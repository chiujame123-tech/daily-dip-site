# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v8.0 Ultimate Edition
=================================================

新增功能：
✅ Short Put 收租選股器 (Income Generator)
✅ VCP 橫盤爆發選股器 (Enhanced)
✅ PCR (Put/Call Ratio) 指標
✅ IV Rank 估算
✅ ADX 指標
✅ Bollinger Band Width

Author: Pro Trader AI
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
# ⚙️ CONFIGURATION
# ============================================
@dataclass
class Config:
    PAGE_TITLE: str = "Market Radar v8.0 Ultimate"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800

CONFIG = Config()

# ============================================
# 📊 STOCK UNIVERSE
# ============================================
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
    'China ADR': [
        'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES',
        'BEKE', 'FUTU', 'TIGR', 'TAL', 'EDU', 'YUMC', 'ZTO', 'QFIN', 'VIPS', 'HTHT'
    ],
}

ALL_STOCKS = list(set([s for stocks in STOCK_UNIVERSE.values() for s in stocks]))
ALL_STOCKS.sort()

SECTORS = {
    'SMH (半導體)': {'etf': 'SMH', 'holdings': STOCK_UNIVERSE['Semiconductors'][:12]},
    'XLK (科技)': {'etf': 'XLK', 'holdings': STOCK_UNIVERSE['Software & Cloud'][:12]},
    'ARKK (成長)': {'etf': 'ARKK', 'holdings': STOCK_UNIVERSE['High Growth'][:12]},
    'KWEB (中概)': {'etf': 'KWEB', 'holdings': STOCK_UNIVERSE['China ADR'][:12]},
}

# ============================================
# 🧮 TECHNICAL ANALYSIS (擴展版)
# ============================================
class TechnicalAnalysis:
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index
        < 20: 無趨勢 (適合收租)
        20-40: 趨勢發展中
        > 40: 強趨勢
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        # Smoothed values
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def bollinger_band_width(prices: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
        """
        Bollinger Band Width
        低於 0.10-0.15 表示波動收窄 (VCP 特徵)
        """
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + std * std_mult
        lower = sma - std * std_mult
        
        bb_width = (upper - lower) / sma
        return bb_width
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def adr_percent(df: pd.DataFrame, period: int = 20) -> pd.Series:
        daily_range = (df['High'] / df['Low'] - 1) * 100
        return daily_range.rolling(period).mean()
    
    @staticmethod
    def rs_rating(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
        if len(stock_df) < 63 or spy_df is None or len(spy_df) < 63:
            return 50
        
        periods = [21, 42, 63]
        weights = [0.4, 0.3, 0.3]
        score = 0
        
        for period, weight in zip(periods, weights):
            try:
                stock_ret = (float(stock_df['Close'].iloc[-1]) / float(stock_df['Close'].iloc[-period]) - 1) * 100
                spy_ret = (float(spy_df['Close'].iloc[-1]) / float(spy_df['Close'].iloc[-period]) - 1) * 100
                score += (stock_ret - spy_ret) * weight
            except:
                pass
        
        rs = 50 + (score / 30) * 49
        return max(1, min(99, rs))
    
    @staticmethod
    def calculate_beta(stock_df: pd.DataFrame, spy_df: pd.DataFrame, period: int = 252) -> float:
        """計算 Beta (相對於 SPY)"""
        if len(stock_df) < period or spy_df is None or len(spy_df) < period:
            return 1.0
        
        try:
            stock_returns = stock_df['Close'].pct_change().tail(period).dropna()
            spy_returns = spy_df['Close'].pct_change().tail(period).dropna()
            
            # 對齊日期
            common_idx = stock_returns.index.intersection(spy_returns.index)
            stock_returns = stock_returns.loc[common_idx]
            spy_returns = spy_returns.loc[common_idx]
            
            covariance = np.cov(stock_returns, spy_returns)[0][1]
            variance = np.var(spy_returns)
            
            beta = covariance / variance if variance > 0 else 1.0
            return round(beta, 2)
        except:
            return 1.0
    
    @staticmethod
    def estimate_iv_rank(df: pd.DataFrame, period: int = 252) -> float:
        """
        估算 IV Rank (使用歷史波動率作為代理)
        真正的 IV Rank 需要期權數據，這裡用 HV Rank 代替
        """
        if len(df) < period:
            return 50
        
        try:
            returns = df['Close'].pct_change().dropna()
            
            # 計算滾動 20 天歷史波動率
            hv_20 = returns.rolling(20).std() * np.sqrt(252) * 100
            
            # 計算過去一年的 HV 範圍
            hv_values = hv_20.tail(period).dropna()
            if len(hv_values) < 20:
                return 50
            
            current_hv = float(hv_values.iloc[-1])
            hv_min = float(hv_values.min())
            hv_max = float(hv_values.max())
            
            if hv_max - hv_min == 0:
                return 50
            
            iv_rank = (current_hv - hv_min) / (hv_max - hv_min) * 100
            return round(max(0, min(100, iv_rank)), 1)
        except:
            return 50
    
    @staticmethod
    def high_low_range(df: pd.DataFrame, period: int = 20) -> float:
        """計算過去 N 天的高低差百分比"""
        if len(df) < period:
            return 100
        
        recent = df.tail(period)
        high = float(recent['High'].max())
        low = float(recent['Low'].min())
        
        range_pct = (high / low - 1) * 100
        return round(range_pct, 2)
    
    @staticmethod
    def relative_volume(df: pd.DataFrame, period: int = 50) -> float:
        """相對成交量 (今日 / 平均)"""
        if len(df) < period:
            return 1.0
        
        avg_vol = float(df['Volume'].tail(period).mean())
        current_vol = float(df['Volume'].iloc[-1])
        
        return round(current_vol / avg_vol, 2) if avg_vol > 0 else 1.0


# ============================================
# 📊 PCR CALCULATOR
# ============================================
class PCRCalculator:
    """Put/Call Ratio 計算器"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_pcr(ticker: str) -> Dict:
        """
        獲取 PCR 相關數據
        使用 yfinance 的期權數據
        """
        try:
            stock = yf.Ticker(ticker)
            
            # 獲取期權到期日列表
            exp_dates = stock.options
            if not exp_dates or len(exp_dates) == 0:
                return {'pcr': None, 'status': 'No options data'}
            
            # 使用最近的到期日
            nearest_exp = exp_dates[0]
            
            # 獲取期權鏈
            opt_chain = stock.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            if calls.empty or puts.empty:
                return {'pcr': None, 'status': 'Empty chain'}
            
            # 計算 PCR (基於 Open Interest)
            call_oi = calls['openInterest'].sum()
            put_oi = puts['openInterest'].sum()
            
            pcr_oi = put_oi / call_oi if call_oi > 0 else 0
            
            # 計算 PCR (基於 Volume)
            call_vol = calls['volume'].sum()
            put_vol = puts['volume'].sum()
            
            pcr_vol = put_vol / call_vol if call_vol > 0 else 0
            
            # 計算隱含波動率 (取 ATM 附近的平均)
            current_price = float(stock.history(period='1d')['Close'].iloc[-1])
            
            atm_calls = calls[abs(calls['strike'] - current_price) / current_price < 0.05]
            atm_puts = puts[abs(puts['strike'] - current_price) / current_price < 0.05]
            
            avg_iv = 0
            if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                call_iv = atm_calls['impliedVolatility'].mean()
                put_iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else call_iv
                avg_iv = (call_iv + put_iv) / 2 * 100
            
            # PCR 解讀
            if pcr_oi > 1.2:
                sentiment = "🐻 看跌情緒高 (可能是反向指標)"
            elif pcr_oi < 0.7:
                sentiment = "🐂 看漲情緒高"
            else:
                sentiment = "😐 中性"
            
            return {
                'pcr_oi': round(pcr_oi, 2),
                'pcr_vol': round(pcr_vol, 2),
                'call_oi': int(call_oi),
                'put_oi': int(put_oi),
                'iv': round(avg_iv, 1),
                'expiry': nearest_exp,
                'sentiment': sentiment,
                'status': 'OK'
            }
            
        except Exception as e:
            return {'pcr': None, 'status': f'Error: {str(e)}'}


# ============================================
# 💰 SHORT PUT SCREENER
# ============================================
@dataclass
class ShortPutCandidate:
    """Short Put 候選股"""
    ticker: str
    price: float
    
    # 趨勢過濾
    adx: float
    above_sma200: bool
    
    # 震盪過濾
    rsi: float
    beta: float
    
    # 波動率
    iv_rank: float
    bb_width: float
    
    # 技術
    high_low_range_20d: float
    
    # PCR
    pcr_oi: float
    pcr_sentiment: str
    
    # 評分
    score: float
    quality: str
    
    # 期權建議
    suggested_strike: float
    suggested_premium_pct: float
    annual_return_est: float
    
    # 詳細
    notes: List[str]


class ShortPutScreener:
    """
    Short Put 收租選股器
    
    理想條件：
    1. ADX < 25 (無趨勢，適合賣 PUT)
    2. Price > SMA200 (長期牛市)
    3. RSI 40-60 (中性區間)
    4. Beta < 1.0 (穩定)
    5. IV Rank > 30 (有足夠權利金)
    """
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.pcr = PCRCalculator()
    
    def screen(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[ShortPutCandidate]:
        """掃描單個股票是否適合 Short Put"""
        if df is None or len(df) < 200:
            return None
        
        try:
            close = df['Close']
            curr_price = float(close.iloc[-1])
            
            # 計算指標
            adx = float(self.ta.adx(df).iloc[-1]) if not pd.isna(self.ta.adx(df).iloc[-1]) else 30
            sma200 = float(close.rolling(200).mean().iloc[-1])
            above_sma200 = curr_price > sma200
            
            rsi = float(self.ta.rsi(close).iloc[-1]) if not pd.isna(self.ta.rsi(close).iloc[-1]) else 50
            beta = self.ta.calculate_beta(df, spy_df) if spy_df is not None else 1.0
            
            iv_rank = self.ta.estimate_iv_rank(df)
            bb_width = float(self.ta.bollinger_band_width(close).iloc[-1]) if not pd.isna(self.ta.bollinger_band_width(close).iloc[-1]) else 0.2
            
            high_low_range = self.ta.high_low_range(df, 20)
            
            # 獲取 PCR
            pcr_data = self.pcr.get_pcr(ticker)
            pcr_oi = pcr_data.get('pcr_oi', 1.0) if pcr_data.get('status') == 'OK' else 1.0
            pcr_sentiment = pcr_data.get('sentiment', '😐 中性') if pcr_data.get('status') == 'OK' else '😐 N/A'
            
            # 評分系統
            score = 0
            notes = []
            
            # 1. ADX 評分 (最高 25 分)
            if adx < 20:
                score += 25
                notes.append(f"✅ ADX {adx:.1f} - 無趨勢，完美收租環境")
            elif adx < 25:
                score += 20
                notes.append(f"✅ ADX {adx:.1f} - 弱趨勢，適合收租")
            elif adx < 30:
                score += 10
                notes.append(f"⚠️ ADX {adx:.1f} - 有趨勢，謹慎")
            else:
                score -= 10
                notes.append(f"❌ ADX {adx:.1f} - 強趨勢，不適合 Short Put")
            
            # 2. SMA200 評分 (最高 20 分)
            if above_sma200:
                dist_from_sma = (curr_price / sma200 - 1) * 100
                if dist_from_sma > 10:
                    score += 15
                    notes.append(f"✅ 在 SMA200 上方 {dist_from_sma:.1f}% - 長期牛市")
                else:
                    score += 20
                    notes.append(f"✅ 在 SMA200 上方 {dist_from_sma:.1f}% - 穩健位置")
            else:
                score -= 15
                notes.append(f"❌ 在 SMA200 下方 - 不建議 Short Put")
            
            # 3. RSI 評分 (最高 20 分)
            if 40 <= rsi <= 60:
                score += 20
                notes.append(f"✅ RSI {rsi:.0f} - 完美中性區間")
            elif 35 <= rsi <= 65:
                score += 15
                notes.append(f"✅ RSI {rsi:.0f} - 可接受範圍")
            elif rsi < 35:
                score += 10
                notes.append(f"⚠️ RSI {rsi:.0f} - 超賣，可能反彈 (賣 PUT 好時機)")
            else:
                score += 5
                notes.append(f"⚠️ RSI {rsi:.0f} - 超買，小心回調")
            
            # 4. Beta 評分 (最高 15 分)
            if beta < 0.8:
                score += 15
                notes.append(f"✅ Beta {beta:.2f} - 非常穩定")
            elif beta < 1.0:
                score += 12
                notes.append(f"✅ Beta {beta:.2f} - 穩定")
            elif beta < 1.2:
                score += 8
                notes.append(f"⚠️ Beta {beta:.2f} - 中等波動")
            else:
                score += 3
                notes.append(f"❌ Beta {beta:.2f} - 高波動，風險較大")
            
            # 5. IV Rank 評分 (最高 20 分)
            if iv_rank >= 50:
                score += 20
                notes.append(f"✅ IV Rank {iv_rank:.0f}% - 高權利金環境")
            elif iv_rank >= 30:
                score += 15
                notes.append(f"✅ IV Rank {iv_rank:.0f}% - 不錯的權利金")
            elif iv_rank >= 20:
                score += 10
                notes.append(f"⚠️ IV Rank {iv_rank:.0f}% - 權利金一般")
            else:
                score += 5
                notes.append(f"❌ IV Rank {iv_rank:.0f}% - 權利金較低")
            
            # 必要條件檢查
            if not above_sma200:
                score = min(score, 40)  # 不在 SMA200 上方，強制降低分數
            
            # 質量評定
            if score >= 80:
                quality = 'A+'
            elif score >= 65:
                quality = 'A'
            elif score >= 50:
                quality = 'B'
            else:
                quality = 'C'
            
            # 計算建議 Strike 和預期回報
            suggested_strike = round(curr_price * 0.90, 2)  # 10% OTM PUT
            
            # 預估權利金 (簡化計算，實際需要 Black-Scholes)
            est_premium_pct = max(1.0, iv_rank * 0.05)  # 粗略估計
            annual_return = est_premium_pct * 12  # 每月到期的年化
            
            return ShortPutCandidate(
                ticker=ticker,
                price=curr_price,
                adx=adx,
                above_sma200=above_sma200,
                rsi=rsi,
                beta=beta,
                iv_rank=iv_rank,
                bb_width=bb_width,
                high_low_range_20d=high_low_range,
                pcr_oi=pcr_oi,
                pcr_sentiment=pcr_sentiment,
                score=score,
                quality=quality,
                suggested_strike=suggested_strike,
                suggested_premium_pct=est_premium_pct,
                annual_return_est=annual_return,
                notes=notes
            )
            
        except Exception as e:
            return None
    
    def scan_all(self, stocks: List[str], spy_df: pd.DataFrame, progress_callback=None) -> List[ShortPutCandidate]:
        """掃描所有股票"""
        results = []
        
        for i, ticker in enumerate(stocks):
            if progress_callback:
                progress_callback(i, len(stocks), ticker)
            
            try:
                df = yf.download(ticker, period='1y', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is not None and len(df) >= 200:
                    candidate = self.screen(df, ticker, spy_df)
                    if candidate and candidate.score >= 40 and candidate.above_sma200:
                        results.append(candidate)
            except:
                continue
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results


# ============================================
# 🎯 VCP SCREENER (增強版)
# ============================================
@dataclass
class VCPCandidate:
    """VCP 候選股"""
    ticker: str
    price: float
    
    # 趨勢
    above_sma50: bool
    above_sma200: bool
    dist_from_52w_high: float
    
    # 橫盤
    bb_width: float
    high_low_range_20d: float
    
    # 動能
    rsi: float
    rs_rating: float
    
    # 成交量
    relative_volume: float
    
    # PCR
    pcr_oi: float
    pcr_sentiment: str
    
    # VCP 特徵
    tightness: float
    contraction_count: int
    pivot_price: float
    
    # 評分
    score: float
    quality: str
    
    # 交易計劃
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    
    # 詳細
    notes: List[str]


class VCPScreener:
    """
    VCP 橫盤爆發選股器 (增強版)
    
    黃金組合條件：
    1. Price > SMA50 & SMA200 (上升趨勢中的橫盤)
    2. BB Width < 0.15 或 High/Low 20d < 10%
    3. Price within 15% of 52-Week High (強勢股高位整理)
    4. RSI 45-65 (橫盤特徵)
    5. Relative Volume < 0.75 (量縮)
    6. PCR 分析
    """
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.pcr = PCRCalculator()
    
    def screen(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[VCPCandidate]:
        """掃描單個股票是否符合 VCP 條件"""
        if df is None or len(df) < 100:
            return None
        
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            curr_price = float(close.iloc[-1])
            
            # 計算指標
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else sma50
            
            above_sma50 = curr_price > sma50
            above_sma200 = curr_price > sma200
            
            # 52 週高點距離
            high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
            dist_from_high = (curr_price / high_52w - 1) * 100
            
            # BB Width
            bb_width = float(self.ta.bollinger_band_width(close).iloc[-1])
            
            # 20 天高低範圍
            high_low_range = self.ta.high_low_range(df, 20)
            
            # RSI
            rsi = float(self.ta.rsi(close).iloc[-1])
            
            # RS Rating
            rs_rating = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            
            # 相對成交量
            rel_vol = self.ta.relative_volume(df)
            
            # PCR
            pcr_data = self.pcr.get_pcr(ticker)
            pcr_oi = pcr_data.get('pcr_oi', 1.0) if pcr_data.get('status') == 'OK' else 1.0
            pcr_sentiment = pcr_data.get('sentiment', '😐 N/A') if pcr_data.get('status') == 'OK' else '😐 N/A'
            
            # VCP 收縮分析
            recent = df.tail(40)
            contractions = []
            
            for i in range(0, min(35, len(recent)-5), 5):
                week = recent.iloc[i:i+5]
                week_range = (float(week['High'].max()) - float(week['Low'].min())) / float(week['Low'].min()) * 100
                contractions.append(week_range)
            
            contraction_count = 0
            if len(contractions) >= 3:
                for i in range(1, len(contractions)):
                    if contractions[i] < contractions[i-1] * 1.1:
                        contraction_count += 1
            
            final_tightness = contractions[-1] if contractions else 100
            
            # Pivot 價格
            pivot = float(recent['High'].max())
            
            # 評分系統
            score = 0
            notes = []
            
            # 1. 趨勢 (最高 25 分)
            if above_sma50 and above_sma200:
                score += 25
                notes.append("✅ 在 SMA50 和 SMA200 之上 - 健康上升趨勢")
            elif above_sma50:
                score += 15
                notes.append("⚠️ 在 SMA50 之上，但低於 SMA200")
            else:
                score -= 10
                notes.append("❌ 在 SMA50 之下 - 不符合 VCP")
            
            # 2. 橫盤/收縮 (最高 25 分)
            if bb_width < 0.10:
                score += 25
                notes.append(f"✅ BB Width {bb_width:.2f} - 極度收窄")
            elif bb_width < 0.15:
                score += 20
                notes.append(f"✅ BB Width {bb_width:.2f} - 收窄中")
            elif bb_width < 0.20:
                score += 10
                notes.append(f"⚠️ BB Width {bb_width:.2f} - 略寬")
            else:
                score += 0
                notes.append(f"❌ BB Width {bb_width:.2f} - 波動太大")
            
            # 3. 52 週高點距離 (最高 15 分)
            if dist_from_high >= -5:
                score += 15
                notes.append(f"✅ 距52週高點 {dist_from_high:.1f}% - 強勢股")
            elif dist_from_high >= -15:
                score += 12
                notes.append(f"✅ 距52週高點 {dist_from_high:.1f}% - 高位整理")
            elif dist_from_high >= -25:
                score += 8
                notes.append(f"⚠️ 距52週高點 {dist_from_high:.1f}%")
            else:
                score += 0
                notes.append(f"❌ 距52週高點 {dist_from_high:.1f}% - 太弱")
            
            # 4. RSI (最高 15 分)
            if 45 <= rsi <= 65:
                score += 15
                notes.append(f"✅ RSI {rsi:.0f} - 完美橫盤區間")
            elif 40 <= rsi <= 70:
                score += 10
                notes.append(f"⚠️ RSI {rsi:.0f} - 可接受")
            else:
                score += 0
                notes.append(f"❌ RSI {rsi:.0f} - 不理想")
            
            # 5. 成交量萎縮 (最高 10 分)
            if rel_vol < 0.6:
                score += 10
                notes.append(f"✅ 量縮 {rel_vol:.2f}x - 賣壓消失")
            elif rel_vol < 0.75:
                score += 8
                notes.append(f"✅ 量縮 {rel_vol:.2f}x")
            elif rel_vol < 1.0:
                score += 5
                notes.append(f"⚠️ 成交量 {rel_vol:.2f}x")
            else:
                score += 0
                notes.append(f"❌ 放量 {rel_vol:.2f}x - 不是收縮")
            
            # 6. RS Rating (最高 10 分)
            if rs_rating >= 80:
                score += 10
                notes.append(f"✅ RS {rs_rating:.0f} - 領漲股")
            elif rs_rating >= 70:
                score += 7
                notes.append(f"✅ RS {rs_rating:.0f}")
            else:
                score += 3
                notes.append(f"⚠️ RS {rs_rating:.0f}")
            
            # 必要條件
            if not above_sma50:
                score = min(score, 30)
            
            # 質量評定
            if score >= 80:
                quality = 'A+'
            elif score >= 65:
                quality = 'A'
            elif score >= 50:
                quality = 'B'
            else:
                quality = 'C'
            
            # 交易計劃
            atr = float(self.ta.atr(df).iloc[-1])
            entry = pivot * 1.001
            stop = max(float(recent['Low'].min()), pivot * 0.95) - atr * 0.2
            
            max_stop = entry * 0.06
            if entry - stop > max_stop:
                stop = entry - max_stop
            
            risk = entry - stop
            target_1 = entry + risk * 2
            target_2 = entry + risk * 3
            rr = (target_1 - entry) / risk if risk > 0 else 0
            
            return VCPCandidate(
                ticker=ticker,
                price=curr_price,
                above_sma50=above_sma50,
                above_sma200=above_sma200,
                dist_from_52w_high=dist_from_high,
                bb_width=bb_width,
                high_low_range_20d=high_low_range,
                rsi=rsi,
                rs_rating=rs_rating,
                relative_volume=rel_vol,
                pcr_oi=pcr_oi,
                pcr_sentiment=pcr_sentiment,
                tightness=final_tightness,
                contraction_count=contraction_count,
                pivot_price=pivot,
                score=score,
                quality=quality,
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                risk_reward=round(rr, 2),
                notes=notes
            )
            
        except:
            return None
    
    def scan_all(self, stocks: List[str], spy_df: pd.DataFrame, progress_callback=None) -> List[VCPCandidate]:
        """掃描所有股票"""
        results = []
        
        for i, ticker in enumerate(stocks):
            if progress_callback:
                progress_callback(i, len(stocks), ticker)
            
            try:
                df = yf.download(ticker, period='1y', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is not None and len(df) >= 100:
                    candidate = self.screen(df, ticker, spy_df)
                    if candidate and candidate.score >= 40 and candidate.above_sma50:
                        results.append(candidate)
            except:
                continue
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results


# ============================================
# 🌡️ MARKET REGIME
# ============================================
class MarketRegime:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_health() -> Dict:
        """
        獲取市場健康狀態
        增強版：更好的錯誤處理和備用數據
        """
        default = {
            'status': '🟡 謹慎', 
            'score': 60, 
            'vix': 18.0,  # 提供默認值而不是 None
            'spy_price': 500.0,  # 提供默認值
            'advice': '正常交易'
        }
        
        try:
            # 嘗試獲取 SPY 數據，增加超時處理
            try:
                spy = yf.download('SPY', period='6mo', progress=False, timeout=15)
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
            except Exception as e:
                print(f"SPY download error: {e}")
                spy = None
            
            # 如果 SPY 獲取失敗，返回默認值（但不是 N/A）
            if spy is None or len(spy) == 0:
                return default
            
            # 嘗試獲取 VIX
            vix_val = 18.0  # 默認值
            try:
                vix = yf.download('^VIX', period='5d', progress=False, timeout=10)
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                if vix is not None and len(vix) > 0 and 'Close' in vix.columns:
                    vix_val = float(vix['Close'].iloc[-1])
            except Exception as e:
                print(f"VIX download error: {e}")
                vix_val = 18.0  # 使用默認值
            
            # 計算 SPY 指標
            spy_close = float(spy['Close'].iloc[-1])
            sma50 = float(spy['Close'].rolling(50).mean().iloc[-1])
            sma200 = float(spy['Close'].rolling(200).mean().iloc[-1]) if len(spy) >= 200 else sma50
            
            # 計算健康評分
            score = 50
            
            # SPY 位置
            if spy_close > sma200: 
                score += 15
            if spy_close > sma50: 
                score += 10
            
            # 月度回報
            if len(spy) >= 21:
                ret = (spy_close / float(spy['Close'].iloc[-21]) - 1) * 100
                if ret > 0: 
                    score += 10
                elif ret < -5: 
                    score -= 15
            
            # VIX 水平
            if vix_val < 15: 
                score += 10
            elif vix_val > 25: 
                score -= 15
            
            # 確定狀態
            if score >= 75:
                status, advice = "🟢 強勢", "全力進攻"
            elif score >= 60:
                status, advice = "🟡 謹慎", "正常交易"
            elif score >= 40:
                status, advice = "🟠 震盪", "減少倉位"
            else:
                status, advice = "🔴 弱勢", "防守"
            
            return {
                'status': status, 
                'score': score, 
                'advice': advice,
                'vix': round(vix_val, 1), 
                'spy_price': round(spy_close, 2)
            }
            
        except Exception as e:
            print(f"MarketRegime error: {e}")
            return default


# ============================================
# 📡 DATA FETCHER
# ============================================
class DataFetcher:
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_stock(ticker: str, period: str = "1y"):
        """獲取股票數據，增加超時處理"""
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=15)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if df is not None and len(df) > 0 else None
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_sector_etfs():
        """獲取板塊 ETF 數據"""
        tickers = [s['etf'] for s in SECTORS.values()] + ['SPY']
        try:
            data = yf.download(tickers, period="6mo", progress=False, timeout=20)
            if data is not None and 'Close' in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else 'Close' in data.columns:
                if isinstance(data.columns, pd.MultiIndex):
                    return data['Close']
                return data
            return None
        except Exception as e:
            print(f"Error fetching sector ETFs: {e}")
            return None


# ============================================
# 📊 CHART BUILDER
# ============================================
class ChartBuilder:
    @staticmethod
    def create_chart(df: pd.DataFrame, ticker: str, show_bb: bool = True) -> go.Figure:
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        
        if show_bb:
            df['BB_Upper'] = df['SMA20'] + df['Close'].rolling(20).std() * 2
            df['BB_Lower'] = df['SMA20'] - df['Close'].rolling(20).std() * 2
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.75, 0.25])
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price',
            increasing_line_color='#00CC96', decreasing_line_color='#EF553B'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                                 line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                                 line=dict(color='blue', width=1)), row=1, col=1)
        
        if len(df) >= 200:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200',
                                     line=dict(color='purple', width=1.5)), row=1, col=1)
        
        if show_bb:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                                     line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                                     line=dict(color='gray', width=1, dash='dash'),
                                     fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        colors = ['#00CC96' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF553B' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors,
                            name='Volume', showlegend=False), row=2, col=1)
        
        fig.update_layout(
            height=600, showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            title=ticker
        )
        
        return fig


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v8.0 Ultimate")
    st.caption("Short Put 收租選股器 | VCP 橫盤爆發選股器 | PCR 分析")
    
    # Market Health
    market = MarketRegime.get_health()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("市場狀態", market['status'])
    col2.metric("健康評分", f"{market['score']}/100")
    col3.metric("VIX", f"{market.get('vix', 18.0):.1f}")
    col4.metric("SPY", f"${market.get('spy_price', 500.0):.2f}")
    col5.metric("建議", market['advice'])
    
    st.divider()
    
    # Tabs
    tabs = st.tabs([
        "🌪️ 板塊輪動",
        "📊 個股分析",
        "💰 Short Put 收租",
        "🎯 VCP 橫盤爆發"
    ])
    
    # ===== TAB 1: Sector Rotation =====
    with tabs[0]:
        st.header("板塊相對強度")
        
        df_etf = DataFetcher.get_sector_etfs()
        if df_etf is not None:
            timeframe = st.selectbox("時間軸", [5, 21, 63], format_func=lambda x: f"{x}天", index=1)
            
            returns = df_etf.pct_change(periods=timeframe).iloc[-1] * 100
            spy_return = returns.get('SPY', 0)
            
            rs_data = []
            for name, info in SECTORS.items():
                if info['etf'] in returns:
                    rs_data.append({
                        '板塊': name,
                        'RS': returns[info['etf']] - spy_return,
                        '回報%': returns[info['etf']]
                    })
            
            if rs_data:
                df_rs = pd.DataFrame(rs_data).sort_values('RS', ascending=False)
                
                fig = px.bar(df_rs, x='RS', y='板塊', orientation='h', color='RS',
                            color_continuous_scale=['#FF4B4B', '#F0F2F6', '#00CC96'],
                            range_color=[-15, 15])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: Stock Analysis =====
    with tabs[1]:
        st.header("📊 個股分析")
        
        ticker = st.text_input("股票代碼", value="AAPL").upper()
        
        if st.button("🔍 分析", type="primary", key="analyze"):
            df = DataFetcher.get_stock(ticker, "1y")
            spy_df = DataFetcher.get_stock('SPY', '1y')
            
            if df is not None:
                ta = TechnicalAnalysis()
                pcr_calc = PCRCalculator()
                
                # 計算指標
                curr_price = float(df['Close'].iloc[-1])
                rsi = float(ta.rsi(df['Close']).iloc[-1])
                adx = float(ta.adx(df).iloc[-1]) if not pd.isna(ta.adx(df).iloc[-1]) else 0
                bb_width = float(ta.bollinger_band_width(df['Close']).iloc[-1])
                beta = ta.calculate_beta(df, spy_df)
                iv_rank = ta.estimate_iv_rank(df)
                rs_rating = ta.rs_rating(df, spy_df)
                
                # 基本信息
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("價格", f"${curr_price:.2f}")
                col2.metric("RSI", f"{rsi:.0f}")
                col3.metric("ADX", f"{adx:.1f}")
                col4.metric("Beta", f"{beta:.2f}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("BB Width", f"{bb_width:.3f}")
                col2.metric("IV Rank", f"{iv_rank:.0f}%")
                col3.metric("RS Rating", f"{rs_rating:.0f}")
                col4.metric("Rel Volume", f"{ta.relative_volume(df):.2f}x")
                
                # PCR
                pcr_data = pcr_calc.get_pcr(ticker)
                if pcr_data.get('status') == 'OK':
                    st.subheader("📊 期權數據")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PCR (OI)", f"{pcr_data['pcr_oi']:.2f}")
                    col2.metric("PCR (Vol)", f"{pcr_data['pcr_vol']:.2f}")
                    col3.metric("IV", f"{pcr_data['iv']:.1f}%")
                    col4.metric("情緒", pcr_data['sentiment'])
                
                # 圖表
                fig = ChartBuilder.create_chart(df, ticker, show_bb=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # 策略建議
                st.subheader("💡 策略建議")
                
                if adx < 25 and 40 <= rsi <= 60 and curr_price > float(df['Close'].rolling(200).mean().iloc[-1]):
                    st.success("✅ **適合 Short Put 策略** - 無趨勢、中性、長期牛市")
                
                if bb_width < 0.15 and ta.relative_volume(df) < 0.8:
                    st.info("🎯 **可能是 VCP 形態** - 波幅收窄、量縮")
    
    # ===== TAB 3: Short Put Screener =====
    with tabs[2]:
        st.header("💰 Short Put 收租選股器")
        
        st.info("""
        **收租策略條件：**
        - ADX < 25 (無趨勢)
        - Price > SMA200 (長期牛市)
        - RSI 40-60 (中性)
        - Beta < 1.0 (穩定)
        - IV Rank > 30 (有權利金)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            scan_scope = st.selectbox(
                "掃描範圍",
                [
                    "🏦 藍籌股 (30隻)",
                    "💵 高息股 (20隻)",
                    "🔥 熱門股 (20隻)",
                    "📊 全部 (100+)"
                ],
                key="short_put_scope"
            )
        with col2:
            min_quality = st.selectbox(
                "最低質量",
                ["全部", "只看 A+ 和 A", "只看 A+"],
                key="short_put_quality"
            )
        
        if st.button("🔍 掃描收租機會", type="primary", key="scan_short_put"):
            if "藍籌" in scan_scope:
                stocks = STOCK_UNIVERSE['Blue Chips (Short Put)']
            elif "高息" in scan_scope:
                stocks = STOCK_UNIVERSE['Dividend Stocks']
            elif "熱門" in scan_scope:
                stocks = STOCK_UNIVERSE['Market Leaders']
            else:
                stocks = ALL_STOCKS[:100]
            
            spy_df = DataFetcher.get_stock('SPY', '1y')
            screener = ShortPutScreener()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(i, total, ticker):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"掃描 {ticker} ({i+1}/{total})...")
            
            results = screener.scan_all(stocks, spy_df, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            # 過濾質量
            if "只看 A+" in min_quality:
                results = [r for r in results if r.quality == 'A+']
            elif "只看 A+ 和 A" in min_quality:
                results = [r for r in results if r.quality in ['A+', 'A']]
            
            st.session_state['short_put_results'] = results
        
        # 顯示結果
        if 'short_put_results' in st.session_state:
            results = st.session_state['short_put_results']
            
            st.success(f"找到 {len(results)} 個收租機會")
            
            if results:
                for candidate in results[:10]:
                    quality_emoji = "⭐" if candidate.quality == 'A+' else "✅" if candidate.quality == 'A' else "⚠️"
                    
                    with st.expander(f"{quality_emoji} **{candidate.ticker}** | {candidate.quality} | {candidate.score:.0f}分 | ${candidate.price:.2f}"):
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**趨勢指標:**")
                            st.write(f"- ADX: {candidate.adx:.1f}")
                            st.write(f"- SMA200: {'✅ 上方' if candidate.above_sma200 else '❌ 下方'}")
                        
                        with col2:
                            st.markdown("**震盪指標:**")
                            st.write(f"- RSI: {candidate.rsi:.0f}")
                            st.write(f"- Beta: {candidate.beta:.2f}")
                        
                        with col3:
                            st.markdown("**期權指標:**")
                            st.write(f"- IV Rank: {candidate.iv_rank:.0f}%")
                            st.write(f"- PCR: {candidate.pcr_oi:.2f} {candidate.pcr_sentiment}")
                        
                        st.divider()
                        
                        st.markdown("**📝 詳細分析:**")
                        for note in candidate.notes:
                            st.write(note)
                        
                        st.divider()
                        
                        st.markdown("**💰 Short Put 建議:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            - **建議 Strike:** ${candidate.suggested_strike:.2f} (約 10% OTM)
                            - **預估權利金:** {candidate.suggested_premium_pct:.1f}%
                            - **年化回報:** {candidate.annual_return_est:.0f}%
                            """)
                        with col2:
                            st.markdown(f"""
                            - **到期日:** 30-45 天
                            - **Delta:** 0.20-0.30
                            - **所需保證金:** ${candidate.suggested_strike * 100:,.0f}/合約
                            """)
                        
                        # 查看圖表
                        if st.button(f"📈 查看 {candidate.ticker} 圖表", key=f"sp_chart_{candidate.ticker}"):
                            df = DataFetcher.get_stock(candidate.ticker, "6mo")
                            if df is not None:
                                fig = ChartBuilder.create_chart(df, candidate.ticker)
                                st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 4: VCP Screener =====
    with tabs[3]:
        st.header("🎯 VCP 橫盤爆發選股器")
        
        st.info("""
        **VCP 黃金組合條件：**
        - Price > SMA50 & SMA200 (上升趨勢)
        - BB Width < 0.15 (波幅收窄)
        - 20天高低差 < 10%
        - 距52週高點 < 15%
        - RSI 45-65 (橫盤特徵)
        - 相對成交量 < 0.75 (量縮)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            scan_scope = st.selectbox(
                "掃描範圍",
                [
                    "🔥 熱門領導股 (20隻)",
                    "🔬 半導體 (20隻)",
                    "💻 軟件雲端 (20隻)",
                    "🚀 高成長股 (20隻)",
                    "📊 全部 (100+)"
                ],
                key="vcp_scope"
            )
        with col2:
            min_quality = st.selectbox(
                "最低質量",
                ["全部", "只看 A+ 和 A", "只看 A+"],
                key="vcp_quality"
            )
        
        if st.button("🔍 掃描 VCP 機會", type="primary", key="scan_vcp"):
            if "熱門" in scan_scope:
                stocks = STOCK_UNIVERSE['Market Leaders']
            elif "半導體" in scan_scope:
                stocks = STOCK_UNIVERSE['Semiconductors']
            elif "軟件" in scan_scope:
                stocks = STOCK_UNIVERSE['Software & Cloud']
            elif "高成長" in scan_scope:
                stocks = STOCK_UNIVERSE['High Growth']
            else:
                stocks = ALL_STOCKS[:100]
            
            spy_df = DataFetcher.get_stock('SPY', '1y')
            screener = VCPScreener()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(i, total, ticker):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"掃描 {ticker} ({i+1}/{total})...")
            
            results = screener.scan_all(stocks, spy_df, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            # 過濾質量
            if "只看 A+" in min_quality:
                results = [r for r in results if r.quality == 'A+']
            elif "只看 A+ 和 A" in min_quality:
                results = [r for r in results if r.quality in ['A+', 'A']]
            
            st.session_state['vcp_results'] = results
        
        # 顯示結果
        if 'vcp_results' in st.session_state:
            results = st.session_state['vcp_results']
            
            st.success(f"找到 {len(results)} 個 VCP 機會")
            
            if results:
                for candidate in results[:10]:
                    quality_emoji = "⭐" if candidate.quality == 'A+' else "✅" if candidate.quality == 'A' else "⚠️"
                    
                    with st.expander(f"{quality_emoji} **{candidate.ticker}** | {candidate.quality} | {candidate.score:.0f}分 | BB Width {candidate.bb_width:.3f}"):
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**趨勢:**")
                            st.write(f"- SMA50: {'✅' if candidate.above_sma50 else '❌'}")
                            st.write(f"- SMA200: {'✅' if candidate.above_sma200 else '❌'}")
                            st.write(f"- 52W High: {candidate.dist_from_52w_high:.1f}%")
                        
                        with col2:
                            st.markdown("**橫盤:**")
                            st.write(f"- BB Width: {candidate.bb_width:.3f}")
                            st.write(f"- 20D Range: {candidate.high_low_range_20d:.1f}%")
                            st.write(f"- 緊縮度: {candidate.tightness:.1f}%")
                        
                        with col3:
                            st.markdown("**動能:**")
                            st.write(f"- RSI: {candidate.rsi:.0f}")
                            st.write(f"- RS: {candidate.rs_rating:.0f}")
                            st.write(f"- 量比: {candidate.relative_volume:.2f}x")
                        
                        st.divider()
                        
                        st.markdown("**📊 PCR 分析:**")
                        st.write(f"PCR (OI): {candidate.pcr_oi:.2f} | {candidate.pcr_sentiment}")
                        
                        st.divider()
                        
                        st.markdown("**📝 詳細分析:**")
                        for note in candidate.notes:
                            st.write(note)
                        
                        st.divider()
                        
                        st.markdown("**📋 交易計劃:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            risk_pct = (candidate.entry_price - candidate.stop_loss) / candidate.entry_price * 100
                            st.markdown(f"""
                            - **Pivot:** ${candidate.pivot_price:.2f}
                            - **入場:** ${candidate.entry_price}
                            - **止損:** ${candidate.stop_loss}
                            - **風險:** {risk_pct:.1f}%
                            """)
                        with col2:
                            st.markdown(f"""
                            - **T1 (2R):** ${candidate.target_1}
                            - **T2 (3R):** ${candidate.target_2}
                            - **R:R:** {candidate.risk_reward}:1
                            """)
                        
                        # 查看圖表
                        if st.button(f"📈 查看 {candidate.ticker} 圖表", key=f"vcp_chart_{candidate.ticker}"):
                            df = DataFetcher.get_stock(candidate.ticker, "6mo")
                            if df is not None:
                                fig = ChartBuilder.create_chart(df, candidate.ticker, show_bb=True)
                                
                                # 添加 Pivot 線
                                fig.add_hline(y=candidate.pivot_price, line_dash="dash", 
                                             line_color="cyan", annotation_text=f"Pivot ${candidate.pivot_price:.2f}")
                                fig.add_hline(y=candidate.entry_price, line_dash="dash",
                                             line_color="green", annotation_text=f"Entry ${candidate.entry_price}")
                                fig.add_hline(y=candidate.stop_loss, line_dash="dash",
                                             line_color="red", annotation_text=f"Stop ${candidate.stop_loss}")
                                
                                st.plotly_chart(fig, use_container_width=True)
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v8.0 Ultimate")
    st.sidebar.markdown(f"""
    **新功能:**
    - ✅ Short Put 收租選股器
    - ✅ VCP 橫盤爆發選股器
    - ✅ PCR 分析
    - ✅ ADX 指標
    - ✅ BB Width 指標
    - ✅ IV Rank 估算
    - ✅ Beta 計算
    
    **股票數量:** {len(ALL_STOCKS)}+
    """)
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 指標說明")
    st.sidebar.markdown("""
    **ADX:**
    - <20: 無趨勢 ✅
    - 20-40: 有趨勢
    - >40: 強趨勢
    
    **BB Width:**
    - <0.10: 極度收窄 ✅
    - 0.10-0.15: 收窄
    - >0.20: 波動大
    
    **PCR:**
    - >1.2: 看跌情緒
    - <0.7: 看漲情緒
    - 0.7-1.2: 中性
    """)


if __name__ == "__main__":
    main()
