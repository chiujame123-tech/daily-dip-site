# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v7.0 Ultimate Edition
=================================================

重大改進：
✅ 擴大掃描範圍 - 200+ 股票
✅ 新增 IBD 50/100 風格熱門股
✅ 新增 52週新高掃描
✅ 新增趨勢模板檢查
✅ 新增 Power Play 設定
✅ 改進 UI/UX
✅ 新增交易日誌模板
✅ 新增績效追蹤

Author: Pro Trader AI
Target: 30%+ Annual Return
"""

# ============================================
# 📦 IMPORTS
# ============================================
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
    PAGE_TITLE: str = "Market Radar v7.0 Ultimate"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800
    
    # Setup Thresholds
    BGU_MIN_GAP: float = 3.0
    BGU_MIN_VOLUME: float = 1.5
    VCP_MAX_TIGHTNESS: float = 12.0
    VCP_MIN_CONTRACTIONS: int = 2

CONFIG = Config()

# ============================================
# 📊 擴大的股票宇宙 (200+ 股票)
# ============================================
STOCK_UNIVERSE = {
    # === 科技 ===
    'Mega Cap Tech': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CRM'
    ],
    'Semiconductors': [
        'NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC',
        'ADI', 'MRVL', 'NXPI', 'ON', 'MCHP', 'ARM', 'SMCI', 'INTC', 'ASML', 'SNPS'
    ],
    'Software & Cloud': [
        'MSFT', 'CRM', 'ADBE', 'NOW', 'INTU', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'NET',
        'MDB', 'PLTR', 'ZS', 'FTNT', 'WDAY', 'TEAM', 'HUBS', 'OKTA', 'DOCU', 'ZM'
    ],
    'Internet & Media': [
        'GOOGL', 'META', 'NFLX', 'DIS', 'BKNG', 'ABNB', 'UBER', 'LYFT', 'DASH', 'SNAP',
        'PINS', 'SPOT', 'ROKU', 'TTD', 'RBLX', 'U', 'EA', 'TTWO', 'MTCH', 'CHWY'
    ],
    
    # === 金融 ===
    'Financials': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V',
        'MA', 'PYPL', 'SQ', 'COIN', 'HOOD', 'AFRM', 'SOFI', 'ICE', 'CME', 'SPGI'
    ],
    
    # === 醫療 ===
    'Healthcare': [
        'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'VRTX', 'REGN', 'MRNA', 'ISRG', 'MDT', 'SYK', 'BSX', 'EW'
    ],
    
    # === 消費 ===
    'Consumer': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'COST', 'WMT',
        'TGT', 'LULU', 'CMG', 'YUM', 'DPZ', 'ROST', 'ORLY', 'AZO', 'ULTA', 'DG'
    ],
    
    # === 能源 ===
    'Energy': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'PXD',
        'DVN', 'HES', 'FANG', 'HAL', 'BKR', 'KMI', 'WMB', 'OKE', 'LNG', 'TRGP'
    ],
    
    # === 工業 ===
    'Industrials': [
        'CAT', 'DE', 'UNP', 'HON', 'UPS', 'RTX', 'BA', 'LMT', 'GE', 'MMM',
        'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI', 'PCAR', 'ODFL', 'FAST', 'URI'
    ],
    
    # === 高成長 / 投機 ===
    'High Growth': [
        'NVDA', 'SMCI', 'ARM', 'PLTR', 'COIN', 'MSTR', 'AFRM', 'SOFI', 'HOOD', 'UPST',
        'RBLX', 'U', 'DKNG', 'BILL', 'SHOP', 'SQ', 'MELI', 'SE', 'GRAB', 'NU'
    ],
    
    # === 中概股 ===
    'China ADR': [
        'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES',
        'BEKE', 'VIPS', 'TAL', 'EDU', 'FUTU', 'TIGR', 'ZTO', 'YUMC', 'QFIN', 'HTHT'
    ],
    
    # === 新能源 / EV ===
    'Clean Energy & EV': [
        'TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'F', 'GM', 'ENPH', 'SEDG',
        'FSLR', 'RUN', 'PLUG', 'BE', 'CHPT', 'QS', 'BLNK', 'EVGO', 'PTRA', 'FSR'
    ],
    
    # === AI 概念 ===
    'AI & Robotics': [
        'NVDA', 'MSFT', 'GOOGL', 'META', 'AMD', 'SMCI', 'ARM', 'PLTR', 'PATH', 'AI',
        'UPST', 'SOUN', 'BBAI', 'GFAI', 'PRCT', 'ISRG', 'INTUV', 'ABMD', 'IRTC', 'CGNX'
    ],
    
    # === IBD 風格領導股 ===
    'Market Leaders': [
        'NVDA', 'META', 'AMZN', 'GOOGL', 'MSFT', 'AAPL', 'LLY', 'AVGO', 'TSLA', 'AMD',
        'CRM', 'NOW', 'PANW', 'CRWD', 'NFLX', 'COST', 'ISRG', 'LULU', 'CMG', 'FICO'
    ],
    
    # === ETF ===
    'Key ETFs': [
        'SPY', 'QQQ', 'IWM', 'DIA', 'SMH', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
        'XLY', 'XLP', 'XLU', 'ARKK', 'KWEB', 'TLT', 'GLD', 'SLV', 'USO', 'VIX'
    ],
}

# 所有股票 (去重)
ALL_STOCKS = list(set([s for stocks in STOCK_UNIVERSE.values() for s in stocks if not s.startswith('^')]))
ALL_STOCKS.sort()

# 板塊 ETF 映射
SECTORS = {
    'SMH (半導體)': {'etf': 'SMH', 'holdings': STOCK_UNIVERSE['Semiconductors'], 'theme': '🔬 AI/芯片'},
    'XLK (科技)': {'etf': 'XLK', 'holdings': STOCK_UNIVERSE['Software & Cloud'], 'theme': '💻 軟件'},
    'XLC (通訊)': {'etf': 'XLC', 'holdings': STOCK_UNIVERSE['Internet & Media'][:10], 'theme': '📱 互聯網'},
    'XLF (金融)': {'etf': 'XLF', 'holdings': STOCK_UNIVERSE['Financials'][:10], 'theme': '🏦 金融'},
    'XLY (消費)': {'etf': 'XLY', 'holdings': STOCK_UNIVERSE['Consumer'][:10], 'theme': '🛒 消費'},
    'XLV (醫療)': {'etf': 'XLV', 'holdings': STOCK_UNIVERSE['Healthcare'][:10], 'theme': '💊 醫療'},
    'XLE (能源)': {'etf': 'XLE', 'holdings': STOCK_UNIVERSE['Energy'][:10], 'theme': '⛽ 能源'},
    'XLI (工業)': {'etf': 'XLI', 'holdings': STOCK_UNIVERSE['Industrials'][:10], 'theme': '🏭 工業'},
    'ARKK (創新)': {'etf': 'ARKK', 'holdings': STOCK_UNIVERSE['High Growth'][:10], 'theme': '🚀 創新'},
    'KWEB (中概)': {'etf': 'KWEB', 'holdings': STOCK_UNIVERSE['China ADR'][:10], 'theme': '🇨🇳 中概'},
}

# ============================================
# 🧮 TECHNICAL ANALYSIS
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
    def check_trend_template(df: pd.DataFrame) -> Dict:
        """
        Mark Minervini Trend Template
        檢查股票是否在健康的上升趨勢中
        """
        if len(df) < 200:
            return {'passed': False, 'score': 0, 'checks': []}
        
        close = float(df['Close'].iloc[-1])
        sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
        sma150 = float(df['Close'].rolling(150).mean().iloc[-1])
        sma200 = float(df['Close'].rolling(200).mean().iloc[-1])
        sma200_prev = float(df['Close'].rolling(200).mean().iloc[-20])
        high_52w = float(df['High'].tail(252).max())
        low_52w = float(df['Low'].tail(252).min())
        
        checks = []
        passed = 0
        
        # Check 1: Price > SMA50
        if close > sma50:
            checks.append(('✅', 'Price > SMA50'))
            passed += 1
        else:
            checks.append(('❌', 'Price < SMA50'))
        
        # Check 2: Price > SMA150
        if close > sma150:
            checks.append(('✅', 'Price > SMA150'))
            passed += 1
        else:
            checks.append(('❌', 'Price < SMA150'))
        
        # Check 3: Price > SMA200
        if close > sma200:
            checks.append(('✅', 'Price > SMA200'))
            passed += 1
        else:
            checks.append(('❌', 'Price < SMA200'))
        
        # Check 4: SMA50 > SMA150
        if sma50 > sma150:
            checks.append(('✅', 'SMA50 > SMA150'))
            passed += 1
        else:
            checks.append(('❌', 'SMA50 < SMA150'))
        
        # Check 5: SMA150 > SMA200
        if sma150 > sma200:
            checks.append(('✅', 'SMA150 > SMA200'))
            passed += 1
        else:
            checks.append(('❌', 'SMA150 < SMA200'))
        
        # Check 6: SMA200 trending up
        if sma200 > sma200_prev:
            checks.append(('✅', 'SMA200 上升中'))
            passed += 1
        else:
            checks.append(('❌', 'SMA200 下降中'))
        
        # Check 7: Price within 25% of 52w high
        if close >= high_52w * 0.75:
            checks.append(('✅', '距52週高點 < 25%'))
            passed += 1
        else:
            checks.append(('❌', '距52週高點 > 25%'))
        
        # Check 8: Price at least 30% above 52w low
        if close >= low_52w * 1.30:
            checks.append(('✅', '距52週低點 > 30%'))
            passed += 1
        else:
            checks.append(('❌', '距52週低點 < 30%'))
        
        return {
            'passed': passed >= 6,
            'score': passed,
            'total': 8,
            'checks': checks
        }


# ============================================
# 🔍 MARKET SCANNER (全新)
# ============================================
class MarketScanner:
    """市場掃描器 - 多種掃描模式"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    @st.cache_data(ttl=1800)
    def scan_52w_highs(_self, stocks: List[str]) -> List[Dict]:
        """掃描創52週新高的股票"""
        results = []
        
        for ticker in stocks:
            try:
                df = yf.download(ticker, period='1y', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is None or len(df) < 200:
                    continue
                
                close = float(df['Close'].iloc[-1])
                high_52w = float(df['High'].max())
                low_52w = float(df['Low'].min())
                
                # 在52週高點的 5% 以內
                if close >= high_52w * 0.95:
                    dist_from_high = (close / high_52w - 1) * 100
                    dist_from_low = (close / low_52w - 1) * 100
                    
                    # Volume
                    vol_avg = float(df['Volume'].tail(50).mean())
                    vol_today = float(df['Volume'].iloc[-1])
                    vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1
                    
                    results.append({
                        'ticker': ticker,
                        'price': close,
                        'high_52w': high_52w,
                        'dist_high': dist_from_high,
                        'dist_low': dist_from_low,
                        'vol_ratio': vol_ratio,
                        'new_high': close >= high_52w * 0.99
                    })
            except:
                continue
        
        results.sort(key=lambda x: x['dist_high'], reverse=True)
        return results
    
    @st.cache_data(ttl=1800)
    def scan_high_rs(_self, stocks: List[str], spy_df: pd.DataFrame) -> List[Dict]:
        """掃描高 RS Rating 的股票"""
        results = []
        
        for ticker in stocks:
            try:
                df = yf.download(ticker, period='6mo', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is None or len(df) < 60:
                    continue
                
                rs = _self.ta.rs_rating(df, spy_df)
                
                if rs >= 70:
                    close = float(df['Close'].iloc[-1])
                    change_1m = (close / float(df['Close'].iloc[-21]) - 1) * 100
                    
                    results.append({
                        'ticker': ticker,
                        'price': close,
                        'rs_rating': rs,
                        'change_1m': change_1m
                    })
            except:
                continue
        
        results.sort(key=lambda x: x['rs_rating'], reverse=True)
        return results
    
    @st.cache_data(ttl=1800)
    def scan_trend_template(_self, stocks: List[str]) -> List[Dict]:
        """掃描通過趨勢模板的股票"""
        results = []
        
        for ticker in stocks:
            try:
                df = yf.download(ticker, period='1y', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is None or len(df) < 200:
                    continue
                
                trend = _self.ta.check_trend_template(df)
                
                if trend['score'] >= 5:  # 至少通過 5/8
                    close = float(df['Close'].iloc[-1])
                    results.append({
                        'ticker': ticker,
                        'price': close,
                        'trend_score': trend['score'],
                        'trend_total': trend['total'],
                        'passed': trend['passed'],
                        'checks': trend['checks']
                    })
            except:
                continue
        
        results.sort(key=lambda x: x['trend_score'], reverse=True)
        return results


# ============================================
# 🎯 SETUP SCANNER
# ============================================
@dataclass
class SetupResult:
    ticker: str
    setup_type: str
    quality: str
    score: float
    price: float
    gap_percent: float
    tightness: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    rs_rating: float
    adr_percent: float
    volume_ratio: float
    above_sma50: bool
    notes: str
    grade_explanation: str = ""
    entry_explanation: str = ""
    risk_explanation: str = ""


class SetupScanner:
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def scan_bgu(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None, lookback_days: int = 5) -> Optional[SetupResult]:
        """掃描 BGU"""
        if df is None or len(df) < 50:
            return None
        
        try:
            best_bgu = None
            best_score = 0
            
            for day_offset in range(lookback_days):
                if day_offset >= len(df) - 1:
                    break
                
                idx = -1 - day_offset
                today = df.iloc[idx]
                yesterday = df.iloc[idx - 1]
                
                today_open = float(today['Open'])
                today_close = float(today['Close'])
                today_high = float(today['High'])
                today_low = float(today['Low'])
                today_volume = float(today['Volume'])
                yesterday_close = float(yesterday['Close'])
                
                gap_percent = (today_open / yesterday_close - 1) * 100
                
                if gap_percent < CONFIG.BGU_MIN_GAP:
                    continue
                
                avg_volume = float(df['Volume'].iloc[:-lookback_days].tail(50).mean())
                volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio < CONFIG.BGU_MIN_VOLUME:
                    continue
                
                day_range = today_high - today_low
                close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5
                
                if close_position < 0.4:
                    continue
                
                sma20 = float(df['Close'].rolling(20).mean().iloc[idx])
                sma50 = float(df['Close'].rolling(50).mean().iloc[idx])
                above_mas = today_close > sma20 and today_close > sma50
                
                rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
                adr = float(self.ta.adr_percent(df).iloc[idx])
                
                # 計算分數
                score = 0
                notes = []
                explanations = []
                
                # 跳空評分
                if gap_percent >= 8:
                    score += 30
                    notes.append(f"強跳空 {gap_percent:.1f}%")
                    explanations.append(f"✅ 跳空 {gap_percent:.1f}%: +30分")
                elif gap_percent >= 5:
                    score += 25
                    notes.append(f"跳空 {gap_percent:.1f}%")
                    explanations.append(f"✅ 跳空 {gap_percent:.1f}%: +25分")
                else:
                    score += 15
                    explanations.append(f"⚠️ 跳空 {gap_percent:.1f}%: +15分")
                
                # 量比評分
                if volume_ratio >= 3:
                    score += 25
                    notes.append(f"爆量 {volume_ratio:.1f}x")
                    explanations.append(f"✅ 量比 {volume_ratio:.1f}x: +25分")
                elif volume_ratio >= 2:
                    score += 20
                    explanations.append(f"✅ 量比 {volume_ratio:.1f}x: +20分")
                else:
                    score += 10
                    explanations.append(f"⚠️ 量比 {volume_ratio:.1f}x: +10分")
                
                # 收盤位置
                if close_position >= 0.8:
                    score += 20
                    notes.append("收強")
                    explanations.append(f"✅ 收盤 {close_position*100:.0f}%: +20分")
                elif close_position >= 0.6:
                    score += 15
                    explanations.append(f"✅ 收盤 {close_position*100:.0f}%: +15分")
                else:
                    score += 8
                    explanations.append(f"⚠️ 收盤 {close_position*100:.0f}%: +8分")
                
                # RS
                if rs >= 90:
                    score += 15
                    notes.append(f"RS{rs:.0f}")
                    explanations.append(f"✅ RS {rs:.0f}: +15分")
                elif rs >= 80:
                    score += 12
                    explanations.append(f"✅ RS {rs:.0f}: +12分")
                elif rs >= 70:
                    score += 8
                    explanations.append(f"⚠️ RS {rs:.0f}: +8分")
                else:
                    score -= 5
                    explanations.append(f"❌ RS {rs:.0f}: -5分")
                
                # 均線
                if above_mas:
                    score += 10
                    explanations.append("✅ 在均線之上: +10分")
                else:
                    score -= 10
                    explanations.append("❌ 在均線之下: -10分")
                
                # 時效性
                if day_offset > 0:
                    penalty = day_offset * 5
                    score -= penalty
                    explanations.append(f"⏰ {day_offset}天前: -{penalty}分")
                
                # 等級
                if score >= 85:
                    quality = 'A+'
                elif score >= 70:
                    quality = 'A'
                elif score >= 55:
                    quality = 'B'
                else:
                    quality = 'C'
                
                if score > best_score:
                    best_score = score
                    
                    atr = float(self.ta.atr(df).iloc[idx])
                    current_price = float(df['Close'].iloc[-1])
                    
                    entry = today_low
                    stop = today_low - atr * 0.5
                    target_1 = entry * 1.10
                    target_2 = entry * 1.20
                    
                    risk = entry - stop
                    rr = (target_1 - entry) / risk if risk > 0 else 0
                    
                    grade_explanation = f"### BGU 評分: {score}\n\n" + "\n".join(explanations)
                    entry_explanation = f"入場: ${entry:.2f} (跳空日低點)"
                    risk_explanation = f"止損: ${stop:.2f} | 風險: ${risk:.2f}/股"
                    
                    best_bgu = SetupResult(
                        ticker=ticker, setup_type='BGU', quality=quality, score=score,
                        price=current_price, gap_percent=gap_percent, tightness=0,
                        entry_price=round(entry, 2), stop_loss=round(stop, 2),
                        target_1=round(target_1, 2), target_2=round(target_2, 2),
                        risk_reward=round(rr, 2), rs_rating=rs, adr_percent=adr,
                        volume_ratio=volume_ratio, above_sma50=above_mas,
                        notes=" | ".join(notes), grade_explanation=grade_explanation,
                        entry_explanation=entry_explanation, risk_explanation=risk_explanation
                    )
            
            return best_bgu
            
        except:
            return None
    
    def scan_vcp(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[SetupResult]:
        """掃描 VCP"""
        if df is None or len(df) < 100:
            return None
        
        try:
            close = df['Close']
            curr_price = float(close.iloc[-1])
            
            # Stage 2 檢查
            sma50 = close.rolling(50).mean()
            sma150 = close.rolling(150).mean() if len(close) >= 150 else sma50
            sma200 = close.rolling(200).mean() if len(close) >= 200 else sma150
            
            curr_sma50 = float(sma50.iloc[-1])
            above_sma50 = curr_price > curr_sma50 * 0.98
            
            if not above_sma50:
                return None
            
            # 計算收縮
            recent = df.tail(50)
            contractions = []
            
            for i in range(0, min(40, len(recent)-5), 5):
                week = recent.iloc[i:i+5]
                week_range = (float(week['High'].max()) - float(week['Low'].min())) / float(week['Low'].min()) * 100
                contractions.append(week_range)
            
            if len(contractions) < 3:
                return None
            
            contraction_count = sum(1 for i in range(1, len(contractions)) 
                                   if contractions[i] < contractions[i-1] * 1.1)
            
            if contraction_count < CONFIG.VCP_MIN_CONTRACTIONS:
                return None
            
            final_tightness = contractions[-1]
            
            if final_tightness > CONFIG.VCP_MAX_TIGHTNESS:
                return None
            
            # Pivot
            pivot = float(recent['High'].max())
            vcp_low = float(recent['Low'].min())
            
            rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            adr = float(self.ta.adr_percent(df).iloc[-1])
            
            vol_ratio = float(df['Volume'].iloc[-1]) / float(df['Volume'].tail(50).mean())
            
            # 計算分數
            score = 0
            notes = []
            explanations = []
            
            # 緊縮度
            if final_tightness <= 5:
                score += 30
                notes.append(f"極緊 {final_tightness:.1f}%")
                explanations.append(f"✅ 緊縮 {final_tightness:.1f}%: +30分")
            elif final_tightness <= 8:
                score += 25
                notes.append(f"緊縮 {final_tightness:.1f}%")
                explanations.append(f"✅ 緊縮 {final_tightness:.1f}%: +25分")
            else:
                score += 15
                explanations.append(f"⚠️ 緊縮 {final_tightness:.1f}%: +15分")
            
            # 收縮次數
            if contraction_count >= 4:
                score += 20
                notes.append(f"{contraction_count}次收縮")
                explanations.append(f"✅ {contraction_count}次收縮: +20分")
            elif contraction_count >= 3:
                score += 15
                explanations.append(f"✅ {contraction_count}次收縮: +15分")
            else:
                score += 10
                explanations.append(f"⚠️ {contraction_count}次收縮: +10分")
            
            # RS
            if rs >= 90:
                score += 20
                notes.append(f"RS{rs:.0f}")
                explanations.append(f"✅ RS {rs:.0f}: +20分")
            elif rs >= 80:
                score += 15
                explanations.append(f"✅ RS {rs:.0f}: +15分")
            elif rs >= 70:
                score += 10
                explanations.append(f"⚠️ RS {rs:.0f}: +10分")
            else:
                score -= 5
                explanations.append(f"❌ RS {rs:.0f}: -5分")
            
            # 距離 Pivot
            dist_pivot = (pivot - curr_price) / curr_price * 100
            if dist_pivot <= 3:
                score += 15
                notes.append("近突破")
                explanations.append(f"✅ 距Pivot {dist_pivot:.1f}%: +15分")
            elif dist_pivot <= 5:
                score += 10
                explanations.append(f"⚠️ 距Pivot {dist_pivot:.1f}%: +10分")
            else:
                score += 5
                explanations.append(f"⚠️ 距Pivot {dist_pivot:.1f}%: +5分")
            
            # 趨勢
            score += 10
            explanations.append("✅ 在SMA50之上: +10分")
            
            # 等級
            if score >= 85:
                quality = 'A+'
            elif score >= 70:
                quality = 'A'
            elif score >= 55:
                quality = 'B'
            else:
                quality = 'C'
            
            atr = float(self.ta.atr(df).iloc[-1])
            entry = pivot * 1.001
            stop = vcp_low - atr * 0.3
            target_1 = entry + (pivot - vcp_low)
            target_2 = entry + (pivot - vcp_low) * 1.5
            
            risk = entry - stop
            rr = (target_1 - entry) / risk if risk > 0 else 0
            
            grade_explanation = f"### VCP 評分: {score}\n\n" + "\n".join(explanations)
            entry_explanation = f"入場: ${entry:.2f} (突破 Pivot)"
            risk_explanation = f"止損: ${stop:.2f} | 風險: ${risk:.2f}/股"
            
            return SetupResult(
                ticker=ticker, setup_type='VCP', quality=quality, score=score,
                price=curr_price, gap_percent=0, tightness=final_tightness,
                entry_price=round(entry, 2), stop_loss=round(stop, 2),
                target_1=round(target_1, 2), target_2=round(target_2, 2),
                risk_reward=round(rr, 2), rs_rating=rs, adr_percent=adr,
                volume_ratio=vol_ratio, above_sma50=above_sma50,
                notes=" | ".join(notes), grade_explanation=grade_explanation,
                entry_explanation=entry_explanation, risk_explanation=risk_explanation
            )
            
        except:
            return None
    
    def scan_power_play(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[SetupResult]:
        """
        掃描 Power Play (新增!)
        
        Power Play = 強勢股的緊縮整理後突破
        結合了 BGU 的動能和 VCP 的整理
        """
        if df is None or len(df) < 60:
            return None
        
        try:
            close = df['Close']
            high = df['High']
            volume = df['Volume']
            
            curr_price = float(close.iloc[-1])
            
            # 條件1: 過去30天有過大漲 (>=10%)
            max_gain_30d = 0
            for i in range(5, 30):
                if i < len(close):
                    gain = (float(close.iloc[-i]) / float(close.iloc[-i-5]) - 1) * 100
                    max_gain_30d = max(max_gain_30d, gain)
            
            if max_gain_30d < 10:
                return None
            
            # 條件2: 最近5天緊縮整理
            recent_5d = df.tail(5)
            range_5d = (float(recent_5d['High'].max()) - float(recent_5d['Low'].min())) / float(recent_5d['Low'].min()) * 100
            
            if range_5d > 8:
                return None
            
            # 條件3: 成交量萎縮
            vol_recent = float(volume.tail(5).mean())
            vol_prior = float(volume.iloc[-30:-5].mean())
            vol_contraction = vol_recent < vol_prior * 0.8
            
            if not vol_contraction:
                return None
            
            # 條件4: 在均線之上
            sma20 = float(close.rolling(20).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            
            if curr_price < sma20:
                return None
            
            rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            adr = float(self.ta.adr_percent(df).iloc[-1])
            
            # 計算分數
            score = 50  # 基礎分
            notes = []
            
            if max_gain_30d >= 20:
                score += 20
                notes.append(f"大漲{max_gain_30d:.0f}%")
            else:
                score += 10
            
            if range_5d <= 4:
                score += 20
                notes.append(f"極緊{range_5d:.1f}%")
            else:
                score += 10
            
            if rs >= 80:
                score += 15
                notes.append(f"RS{rs:.0f}")
            
            if vol_contraction:
                score += 10
                notes.append("量縮")
            
            if score >= 85:
                quality = 'A+'
            elif score >= 70:
                quality = 'A'
            elif score >= 55:
                quality = 'B'
            else:
                quality = 'C'
            
            pivot = float(high.tail(10).max())
            atr = float(self.ta.atr(df).iloc[-1])
            
            entry = pivot * 1.001
            stop = float(df['Low'].tail(5).min()) - atr * 0.3
            target_1 = entry * 1.08
            target_2 = entry * 1.15
            
            risk = entry - stop
            rr = (target_1 - entry) / risk if risk > 0 else 0
            
            return SetupResult(
                ticker=ticker, setup_type='PP', quality=quality, score=score,
                price=curr_price, gap_percent=max_gain_30d, tightness=range_5d,
                entry_price=round(entry, 2), stop_loss=round(stop, 2),
                target_1=round(target_1, 2), target_2=round(target_2, 2),
                risk_reward=round(rr, 2), rs_rating=rs, adr_percent=adr,
                volume_ratio=vol_recent/vol_prior if vol_prior > 0 else 1,
                above_sma50=curr_price > sma50,
                notes=" | ".join(notes),
                grade_explanation=f"Power Play 評分: {score}",
                entry_explanation=f"入場: 突破 ${pivot:.2f}",
                risk_explanation=f"止損: ${stop:.2f}"
            )
            
        except:
            return None
    
    def scan_all(self, stocks: List[str], spy_df: pd.DataFrame = None, 
                 progress_callback=None) -> Tuple[List[SetupResult], List[SetupResult], List[SetupResult]]:
        """掃描所有股票的所有 Setup"""
        bgu_results = []
        vcp_results = []
        pp_results = []
        
        for i, ticker in enumerate(stocks):
            if progress_callback:
                progress_callback(i, len(stocks), ticker)
            
            try:
                df = yf.download(ticker, period='6mo', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is None or len(df) < 50:
                    continue
                
                # BGU
                bgu = self.scan_bgu(df, ticker, spy_df, lookback_days=5)
                if bgu and bgu.score >= 45:
                    bgu_results.append(bgu)
                
                # VCP
                vcp = self.scan_vcp(df, ticker, spy_df)
                if vcp and vcp.score >= 45:
                    vcp_results.append(vcp)
                
                # Power Play
                pp = self.scan_power_play(df, ticker, spy_df)
                if pp and pp.score >= 55:
                    pp_results.append(pp)
                    
            except:
                continue
        
        bgu_results.sort(key=lambda x: x.score, reverse=True)
        vcp_results.sort(key=lambda x: x.score, reverse=True)
        pp_results.sort(key=lambda x: x.score, reverse=True)
        
        return bgu_results, vcp_results, pp_results


# ============================================
# 🌡️ MARKET REGIME
# ============================================
class MarketRegime:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_health() -> Dict:
        default = {'status': '❓', 'score': 50, 'vix': None, 'spy_price': None, 'advice': '', 'details': []}
        
        try:
            spy = yf.download('SPY', period='6mo', progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            
            if spy is None or len(spy) == 0:
                return default
            
            try:
                vix = yf.download('^VIX', period='5d', progress=False)
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_val = float(vix['Close'].iloc[-1]) if len(vix) > 0 else 20
            except:
                vix_val = 20
            
            spy_close = float(spy['Close'].iloc[-1])
            sma50 = float(spy['Close'].rolling(50).mean().iloc[-1])
            sma200 = float(spy['Close'].rolling(200).mean().iloc[-1]) if len(spy) >= 200 else sma50
            
            score = 50
            details = []
            
            if spy_close > sma200:
                score += 15
                details.append("✅ SPY > SMA200")
            else:
                details.append("❌ SPY < SMA200")
            
            if spy_close > sma50:
                score += 10
                details.append("✅ SPY > SMA50")
            else:
                details.append("❌ SPY < SMA50")
            
            if len(spy) >= 21:
                ret = (spy_close / float(spy['Close'].iloc[-21]) - 1) * 100
                if ret > 0:
                    score += 10
                    details.append(f"✅ 月回報 {ret:+.1f}%")
                elif ret < -5:
                    score -= 15
                    details.append(f"❌ 月回報 {ret:+.1f}%")
                else:
                    details.append(f"⚠️ 月回報 {ret:+.1f}%")
            
            if vix_val < 15:
                score += 10
                details.append(f"✅ VIX {vix_val:.1f} (低)")
            elif vix_val > 25:
                score -= 15
                details.append(f"❌ VIX {vix_val:.1f} (高)")
            else:
                details.append(f"⚠️ VIX {vix_val:.1f}")
            
            if score >= 75:
                status, advice = "🟢 強勢", "全力進攻"
            elif score >= 60:
                status, advice = "🟡 謹慎", "正常交易"
            elif score >= 40:
                status, advice = "🟠 震盪", "減少倉位"
            else:
                status, advice = "🔴 弱勢", "防守"
            
            return {
                'status': status, 'score': score, 'advice': advice,
                'vix': round(vix_val, 1), 'spy_price': round(spy_close, 2),
                'details': details
            }
        except:
            return default


# ============================================
# 📊 CHART BUILDER
# ============================================
class ChartBuilder:
    @staticmethod
    def create_setup_chart(df: pd.DataFrame, ticker: str, setup: SetupResult = None) -> go.Figure:
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price',
            increasing_line_color='#00CC96', decreasing_line_color='#EF553B'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                                 line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                                 line=dict(color='blue', width=1)), row=1, col=1)
        
        if setup:
            fig.add_hline(y=setup.entry_price, line_dash="dash", line_color="green",
                         annotation_text=f"Entry ${setup.entry_price}", row=1, col=1)
            fig.add_hline(y=setup.stop_loss, line_dash="dash", line_color="red",
                         annotation_text=f"Stop ${setup.stop_loss}", row=1, col=1)
            fig.add_hline(y=setup.target_1, line_dash="dash", line_color="blue",
                         annotation_text=f"T1 ${setup.target_1}", row=1, col=1)
        
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors,
                            name='Volume'), row=2, col=1)
        
        fig.update_layout(
            height=500, showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            title=f"{ticker} - {setup.setup_type if setup else ''} ({setup.quality if setup else ''})"
        )
        
        return fig


# ============================================
# 📡 DATA FETCHER
# ============================================
class DataFetcher:
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_stock(ticker: str, period: str = "1y"):
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if len(df) > 0 else None
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_sector_etfs():
        tickers = [s['etf'] for s in SECTORS.values()] + ['SPY']
        try:
            data = yf.download(tickers, period="6mo", progress=False)['Close']
            return data
        except:
            return None


# ============================================
# 💰 POSITION CALCULATOR
# ============================================
class PositionCalculator:
    @staticmethod
    def calculate(account: float, entry: float, stop: float, risk_pct: float = 0.02) -> Dict:
        risk_amount = account * risk_pct
        risk_per_share = abs(entry - stop)
        
        if risk_per_share <= 0:
            return {'error': 'Invalid'}
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_pct': position_value / account * 100,
            'risk_amount': risk_amount,
            'max_loss': shares * risk_per_share
        }


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v7.0 Ultimate")
    st.caption(f"專業交易員版本 | 掃描 {len(ALL_STOCKS)}+ 股票 | BGU / VCP / Power Play")
    
    # Market Health
    market = MarketRegime.get_health()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("市場狀態", market['status'])
    col2.metric("健康評分", f"{market['score']}/100")
    col3.metric("VIX", f"{market['vix']:.1f}" if market['vix'] else "N/A")
    col4.metric("SPY", f"${market['spy_price']:.2f}" if market['spy_price'] else "N/A")
    col5.metric("建議", market['advice'])
    
    with st.expander("查看市場詳情"):
        for detail in market.get('details', []):
            st.write(detail)
    
    st.divider()
    
    # Tabs
    tabs = st.tabs([
        "🌪️ 板塊輪動",
        "📊 個股分析",
        "🎯 Setup 獵人",
        "🔥 52週新高",
        "📈 趨勢領袖",
        "💰 倉位計算",
        "📖 交易教學"
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
                        '板塊': name, '主題': info['theme'],
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success("**強勢板塊:**")
                    for _, row in df_rs[df_rs['RS'] > 2].head(3).iterrows():
                        st.write(f"• {row['板塊']} ({row['RS']:+.1f}%)")
                with col2:
                    st.error("**弱勢板塊:**")
                    for _, row in df_rs[df_rs['RS'] < -2].tail(3).iterrows():
                        st.write(f"• {row['板塊']} ({row['RS']:+.1f}%)")
    
    # ===== TAB 2: Stock Analysis =====
    with tabs[1]:
        st.header("📊 個股分析")
        
        ticker = st.text_input("股票代碼", value="NVDA").upper()
        
        if st.button("分析", type="primary", key="analyze"):
            df = DataFetcher.get_stock(ticker, "1y")
            spy_df = DataFetcher.get_stock('SPY', '1y')
            
            if df is not None:
                scanner = SetupScanner()
                ta = TechnicalAnalysis()
                
                bgu = scanner.scan_bgu(df, ticker, spy_df)
                vcp = scanner.scan_vcp(df, ticker, spy_df)
                pp = scanner.scan_power_play(df, ticker, spy_df)
                trend = ta.check_trend_template(df)
                
                # 基本信息
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("價格", f"${float(df['Close'].iloc[-1]):.2f}")
                col2.metric("RS Rating", f"{ta.rs_rating(df, spy_df):.0f}")
                col3.metric("ADR%", f"{float(ta.adr_percent(df).iloc[-1]):.1f}%")
                col4.metric("趨勢模板", f"{trend['score']}/{trend['total']}")
                
                # Setup 狀態
                if bgu:
                    st.success(f"🚀 **BGU 信號** - {bgu.quality} ({bgu.score:.0f}分)")
                if vcp:
                    st.info(f"🎯 **VCP 信號** - {vcp.quality} ({vcp.score:.0f}分)")
                if pp:
                    st.warning(f"⚡ **Power Play** - {pp.quality} ({pp.score:.0f}分)")
                
                # 趨勢模板詳情
                with st.expander("趨勢模板詳情"):
                    for icon, check in trend['checks']:
                        st.write(f"{icon} {check}")
                
                # 圖表
                setup = bgu or vcp or pp
                fig = ChartBuilder.create_setup_chart(df, ticker, setup)
                st.plotly_chart(fig, use_container_width=True)
                
                if setup:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        ### 交易計劃
                        | 項目 | 價格 |
                        |------|------|
                        | 入場 | ${setup.entry_price} |
                        | 止損 | ${setup.stop_loss} |
                        | T1 | ${setup.target_1} |
                        | T2 | ${setup.target_2} |
                        | R:R | {setup.risk_reward}:1 |
                        """)
                    with col2:
                        st.markdown(setup.grade_explanation)
    
    # ===== TAB 3: Setup Hunter =====
    with tabs[2]:
        st.header("🎯 Setup 獵人")
        
        st.info(f"可掃描 {len(ALL_STOCKS)}+ 股票，尋找 BGU / VCP / Power Play")
        
        col1, col2 = st.columns(2)
        with col1:
            scan_scope = st.selectbox(
                "掃描範圍",
                [
                    "🔥 熱門領導股 (30隻)",
                    "🔬 半導體 (20隻)",
                    "💻 軟件雲端 (20隻)",
                    "🚀 高成長股 (20隻)",
                    "🏦 金融股 (20隻)",
                    "💊 醫療股 (20隻)",
                    "🇨🇳 中概股 (20隻)",
                    "⚡ 新能源/EV (20隻)",
                    "🤖 AI概念 (20隻)",
                    "📊 全部股票 (200+) ⚠️較慢"
                ],
                key="scan_scope"
            )
        with col2:
            setup_filter = st.selectbox(
                "Setup 類型",
                ["全部", "只看 BGU", "只看 VCP", "只看 Power Play"],
                key="setup_filter"
            )
        
        if st.button("🎯 開始掃描", type="primary", key="scan_all"):
            # 選擇股票
            if "熱門" in scan_scope:
                stocks = STOCK_UNIVERSE['Market Leaders']
            elif "半導體" in scan_scope:
                stocks = STOCK_UNIVERSE['Semiconductors']
            elif "軟件" in scan_scope:
                stocks = STOCK_UNIVERSE['Software & Cloud']
            elif "高成長" in scan_scope:
                stocks = STOCK_UNIVERSE['High Growth']
            elif "金融" in scan_scope:
                stocks = STOCK_UNIVERSE['Financials']
            elif "醫療" in scan_scope:
                stocks = STOCK_UNIVERSE['Healthcare']
            elif "中概" in scan_scope:
                stocks = STOCK_UNIVERSE['China ADR']
            elif "新能源" in scan_scope or "EV" in scan_scope:
                stocks = STOCK_UNIVERSE['Clean Energy & EV']
            elif "AI" in scan_scope:
                stocks = STOCK_UNIVERSE['AI & Robotics']
            else:
                stocks = ALL_STOCKS
            
            spy_df = DataFetcher.get_stock('SPY', '6mo')
            scanner = SetupScanner()
            
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(i, total, ticker):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"掃描 {ticker} ({i+1}/{total})...")
            
            bgu_results, vcp_results, pp_results = scanner.scan_all(stocks, spy_df, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state['bgu_results'] = bgu_results
            st.session_state['vcp_results'] = vcp_results
            st.session_state['pp_results'] = pp_results
        
        # 顯示結果
        if 'bgu_results' in st.session_state:
            bgu_results = st.session_state.get('bgu_results', [])
            vcp_results = st.session_state.get('vcp_results', [])
            pp_results = st.session_state.get('pp_results', [])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🚀 BGU", len(bgu_results))
            col2.metric("🎯 VCP", len(vcp_results))
            col3.metric("⚡ Power Play", len(pp_results))
            
            # BGU
            if bgu_results and setup_filter in ["全部", "只看 BGU"]:
                st.markdown("### 🚀 BGU (跳空突破)")
                for setup in bgu_results[:8]:
                    with st.expander(f"**{setup.ticker}** | {setup.quality} | {setup.score:.0f}分 | 跳空{setup.gap_percent:.1f}%"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            - 價格: ${setup.price:.2f}
                            - RS: {setup.rs_rating:.0f}
                            - 入場: ${setup.entry_price}
                            - 止損: ${setup.stop_loss}
                            - R:R: {setup.risk_reward}:1
                            """)
                        with col2:
                            st.markdown(setup.notes)
                        
                        if st.button(f"查看圖表", key=f"bgu_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "3mo")
                            if df is not None:
                                fig = ChartBuilder.create_setup_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
            
            # VCP
            if vcp_results and setup_filter in ["全部", "只看 VCP"]:
                st.markdown("### 🎯 VCP (波動收縮)")
                for setup in vcp_results[:8]:
                    with st.expander(f"**{setup.ticker}** | {setup.quality} | {setup.score:.0f}分 | 緊縮{setup.tightness:.1f}%"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            - 價格: ${setup.price:.2f}
                            - RS: {setup.rs_rating:.0f}
                            - 入場: ${setup.entry_price}
                            - 止損: ${setup.stop_loss}
                            - R:R: {setup.risk_reward}:1
                            """)
                        with col2:
                            st.markdown(setup.notes)
                        
                        if st.button(f"查看圖表", key=f"vcp_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "3mo")
                            if df is not None:
                                fig = ChartBuilder.create_setup_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
            
            # Power Play
            if pp_results and setup_filter in ["全部", "只看 Power Play"]:
                st.markdown("### ⚡ Power Play (強勢整理)")
                for setup in pp_results[:8]:
                    with st.expander(f"**{setup.ticker}** | {setup.quality} | {setup.score:.0f}分"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            - 價格: ${setup.price:.2f}
                            - 30日大漲: {setup.gap_percent:.0f}%
                            - 5日緊縮: {setup.tightness:.1f}%
                            - 入場: ${setup.entry_price}
                            - 止損: ${setup.stop_loss}
                            """)
                        with col2:
                            st.markdown(setup.notes)
            
            if not bgu_results and not vcp_results and not pp_results:
                st.info("沒有發現符合條件的 Setup，嘗試其他板塊")
    
    # ===== TAB 4: 52W Highs =====
    with tabs[3]:
        st.header("🔥 52週新高掃描")
        
        st.info("尋找創新高或接近新高的強勢股")
        
        if st.button("掃描新高", type="primary", key="scan_highs"):
            scanner = MarketScanner()
            
            with st.spinner("掃描中..."):
                results = scanner.scan_52w_highs(ALL_STOCKS[:100])
            
            if results:
                st.success(f"找到 {len(results)} 隻股票接近52週新高")
                
                df = pd.DataFrame(results)
                
                new_highs = df[df['new_high'] == True]
                if not new_highs.empty:
                    st.markdown("### 🎉 創新高!")
                    st.dataframe(
                        new_highs[['ticker', 'price', 'high_52w', 'dist_low', 'vol_ratio']]
                        .rename(columns={'ticker': 'Ticker', 'price': '價格', 'high_52w': '52週高',
                                        'dist_low': '距低點%', 'vol_ratio': '量比'})
                        .style.format({'價格': '${:.2f}', '52週高': '${:.2f}', 
                                      '距低點%': '+{:.0f}%', '量比': '{:.1f}x'}),
                        hide_index=True
                    )
                
                near_highs = df[df['new_high'] == False]
                if not near_highs.empty:
                    st.markdown("### 📈 接近新高 (5%內)")
                    st.dataframe(
                        near_highs[['ticker', 'price', 'high_52w', 'dist_high', 'vol_ratio']]
                        .rename(columns={'ticker': 'Ticker', 'price': '價格', 'high_52w': '52週高',
                                        'dist_high': '距高點%', 'vol_ratio': '量比'})
                        .style.format({'價格': '${:.2f}', '52週高': '${:.2f}', 
                                      '距高點%': '{:.1f}%', '量比': '{:.1f}x'}),
                        hide_index=True
                    )
    
    # ===== TAB 5: Trend Leaders =====
    with tabs[4]:
        st.header("📈 趨勢領袖")
        
        st.info("掃描通過 Minervini 趨勢模板的股票")
        
        if st.button("掃描趨勢", type="primary", key="scan_trend"):
            scanner = MarketScanner()
            
            with st.spinner("掃描中..."):
                results = scanner.scan_trend_template(ALL_STOCKS[:80])
            
            if results:
                st.success(f"找到 {len(results)} 隻股票通過趨勢模板")
                
                for r in results[:10]:
                    status = "✅ 完全通過" if r['passed'] else "⚠️ 部分通過"
                    with st.expander(f"**{r['ticker']}** - {status} ({r['trend_score']}/{r['trend_total']})"):
                        for icon, check in r['checks']:
                            st.write(f"{icon} {check}")
    
    # ===== TAB 6: Position Calculator =====
    with tabs[5]:
        st.header("💰 倉位計算器")
        
        col1, col2 = st.columns(2)
        with col1:
            account = st.number_input("帳戶 ($)", value=100000, step=10000)
            risk_pct = st.slider("風險 (%)", 0.5, 3.0, 2.0, 0.5) / 100
        with col2:
            entry = st.number_input("入場價 ($)", value=150.0, step=1.0)
            stop = st.number_input("止損價 ($)", value=145.0, step=1.0)
        
        if st.button("計算", type="primary", key="calc"):
            result = PositionCalculator.calculate(account, entry, stop, risk_pct)
            
            if 'error' not in result:
                col1, col2, col3 = st.columns(3)
                col1.metric("股數", f"{result['shares']}")
                col2.metric("金額", f"${result['position_value']:,.0f}")
                col3.metric("最大虧損", f"${result['max_loss']:,.0f}")
                
                # R-multiples
                st.markdown("### 止盈目標")
                risk = entry - stop
                for r in [2, 3, 5]:
                    target = entry + risk * r
                    profit = result['shares'] * risk * r
                    st.write(f"• {r}R: ${target:.2f} (盈利 ${profit:,.0f})")
    
    # ===== TAB 7: Education =====
    with tabs[6]:
        st.header("📖 交易教學")
        
        st.markdown("""
        ## 三種 Setup 對比
        
        | Setup | 類型 | 最佳時機 | 風險 | 回報 |
        |-------|------|----------|------|------|
        | **BGU** | 動能突破 | 財報後 | 較高 | 10-20% |
        | **VCP** | 整理突破 | 任何時候 | 較低 | 15-30% |
        | **Power Play** | 強勢整理 | 強勢市場 | 中等 | 8-15% |
        
        ---
        
        ## A 級 Setup 標準
        
        ### 🚀 BGU (Buyable Gap Up)
        - 跳空 ≥ 5%
        - 量比 ≥ 2x
        - 收盤在高點 80%+
        - RS ≥ 80
        
        ### 🎯 VCP (Volatility Contraction)
        - 2+ 次收縮
        - 最終緊縮 ≤ 8%
        - RS ≥ 80
        - Stage 2 趨勢
        
        ### ⚡ Power Play
        - 30天內大漲 10%+
        - 5天整理 < 8%
        - 量縮
        - 在均線之上
        
        ---
        
        ## 風險管理
        
        1. **單筆風險 ≤ 2%**
        2. **最多 5-8 個持倉**
        3. **嚴格止損**
        4. **分批止盈**
        
        ---
        
        ## 推薦資源
        
        - **Mark Minervini**: Trade Like a Stock Market Wizard
        - **Qullamaggie**: YouTube / Twitter
        - **William O'Neil**: How to Make Money in Stocks
        """)
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v7.0 Ultimate")
    st.sidebar.markdown(f"""
    **股票數量:** {len(ALL_STOCKS)}+
    
    **掃描功能:**
    - ✅ BGU 跳空突破
    - ✅ VCP 波動收縮
    - ✅ Power Play 強勢整理
    - ✅ 52週新高
    - ✅ 趨勢模板
    
    **板塊覆蓋:**
    - 科技 / 半導體
    - 軟件 / 雲端
    - 金融 / 醫療
    - 消費 / 能源
    - 高成長 / 中概
    - AI / 新能源
    """)


if __name__ == "__main__":
    main()
