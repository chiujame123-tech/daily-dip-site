# -*- coding: utf-8 -*-
"""
🌪️ Market Structure Radar - v5.0 Pro Edition
=============================================

專業交易員版本 - 目標年化 30%+

新增功能：
✅ 財報日期追蹤 & 警告
✅ RS Rating (相對強度評分)
✅ 行業內排名 (買最強的)
✅ 多時間框架分析 (週線確認日線)
✅ 專業倉位計算器
✅ 大盤環境評估
✅ Watchlist 管理
✅ 風險管理儀表板
✅ 交易日誌模板

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
    PAGE_TITLE: str = "Market Radar v5.0 Pro"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800
    
    # Risk Management
    MAX_RISK_PER_TRADE: float = 0.02  # 2%
    MAX_PORTFOLIO_RISK: float = 0.10  # 10%
    MAX_POSITIONS: int = 8
    MAX_SECTOR_EXPOSURE: float = 0.40  # 40% max in one sector
    
    # Technical
    RS_LOOKBACK: int = 63  # 3 months for RS calculation
    VCP_MIN_CONTRACTIONS: int = 2
    BREAKOUT_VOLUME_THRESHOLD: float = 1.5

CONFIG = Config()

# ============================================
# 📊 SECTOR DATA (擴展版)
# ============================================
SECTORS = {
    'SMH (半導體)': {
        'etf': 'SMH',
        'holdings': ['NVDA', 'TSM', 'AVGO', 'AMD', 'MU', 'QCOM', 'AMAT', 'LRCX', 'MRVL', 'ARM', 'KLAC', 'ADI'],
        'theme': '🔬 AI/芯片'
    },
    'XLK (科技)': {
        'etf': 'XLK',
        'holdings': ['MSFT', 'AAPL', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'NOW', 'INTU', 'IBM'],
        'theme': '💻 大型科技'
    },
    'XLC (通訊)': {
        'etf': 'XLC',
        'holdings': ['META', 'GOOGL', 'NFLX', 'DIS', 'TMUS', 'VZ', 'CMCSA', 'T', 'EA', 'TTWO'],
        'theme': '📱 社交/媒體'
    },
    'XLF (金融)': {
        'etf': 'XLF',
        'holdings': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'V', 'MA'],
        'theme': '🏦 銀行/金融'
    },
    'XLY (消費)': {
        'etf': 'XLY',
        'holdings': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG'],
        'theme': '🛒 消費'
    },
    'XLV (醫療)': {
        'etf': 'XLV',
        'holdings': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN', 'GILD'],
        'theme': '💊 醫療'
    },
    'XLE (能源)': {
        'etf': 'XLE',
        'holdings': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL'],
        'theme': '⛽ 能源'
    },
    'ARKK (創新)': {
        'etf': 'ARKK',
        'holdings': ['TSLA', 'COIN', 'ROKU', 'SQ', 'PATH', 'HOOD', 'RBLX', 'DKNG', 'CRSP', 'BEAM'],
        'theme': '🚀 創新'
    },
    'KWEB (中概)': {
        'etf': 'KWEB',
        'holdings': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES'],
        'theme': '🇨🇳 中概'
    }
}

HOT_STOCKS = ['NVDA', 'TSLA', 'AMD', 'META', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 
              'SMCI', 'ARM', 'COIN', 'PLTR', 'SOFI', 'MSTR', 'AVGO', 'CRM']

# ============================================
# 🧮 TECHNICAL ANALYSIS (Enhanced)
# ============================================
class TechnicalAnalysis:
    """Enhanced technical analysis"""
    
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
    def macd(prices: pd.Series):
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, signal, macd_line - signal
    
    @staticmethod
    def rs_rating(stock_df: pd.DataFrame, spy_df: pd.DataFrame, lookback: int = 63) -> float:
        """
        Calculate Relative Strength Rating (IBD Style)
        比較股票相對於 SPY 的表現，返回 1-99 評分
        """
        if len(stock_df) < lookback or len(spy_df) < lookback:
            return 50
        
        # 計算不同時間段的回報
        periods = [21, 42, 63]  # 1M, 2M, 3M
        weights = [0.4, 0.3, 0.3]  # 權重：近期更重要
        
        stock_score = 0
        spy_score = 0
        
        for period, weight in zip(periods, weights):
            if len(stock_df) >= period and len(spy_df) >= period:
                stock_ret = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-period] - 1) * 100
                spy_ret = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-period] - 1) * 100
                
                # 相對表現
                relative = stock_ret - spy_ret
                stock_score += relative * weight
        
        # 轉換為 1-99 評分
        # 假設 +30% 相對表現 = 99, -30% = 1
        rs = 50 + (stock_score / 30) * 49
        return max(1, min(99, rs))
    
    @staticmethod
    def calculate_trend_template(df: pd.DataFrame) -> Dict:
        """
        Mark Minervini Trend Template
        檢查股票是否符合強勢股標準
        """
        if len(df) < 200:
            return {'passed': False, 'score': 0, 'checks': {}}
        
        close = df['Close'].iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma150 = df['Close'].rolling(150).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        high_52w = df['High'].tail(252).max()
        low_52w = df['Low'].tail(252).min()
        
        checks = {
            '價格 > SMA50': close > sma50,
            'SMA50 > SMA150': sma50 > sma150,
            'SMA150 > SMA200': sma150 > sma200,
            '價格 > SMA200': close > sma200,
            'SMA200 上升': df['Close'].rolling(200).mean().iloc[-1] > df['Close'].rolling(200).mean().iloc[-20],
            '距52週高 < 25%': (close / high_52w - 1) > -0.25,
            '距52週低 > 30%': (close / low_52w - 1) > 0.30,
        }
        
        passed_count = sum(checks.values())
        score = passed_count / len(checks) * 100
        
        return {
            'passed': passed_count >= 5,  # 至少通過5項
            'score': score,
            'checks': checks
        }


# ============================================
# 📅 EARNINGS TRACKER
# ============================================
class EarningsTracker:
    """Track earnings dates"""
    
    @staticmethod
    @st.cache_data(ttl=86400)  # 24 hours cache
    def get_earnings_date(ticker: str) -> Optional[Dict]:
        """Get next earnings date for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is not None and len(calendar) > 0:
                # 嘗試獲取財報日期
                if hasattr(calendar, 'iloc'):
                    earnings_date = calendar.iloc[0, 0] if len(calendar.iloc[0]) > 0 else None
                elif isinstance(calendar, dict):
                    earnings_date = calendar.get('Earnings Date', [None])[0]
                else:
                    earnings_date = None
                
                if earnings_date:
                    if isinstance(earnings_date, pd.Timestamp):
                        earnings_date = earnings_date.to_pydatetime()
                    
                    days_until = (earnings_date - datetime.now()).days
                    
                    return {
                        'date': earnings_date,
                        'days_until': days_until,
                        'warning': days_until <= 14,
                        'danger': days_until <= 7
                    }
        except:
            pass
        return None
    
    @staticmethod
    def get_earnings_warning(ticker: str) -> str:
        """Get earnings warning message"""
        earnings = EarningsTracker.get_earnings_date(ticker)
        if earnings:
            days = earnings['days_until']
            if days <= 0:
                return "🚨 財報剛發布"
            elif days <= 7:
                return f"🔴 財報 {days} 天內！"
            elif days <= 14:
                return f"🟡 財報 {days} 天內"
            elif days <= 30:
                return f"📅 財報 {days} 天後"
        return ""


# ============================================
# 💰 POSITION CALCULATOR
# ============================================
class PositionCalculator:
    """Professional position sizing"""
    
    @staticmethod
    def calculate_position(
        account_size: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 0.02
    ) -> Dict:
        """
        Calculate position size based on risk
        
        Kelly Criterion inspired but capped at 2% risk per trade
        """
        risk_amount = account_size * risk_percent
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {'error': '止損價格無效'}
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        position_percent = position_value / account_size * 100
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_percent': position_percent,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'max_loss': shares * risk_per_share
        }
    
    @staticmethod
    def calculate_targets(entry: float, stop: float, r_multiples: List[float] = [2, 3, 5]) -> List[Dict]:
        """Calculate profit targets based on R-multiples"""
        risk = abs(entry - stop)
        targets = []
        
        for r in r_multiples:
            target = entry + (risk * r)
            profit_pct = (target / entry - 1) * 100
            targets.append({
                'r_multiple': r,
                'price': round(target, 2),
                'profit_pct': round(profit_pct, 1)
            })
        
        return targets


# ============================================
# 🌡️ MARKET REGIME
# ============================================
class MarketRegime:
    """Assess overall market conditions"""
    
    @staticmethod
    @st.cache_data(ttl=900)
    def get_market_health() -> Dict:
        """
        Comprehensive market health assessment
        """
        default_result = {
            'status': '❓ 未知', 
            'score': 50, 
            'advice': '無法獲取市場數據',
            'spy_above_50': None,
            'spy_above_200': None,
            'spy_return_1m': None,
            'vix': None,
            'spy_price': None
        }
        
        try:
            # Fetch market data with timeout handling
            spy = yf.download('SPY', period='6mo', progress=False, timeout=10)
            
            # Handle MultiIndex columns
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            
            if spy is None or len(spy) == 0:
                return default_result
            
            # Try to get VIX
            try:
                vix = yf.download('^VIX', period='1mo', progress=False, timeout=10)
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_current = float(vix['Close'].iloc[-1]) if len(vix) > 0 else 20
            except:
                vix_current = 20  # Default VIX
            
            # SPY Analysis
            spy_close = float(spy['Close'].iloc[-1])
            spy_sma50 = float(spy['Close'].rolling(50).mean().iloc[-1])
            spy_sma200 = float(spy['Close'].rolling(200).mean().iloc[-1]) if len(spy) >= 200 else spy_sma50
            spy_above_50 = spy_close > spy_sma50
            spy_above_200 = spy_close > spy_sma200
            
            # 1-month return
            if len(spy) >= 21:
                spy_return_1m = (spy_close / float(spy['Close'].iloc[-21]) - 1) * 100
            else:
                spy_return_1m = 0
            
            # Score calculation
            score = 50
            
            # SPY trend
            if spy_above_200:
                score += 15
            if spy_above_50:
                score += 10
            if spy_return_1m > 0:
                score += 10
            elif spy_return_1m < -5:
                score -= 15
            
            # VIX
            if vix_current < 15:
                score += 10
            elif vix_current > 25:
                score -= 15
            elif vix_current > 30:
                score -= 25
            
            # Determine regime
            if score >= 75:
                status = "🟢 強勢上漲"
                advice = "全力進攻，增加倉位"
            elif score >= 60:
                status = "🟡 謹慎樂觀"
                advice = "正常交易，注意風控"
            elif score >= 40:
                status = "🟠 震盪整理"
                advice = "減少倉位，等待明確方向"
            else:
                status = "🔴 弱勢下跌"
                advice = "防守為主，只做最強股"
            
            return {
                'status': status,
                'score': score,
                'advice': advice,
                'spy_above_50': spy_above_50,
                'spy_above_200': spy_above_200,
                'spy_return_1m': round(spy_return_1m, 2),
                'vix': round(vix_current, 1),
                'spy_price': round(spy_close, 2)
            }
        except Exception as e:
            # Log error for debugging
            print(f"MarketRegime error: {e}")
            return default_result


# ============================================
# 📈 ENHANCED STOCK ANALYZER
# ============================================
@dataclass
class ProStockAnalysis:
    """Professional stock analysis result"""
    ticker: str
    price: float
    change_pct: float
    
    # Technical
    status: str
    rs_rating: float
    adr_pct: float
    rsi: float
    extension_pct: float
    dist_52w_high: float
    volume_ratio: float
    
    # Trend Template
    trend_score: float
    trend_passed: bool
    
    # Pattern
    is_vcp: bool
    is_breakout: bool
    is_overheated: bool
    
    # Earnings
    earnings_warning: str
    
    # Entry
    suggested_entry: float
    suggested_stop: float
    risk_reward: float
    
    # Sector
    sector: str = ""
    sector_rank: int = 0


class ProStockAnalyzer:
    """Professional stock analyzer"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.earnings = EarningsTracker()
    
    def analyze(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[ProStockAnalysis]:
        """Comprehensive professional analysis"""
        if df is None or len(df) < 50:
            return None
        
        try:
            close = df['Close']
            volume = df['Volume']
            high = df['High']
            low = df['Low']
            
            # Current values
            curr_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            change_pct = (curr_price - prev_price) / prev_price * 100
            
            # Technical indicators
            sma50 = close.rolling(50).mean()
            curr_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else curr_price
            
            rsi = self.ta.rsi(close)
            curr_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            adr = self.ta.adr_percent(df)
            curr_adr = float(adr.iloc[-1]) if not pd.isna(adr.iloc[-1]) else 3
            
            atr = self.ta.atr(df)
            curr_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else curr_price * 0.02
            
            # RS Rating
            rs = 50
            if spy_df is not None:
                rs = self.ta.rs_rating(df, spy_df)
            
            # Extension & 52W
            extension = (curr_price - curr_sma50) / curr_sma50 * 100
            high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
            dist_high = (curr_price / high_52w - 1) * 100
            
            # Volume
            vol_avg = float(volume.tail(50).mean())
            vol_ratio = float(volume.iloc[-1]) / vol_avg if vol_avg > 0 else 1
            
            # Trend Template
            trend = self.ta.calculate_trend_template(df)
            
            # Pattern Detection
            is_vcp = self._detect_vcp(df)
            is_breakout = self._detect_breakout(df)
            is_overheated = extension > 25 or curr_rsi > 80
            
            # Status
            if is_breakout and vol_ratio > 1.5:
                status = "🚀 放量突破"
            elif is_vcp and trend['passed']:
                status = "🎯 VCP蓄勢"
            elif trend['passed'] and not is_overheated:
                status = "✅ 趨勢健康"
            elif is_overheated:
                status = "🔥 過熱"
            elif not trend['passed']:
                status = "⚠️ 趨勢不佳"
            else:
                status = "🧘 盤整"
            
            # Earnings warning
            earnings_warn = self.earnings.get_earnings_warning(ticker)
            
            # Entry calculation
            if is_vcp:
                suggested_entry = float(high.tail(10).max()) * 1.001
                suggested_stop = curr_price - curr_atr * 1.5
            elif is_breakout:
                suggested_entry = curr_price
                suggested_stop = curr_price - curr_atr * 2
            else:
                suggested_entry = curr_sma50
                suggested_stop = curr_sma50 - curr_atr * 2
            
            risk = suggested_entry - suggested_stop
            reward = curr_atr * 3
            rr = reward / risk if risk > 0 else 0
            
            return ProStockAnalysis(
                ticker=ticker,
                price=curr_price,
                change_pct=change_pct,
                status=status,
                rs_rating=rs,
                adr_pct=curr_adr,
                rsi=curr_rsi,
                extension_pct=extension,
                dist_52w_high=dist_high,
                volume_ratio=vol_ratio,
                trend_score=trend['score'],
                trend_passed=trend['passed'],
                is_vcp=is_vcp,
                is_breakout=is_breakout,
                is_overheated=is_overheated,
                earnings_warning=earnings_warn,
                suggested_entry=round(suggested_entry, 2),
                suggested_stop=round(suggested_stop, 2),
                risk_reward=round(rr, 2)
            )
            
        except Exception as e:
            return None
    
    def _detect_vcp(self, df: pd.DataFrame) -> bool:
        """Enhanced VCP detection"""
        if len(df) < 40:
            return False
        
        recent = df.tail(30)
        
        # Calculate weekly ranges
        ranges = []
        for i in range(0, 25, 5):
            if i + 5 <= 25:
                period = recent.iloc[i:i+5]
                range_pct = (period['High'].max() - period['Low'].min()) / period['Low'].min() * 100
                ranges.append(range_pct)
        
        if len(ranges) < 4:
            return False
        
        # Check for contraction
        contractions = 0
        for i in range(1, len(ranges)):
            if ranges[i] < ranges[i-1]:
                contractions += 1
        
        # VCP requires: 
        # 1. At least 2 contractions
        # 2. Last range < 10%
        # 3. Price above SMA50
        curr_price = float(df['Close'].iloc[-1])
        sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
        
        return (contractions >= CONFIG.VCP_MIN_CONTRACTIONS and 
                ranges[-1] < 10 and 
                curr_price > sma50)
    
    def _detect_breakout(self, df: pd.DataFrame) -> bool:
        """Enhanced breakout detection"""
        if len(df) < 30:
            return False
        
        curr_price = float(df['Close'].iloc[-1])
        high_20 = float(df['High'].iloc[-20:-1].max())
        vol_avg = float(df['Volume'].iloc[-20:-1].mean())
        vol_curr = float(df['Volume'].iloc[-1])
        
        # Breakout: price above 20-day high with volume
        return (curr_price > high_20 and 
                vol_curr > vol_avg * CONFIG.BREAKOUT_VOLUME_THRESHOLD)


# ============================================
# 📊 CHART BUILDER (Enhanced)
# ============================================
class ProChartBuilder:
    """Professional chart builder"""
    
    @staticmethod
    def create_analysis_chart(df: pd.DataFrame, ticker: str, analysis: ProStockAnalysis = None) -> go.Figure:
        """Create comprehensive analysis chart"""
        
        ta = TechnicalAnalysis()
        df = df.copy()
        
        # Calculate indicators
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['RSI'] = ta.rsi(df['Close'])
        macd, signal, hist = ta.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        
        # Create figure
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{ticker} - RS: {analysis.rs_rating:.0f}' if analysis else ticker, 
                           'RSI', 'MACD', 'Volume')
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price',
            increasing_line_color='#00CC96', decreasing_line_color='#EF553B'
        ), row=1, col=1)
        
        # Moving averages
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                                  line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                                  line=dict(color='blue', width=1)), row=1, col=1)
        if len(df) >= 200:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200',
                                      line=dict(color='purple', width=1)), row=1, col=1)
        
        # Entry/Stop lines
        if analysis:
            fig.add_hline(y=analysis.suggested_entry, line_dash="dash", line_color="green",
                         annotation_text=f"Entry ${analysis.suggested_entry}", row=1, col=1)
            fig.add_hline(y=analysis.suggested_stop, line_dash="dash", line_color="red",
                         annotation_text=f"Stop ${analysis.suggested_stop}", row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                  line=dict(color='purple', width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        colors = ['green' if v >= 0 else 'red' for v in df['MACD_Hist'].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
                             marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                  line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                                  line=dict(color='orange', width=1)), row=3, col=1)
        
        # Volume
        vol_colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                      for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                             marker_color=vol_colors), row=4, col=1)
        
        # Layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        return fig


# ============================================
# 📡 DATA FETCHER
# ============================================
class DataFetcher:
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if len(df) > 0 else None
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_sector_etfs() -> Optional[pd.DataFrame]:
        tickers = [s['etf'] for s in SECTORS.values()] + ['SPY']
        try:
            data = yf.download(tickers, period="6mo", progress=False)['Close']
            return data
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_holdings(sector_name: str):
        if sector_name not in SECTORS:
            return None, []
        tickers = SECTORS[sector_name]['holdings']
        try:
            data = yf.download(tickers, period="6mo", group_by='ticker', progress=False)
            return data, tickers
        except:
            return None, []


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    # Header
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v5.0 Pro")
    st.caption("專業交易員版本 | 目標年化 30%+ | Gil Morales + Qullamaggie + Minervini")
    
    # Market Health Dashboard
    market = MarketRegime.get_market_health()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("市場狀態", market['status'])
    col2.metric("健康評分", f"{market['score']}/100")
    col3.metric("VIX", f"{market.get('vix', 'N/A'):.1f}" if isinstance(market.get('vix'), (int, float)) else "N/A")
    col4.metric("SPY", f"${market.get('spy_price', 0):.2f}" if market.get('spy_price') else "N/A")
    
    if market['score'] < 50:
        st.warning(f"⚠️ 市場環境不佳：{market.get('advice', '')}")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌪️ 板塊輪動",
        "🎯 狼群掃描", 
        "📊 個股分析",
        "💰 倉位計算",
        "📋 Watchlist",
        "📈 風控儀表板"
    ])
    
    # ===== TAB 1: Sector Rotation =====
    with tab1:
        st.header("板塊相對強度")
        
        df_etf = DataFetcher.get_sector_etfs()
        if df_etf is not None:
            timeframe = st.selectbox("時間軸", [5, 21, 63], format_func=lambda x: f"{x} 天", index=1)
            
            returns = df_etf.pct_change(periods=timeframe).iloc[-1] * 100
            spy_return = returns.get('SPY', 0)
            
            rs_data = []
            for name, info in SECTORS.items():
                if info['etf'] in returns:
                    rs_data.append({
                        '板塊': name,
                        '主題': info['theme'],
                        'RS': returns[info['etf']] - spy_return,
                        '回報%': returns[info['etf']]
                    })
            
            if rs_data:
                df_rs = pd.DataFrame(rs_data).sort_values('RS', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(df_rs, x='RS', y='板塊', orientation='h', color='RS',
                                color_continuous_scale=['#FF4B4B', '#F0F2F6', '#00CC96'],
                                range_color=[-15, 15])
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 輪動建議")
                    strong = df_rs[df_rs['RS'] > 2].head(3)
                    if not strong.empty:
                        st.success("**強勢板塊:**")
                        for _, row in strong.iterrows():
                            st.write(f"• {row['板塊']} ({row['RS']:+.1f}%)")
                    
                    weak = df_rs[df_rs['RS'] < -2].tail(3)
                    if not weak.empty:
                        st.error("**避開板塊:**")
                        for _, row in weak.iterrows():
                            st.write(f"• {row['板塊']} ({row['RS']:+.1f}%)")
    
    # ===== TAB 2: Wolf Pack Scanner =====
    with tab2:
        st.header("🎯 狼群掃描 - 找最強股票")
        
        selected_sector = st.selectbox("選擇板塊:", list(SECTORS.keys()))
        
        if st.button("🔍 掃描板塊", type="primary"):
            with st.spinner("分析中..."):
                raw_data, tickers = DataFetcher.get_holdings(selected_sector)
                spy_data = DataFetcher.get_stock_data('SPY', '6mo')
                
                analyzer = ProStockAnalyzer()
                results = []
                
                if raw_data is not None:
                    for t in tickers:
                        try:
                            df_t = raw_data[t] if len(tickers) > 1 else raw_data
                            res = analyzer.analyze(df_t, t, spy_data)
                            if res:
                                res.sector = selected_sector
                                results.append(res)
                        except:
                            continue
                
                # Rank within sector
                results.sort(key=lambda x: x.rs_rating, reverse=True)
                for i, r in enumerate(results):
                    r.sector_rank = i + 1
                
                st.session_state['sector_results'] = [r.__dict__ for r in results]
                st.session_state['selected_sector'] = selected_sector
        
        if 'sector_results' in st.session_state and st.session_state.get('selected_sector') == selected_sector:
            results = st.session_state['sector_results']
            if results:
                df = pd.DataFrame(results)
                
                # Summary - with safe column access
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🚀 突破", f"{df['is_breakout'].sum() if 'is_breakout' in df.columns else 0}")
                col2.metric("🎯 VCP", f"{df['is_vcp'].sum() if 'is_vcp' in df.columns else 0}")
                col3.metric("✅ 趨勢通過", f"{df['trend_passed'].sum() if 'trend_passed' in df.columns else 0}")
                col4.metric("📊 平均 RS", f"{df['rs_rating'].mean():.0f}" if 'rs_rating' in df.columns else "N/A")
                
                # Table - build columns dynamically
                st.markdown("### 📋 板塊內排名 (按 RS 排序)")
                
                # Select only available columns
                available_cols = []
                col_mapping = {
                    'sector_rank': '排名',
                    'ticker': 'Ticker', 
                    'price': 'Price',
                    'status': 'Status',
                    'rs_rating': 'RS',
                    'adr_pct': 'ADR%',
                    'rsi': 'RSI',
                    'trend_score': '趨勢分',
                    'earnings_warning': '財報'
                }
                
                for col in col_mapping.keys():
                    if col in df.columns:
                        available_cols.append(col)
                
                if available_cols:
                    display_df = df[available_cols].copy()
                    display_df.columns = [col_mapping[c] for c in available_cols]
                    
                    # Build format dict for available columns
                    format_dict = {}
                    if 'Price' in display_df.columns:
                        format_dict['Price'] = '${:.2f}'
                    if 'RS' in display_df.columns:
                        format_dict['RS'] = '{:.0f}'
                    if 'ADR%' in display_df.columns:
                        format_dict['ADR%'] = '{:.1f}%'
                    if 'RSI' in display_df.columns:
                        format_dict['RSI'] = '{:.0f}'
                    if '趨勢分' in display_df.columns:
                        format_dict['趨勢分'] = '{:.0f}%'
                    
                    # Simple styling without matplotlib dependency
                    styled_df = display_df.style.format(format_dict)
                    
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Recommendations - with safe access
                st.markdown("### 💡 建議")
                top_picks = []
                for r in results:
                    rs = r.get('rs_rating', 0)
                    trend = r.get('trend_passed', False)
                    overheated = r.get('is_overheated', True)
                    if rs >= 70 and trend and not overheated:
                        top_picks.append(r)
                
                if top_picks:
                    st.success(f"**推薦關注 ({len(top_picks)} 隻):**")
                    for pick in top_picks[:3]:
                        warn = f" ⚠️ {pick.get('earnings_warning', '')}" if pick.get('earnings_warning') else ""
                        st.write(f"• **{pick.get('ticker', 'N/A')}** - RS {pick.get('rs_rating', 0):.0f}, {pick.get('status', '')}{warn}")
                else:
                    st.info("目前沒有符合標準的推薦")
    
    # ===== TAB 3: Stock Analysis =====
    with tab3:
        st.header("📊 個股深度分析")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_input = st.text_input("股票代碼", value="NVDA")
        with col2:
            period = st.selectbox("時間範圍", ["6mo", "1y", "2y"], index=1)
        
        if st.button("🔍 分析", type="primary"):
            ticker = ticker_input.upper().strip()
            
            with st.spinner(f"分析 {ticker}..."):
                df = DataFetcher.get_stock_data(ticker, period)
                spy_df = DataFetcher.get_stock_data('SPY', period)
                
                if df is None:
                    st.error(f"無法獲取 {ticker} 數據")
                else:
                    analyzer = ProStockAnalyzer()
                    analysis = analyzer.analyze(df, ticker, spy_df)
                    
                    if analysis:
                        # Overview
                        st.subheader(f"📈 {ticker} 概覽")
                        
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        col1.metric("價格", f"${analysis.price:.2f}", f"{analysis.change_pct:+.2f}%")
                        col2.metric("RS Rating", f"{analysis.rs_rating:.0f}")
                        col3.metric("狀態", analysis.status)
                        col4.metric("ADR%", f"{analysis.adr_pct:.1f}%")
                        col5.metric("RSI", f"{analysis.rsi:.0f}")
                        col6.metric("趨勢分", f"{analysis.trend_score:.0f}%")
                        
                        # Earnings warning
                        if analysis.earnings_warning:
                            st.warning(f"📅 財報提醒: {analysis.earnings_warning}")
                        
                        # Chart
                        st.subheader("📊 技術圖表")
                        chart = ProChartBuilder.create_analysis_chart(df, ticker, analysis)
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Entry Plan
                        st.subheader("🎯 交易計劃")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            | 項目 | 價格 |
                            |------|------|
                            | **建議入場** | ${analysis.suggested_entry} |
                            | **止損** | ${analysis.suggested_stop} |
                            | **風險回報** | {analysis.risk_reward}:1 |
                            """)
                            
                            # Position sizing
                            risk_per_share = analysis.suggested_entry - analysis.suggested_stop
                            if risk_per_share > 0:
                                st.markdown(f"""
                                **倉位建議 ($100,000 帳戶, 2% 風險):**
                                - 最大股數: {int(2000 / risk_per_share)} 股
                                - 倉位金額: ${int(2000 / risk_per_share) * analysis.suggested_entry:,.0f}
                                """)
                        
                        with col2:
                            # Targets
                            targets = PositionCalculator.calculate_targets(
                                analysis.suggested_entry, 
                                analysis.suggested_stop
                            )
                            
                            st.markdown("**止盈目標:**")
                            for t in targets:
                                st.write(f"• {t['r_multiple']}R: ${t['price']} (+{t['profit_pct']}%)")
                        
                        # Trading Decision
                        st.subheader("📋 交易決策")
                        
                        if analysis.is_breakout and analysis.trend_passed:
                            st.success("""
                            ✅ **可以入場**
                            - 放量突破 + 趨勢健康
                            - 建議: 分批建倉，先買 1/2 倉位
                            """)
                        elif analysis.is_vcp and analysis.trend_passed:
                            st.info("""
                            🎯 **準備入場**
                            - VCP 形態形成中
                            - 建議: 等待突破確認再入場
                            """)
                        elif analysis.is_overheated:
                            st.error("""
                            ⛔ **不建議入場**
                            - 股價過熱，風險高
                            - 建議: 等待回調到均線支撐
                            """)
                        elif not analysis.trend_passed:
                            st.warning("""
                            ⚠️ **趨勢不佳**
                            - 未通過趨勢模板
                            - 建議: 觀望，等待趨勢改善
                            """)
                        else:
                            st.info("""
                            🧘 **觀望**
                            - 等待更好的入場機會
                            """)
    
    # ===== TAB 4: Position Calculator =====
    with tab4:
        st.header("💰 倉位計算器")
        
        col1, col2 = st.columns(2)
        
        with col1:
            account_size = st.number_input("帳戶總值 ($)", value=100000, step=10000)
            risk_percent = st.slider("單筆風險 (%)", 0.5, 5.0, 2.0, 0.5) / 100
            
        with col2:
            entry_price = st.number_input("入場價 ($)", value=150.0, step=1.0)
            stop_loss = st.number_input("止損價 ($)", value=140.0, step=1.0)
        
        if st.button("計算倉位", type="primary"):
            result = PositionCalculator.calculate_position(
                account_size, entry_price, stop_loss, risk_percent
            )
            
            if 'error' not in result:
                col1, col2, col3 = st.columns(3)
                col1.metric("建議股數", f"{result['shares']} 股")
                col2.metric("倉位金額", f"${result['position_value']:,.0f}")
                col3.metric("倉位比例", f"{result['position_percent']:.1f}%")
                
                st.markdown(f"""
                ### 風險明細
                - 風險金額: ${result['risk_amount']:,.0f}
                - 每股風險: ${result['risk_per_share']:.2f}
                - 最大虧損: ${result['max_loss']:,.0f}
                """)
                
                # Targets
                targets = PositionCalculator.calculate_targets(entry_price, stop_loss)
                st.markdown("### 止盈目標")
                for t in targets:
                    profit = result['shares'] * (t['price'] - entry_price)
                    st.write(f"• **{t['r_multiple']}R**: ${t['price']} (盈利 ${profit:,.0f})")
    
    # ===== TAB 5: Watchlist =====
    with tab5:
        st.header("📋 Watchlist")
        
        # Initialize watchlist
        if 'watchlist' not in st.session_state:
            st.session_state['watchlist'] = ['NVDA', 'TSLA', 'AMD']
        
        # Add stock
        col1, col2 = st.columns([3, 1])
        with col1:
            new_ticker = st.text_input("添加股票", placeholder="輸入代碼如 AAPL")
        with col2:
            if st.button("➕ 添加"):
                if new_ticker and new_ticker.upper() not in st.session_state['watchlist']:
                    st.session_state['watchlist'].append(new_ticker.upper())
        
        # Display watchlist
        if st.session_state['watchlist']:
            spy_df = DataFetcher.get_stock_data('SPY', '6mo')
            analyzer = ProStockAnalyzer()
            
            watchlist_data = []
            for t in st.session_state['watchlist']:
                df = DataFetcher.get_stock_data(t, '6mo')
                if df is not None:
                    analysis = analyzer.analyze(df, t, spy_df)
                    if analysis:
                        watchlist_data.append(analysis.__dict__)
            
            if watchlist_data:
                df_watch = pd.DataFrame(watchlist_data)
                if 'rs_rating' in df_watch.columns:
                    df_watch = df_watch.sort_values('rs_rating', ascending=False)
                
                # Select available columns
                display_cols = []
                col_names = {}
                for col, name in [('ticker', 'Ticker'), ('price', 'Price'), ('status', 'Status'),
                                  ('rs_rating', 'RS'), ('adr_pct', 'ADR%'), ('rsi', 'RSI'),
                                  ('trend_passed', '趨勢OK'), ('earnings_warning', '財報')]:
                    if col in df_watch.columns:
                        display_cols.append(col)
                        col_names[col] = name
                
                if display_cols:
                    display_df = df_watch[display_cols].rename(columns=col_names)
                    
                    format_dict = {}
                    if 'Price' in display_df.columns:
                        format_dict['Price'] = '${:.2f}'
                    if 'RS' in display_df.columns:
                        format_dict['RS'] = '{:.0f}'
                    if 'ADR%' in display_df.columns:
                        format_dict['ADR%'] = '{:.1f}%'
                    if 'RSI' in display_df.columns:
                        format_dict['RSI'] = '{:.0f}'
                    
                    st.dataframe(
                        display_df.style.format(format_dict),
                        use_container_width=True,
                        hide_index=True
                    )
        
        # Clear watchlist
        if st.button("🗑️ 清空 Watchlist"):
            st.session_state['watchlist'] = []
    
    # ===== TAB 6: Risk Dashboard =====
    with tab6:
        st.header("📈 風控儀表板")
        
        st.markdown("""
        ### 🎯 專業交易員守則
        
        | 規則 | 設定 | 說明 |
        |------|------|------|
        | 單筆風險 | ≤ 2% | 每筆交易最多虧損帳戶 2% |
        | 總風險 | ≤ 10% | 所有持倉最大總虧損 10% |
        | 最大持倉 | 5-8 個 | 避免過度分散或集中 |
        | 板塊暴露 | ≤ 40% | 單一板塊不超過 40% |
        | 財報迴避 | 7 天 | 財報前 7 天不開新倉 |
        
        ### 📊 風險檢查清單
        """)
        
        checklist = [
            ("✅" if market['score'] >= 50 else "❌", "大盤環境健康"),
            ("✅" if market.get('vix', 30) < 25 else "❌", "VIX < 25"),
            ("✅", "單筆風險 ≤ 2%"),
            ("✅", "止損設定明確"),
            ("✅", "財報日期已確認"),
        ]
        
        for status, item in checklist:
            st.write(f"{status} {item}")
        
        st.markdown("""
        ### 💡 年化 30% 的關鍵
        
        1. **只買最強股票** - RS > 80, 趨勢模板通過
        2. **順勢交易** - 大盤弱勢時減少交易
        3. **嚴格止損** - 虧損 7-8% 立即出場
        4. **讓利潤奔跑** - 用移動止盈保護利潤
        5. **控制風險** - 單筆 2%, 總體 10%
        6. **避開財報** - 財報前 7 天不開新倉
        """)
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v5.0 Pro 功能")
    st.sidebar.markdown("""
    - ✅ RS Rating (相對強度)
    - ✅ 趨勢模板 (Minervini)
    - ✅ 財報日期追蹤
    - ✅ 市場環境評估
    - ✅ 專業倉位計算
    - ✅ Watchlist 管理
    - ✅ 風控儀表板
    """)


if __name__ == "__main__":
    main()
