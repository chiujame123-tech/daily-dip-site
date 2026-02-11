# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v6.0 Pro Edition
=============================================

專業交易員版本 - 目標年化 30%+

新增功能：
✅ Tab 7: 財報日曆 - 未來20天重要財報
✅ Tab 8: Setup 獵人 - BGU & VCP 專業掃描

Setup 類型：
1. BGU (Buyable Gap Up) - Qullamaggie 風格
2. VCP (Volatility Contraction Pattern) - Minervini 風格

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
    PAGE_TITLE: str = "Market Radar v6.0 Pro"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800
    
    # Risk Management
    MAX_RISK_PER_TRADE: float = 0.02
    MAX_POSITIONS: int = 8
    
    # Setup Thresholds
    BGU_MIN_GAP: float = 4.0  # 最小跳空 4%
    BGU_MIN_VOLUME: float = 2.0  # 最小量比 2x
    VCP_MAX_TIGHTNESS: float = 10.0  # VCP 最大緊縮度
    VCP_MIN_CONTRACTIONS: int = 2  # 最少收縮次數

CONFIG = Config()

# ============================================
# 📊 STOCK UNIVERSE
# ============================================
# 重要股票列表 - 用於財報追蹤和 Setup 掃描
STOCK_UNIVERSE = {
    'Mega Cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V'],
    'Semiconductors': ['NVDA', 'AMD', 'AVGO', 'TSM', 'MU', 'QCOM', 'AMAT', 'LRCX', 'KLAC', 'ARM', 'MRVL', 'INTC'],
    'Software': ['MSFT', 'CRM', 'ADBE', 'NOW', 'INTU', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'NET', 'MDB', 'PLTR'],
    'Internet': ['GOOGL', 'META', 'AMZN', 'NFLX', 'BKNG', 'ABNB', 'UBER', 'DASH', 'SNAP', 'PINS'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 'ABT', 'BMY', 'AMGN'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'COST', 'WMT', 'TGT'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL'],
    'Growth': ['NVDA', 'TSLA', 'AMD', 'SMCI', 'ARM', 'PLTR', 'COIN', 'MSTR', 'AFRM', 'SOFI', 'HOOD', 'RBLX'],
    'China ADR': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES'],
}

# 扁平化列表
ALL_STOCKS = list(set([s for stocks in STOCK_UNIVERSE.values() for s in stocks]))

SECTORS = {
    'SMH (半導體)': {'etf': 'SMH', 'holdings': STOCK_UNIVERSE['Semiconductors'], 'theme': '🔬 AI/芯片'},
    'XLK (科技)': {'etf': 'XLK', 'holdings': STOCK_UNIVERSE['Software'][:10], 'theme': '💻 軟件'},
    'XLC (通訊)': {'etf': 'XLC', 'holdings': STOCK_UNIVERSE['Internet'], 'theme': '📱 互聯網'},
    'XLF (金融)': {'etf': 'XLF', 'holdings': STOCK_UNIVERSE['Financials'], 'theme': '🏦 金融'},
    'XLY (消費)': {'etf': 'XLY', 'holdings': STOCK_UNIVERSE['Consumer'], 'theme': '🛒 消費'},
    'XLV (醫療)': {'etf': 'XLV', 'holdings': STOCK_UNIVERSE['Healthcare'], 'theme': '💊 醫療'},
    'XLE (能源)': {'etf': 'XLE', 'holdings': STOCK_UNIVERSE['Energy'], 'theme': '⛽ 能源'},
    'ARKK (成長)': {'etf': 'ARKK', 'holdings': STOCK_UNIVERSE['Growth'], 'theme': '🚀 成長'},
    'KWEB (中概)': {'etf': 'KWEB', 'holdings': STOCK_UNIVERSE['China ADR'], 'theme': '🇨🇳 中概'},
}

# ============================================
# 🧮 TECHNICAL ANALYSIS
# ============================================
class TechnicalAnalysis:
    """Technical indicator calculations"""
    
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
        """Calculate Relative Strength Rating"""
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


# ============================================
# 📅 EARNINGS CALENDAR
# ============================================
class EarningsCalendar:
    """Track earnings dates for major stocks"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_upcoming_earnings(stocks: List[str], days_ahead: int = 20) -> List[Dict]:
        """Get upcoming earnings for a list of stocks"""
        earnings_list = []
        today = datetime.now()
        cutoff = today + timedelta(days=days_ahead)
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(stocks):
            try:
                progress_text.text(f"掃描 {ticker}...")
                progress_bar.progress((i + 1) / len(stocks))
                
                stock = yf.Ticker(ticker)
                
                # Try to get earnings date
                try:
                    calendar = stock.calendar
                    if calendar is not None:
                        # Handle different calendar formats
                        earnings_date = None
                        
                        if isinstance(calendar, pd.DataFrame) and len(calendar) > 0:
                            if 'Earnings Date' in calendar.index:
                                earnings_date = calendar.loc['Earnings Date'].iloc[0]
                            elif len(calendar.columns) > 0:
                                earnings_date = calendar.iloc[0, 0]
                        elif isinstance(calendar, dict):
                            ed = calendar.get('Earnings Date', [])
                            if ed and len(ed) > 0:
                                earnings_date = ed[0]
                        
                        if earnings_date:
                            if isinstance(earnings_date, pd.Timestamp):
                                earnings_date = earnings_date.to_pydatetime()
                            elif isinstance(earnings_date, str):
                                earnings_date = pd.to_datetime(earnings_date).to_pydatetime()
                            
                            # Check if within range
                            if today <= earnings_date <= cutoff:
                                # Get additional info
                                info = stock.info
                                market_cap = info.get('marketCap', 0)
                                sector = info.get('sector', 'Unknown')
                                
                                # Get recent price data
                                hist = stock.history(period='5d')
                                if len(hist) > 0:
                                    price = float(hist['Close'].iloc[-1])
                                    change = (price / float(hist['Close'].iloc[0]) - 1) * 100 if len(hist) > 1 else 0
                                else:
                                    price = 0
                                    change = 0
                                
                                days_until = (earnings_date - today).days
                                
                                earnings_list.append({
                                    'ticker': ticker,
                                    'earnings_date': earnings_date,
                                    'days_until': days_until,
                                    'price': price,
                                    'change_5d': change,
                                    'market_cap': market_cap,
                                    'sector': sector,
                                    'urgency': '🔴' if days_until <= 3 else '🟡' if days_until <= 7 else '🟢'
                                })
                except:
                    pass
                    
            except Exception as e:
                continue
        
        progress_text.empty()
        progress_bar.empty()
        
        # Sort by date
        earnings_list.sort(key=lambda x: x['earnings_date'])
        return earnings_list


# ============================================
# 🎯 SETUP SCANNER - BGU & VCP
# ============================================
@dataclass
class SetupResult:
    """Setup detection result"""
    ticker: str
    setup_type: str  # 'BGU' or 'VCP'
    quality: str  # 'A+', 'A', 'B', 'C'
    score: float
    
    # Price data
    price: float
    gap_percent: float  # For BGU
    tightness: float  # For VCP
    
    # Entry/Exit
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    
    # Technicals
    rs_rating: float
    adr_percent: float
    volume_ratio: float
    above_sma50: bool
    
    # Details
    notes: str
    chart_url: str = ""


class SetupScanner:
    """
    Professional Setup Scanner
    
    Two main setups:
    1. BGU (Buyable Gap Up) - Qullamaggie Style
    2. VCP (Volatility Contraction Pattern) - Minervini Style
    """
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def scan_bgu(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[SetupResult]:
        """
        Scan for Buyable Gap Up (BGU)
        
        Qullamaggie BGU Criteria:
        1. Gap up >= 4% from previous close
        2. Volume >= 2x average
        3. Close in upper half of day's range
        4. Price above all major MAs
        5. RS Rating > 70
        6. No earnings within 2 weeks (ideally post-earnings gap)
        
        Entry: Buy at open or first pullback to gap level
        Stop: Below gap day low
        Target: 10-20% based on prior base depth
        """
        if df is None or len(df) < 50:
            return None
        
        try:
            # Get recent data
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            today_open = float(today['Open'])
            today_close = float(today['Close'])
            today_high = float(today['High'])
            today_low = float(today['Low'])
            today_volume = float(today['Volume'])
            yesterday_close = float(yesterday['Close'])
            
            # Calculate gap
            gap_percent = (today_open / yesterday_close - 1) * 100
            
            # Not a gap up
            if gap_percent < CONFIG.BGU_MIN_GAP:
                return None
            
            # Volume check
            avg_volume = float(df['Volume'].tail(50).mean())
            volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio < CONFIG.BGU_MIN_VOLUME:
                return None
            
            # Close in upper half of range
            day_range = today_high - today_low
            if day_range > 0:
                close_position = (today_close - today_low) / day_range
            else:
                close_position = 0.5
            
            upper_half_close = close_position >= 0.5
            
            # Price above MAs
            sma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
            above_mas = today_close > sma20 and today_close > sma50
            
            # RS Rating
            rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            
            # ADR%
            adr = float(self.ta.adr_percent(df).iloc[-1])
            
            # Calculate score
            score = 0
            notes = []
            
            # Gap size scoring
            if gap_percent >= 8:
                score += 30
                notes.append(f"強勁跳空 {gap_percent:.1f}%")
            elif gap_percent >= 6:
                score += 25
                notes.append(f"良好跳空 {gap_percent:.1f}%")
            else:
                score += 15
                notes.append(f"跳空 {gap_percent:.1f}%")
            
            # Volume scoring
            if volume_ratio >= 3:
                score += 25
                notes.append(f"爆量 {volume_ratio:.1f}x")
            elif volume_ratio >= 2:
                score += 20
                notes.append(f"放量 {volume_ratio:.1f}x")
            else:
                score += 10
            
            # Close position scoring
            if close_position >= 0.8:
                score += 20
                notes.append("收盤極強")
            elif close_position >= 0.6:
                score += 15
                notes.append("收盤強勢")
            elif close_position >= 0.5:
                score += 10
            else:
                return None  # Weak close, not buyable
            
            # RS scoring
            if rs >= 90:
                score += 15
                notes.append(f"RS 極強 {rs:.0f}")
            elif rs >= 80:
                score += 12
            elif rs >= 70:
                score += 8
            else:
                score -= 10  # Weak RS is a negative
            
            # Above MAs
            if above_mas:
                score += 10
            else:
                score -= 15
            
            # Determine quality
            if score >= 85:
                quality = 'A+'
            elif score >= 70:
                quality = 'A'
            elif score >= 55:
                quality = 'B'
            else:
                quality = 'C'
            
            # Calculate entry/exit
            atr = float(self.ta.atr(df).iloc[-1])
            
            # Entry: Gap day low or 1-3% below current (for pullback entry)
            entry = today_low  # Buy at pullback to gap day low
            stop = today_low - atr * 0.5  # Tight stop below gap day low
            target_1 = entry * 1.10  # 10% target
            target_2 = entry * 1.20  # 20% target
            
            risk = entry - stop
            reward = target_1 - entry
            rr = reward / risk if risk > 0 else 0
            
            return SetupResult(
                ticker=ticker,
                setup_type='BGU',
                quality=quality,
                score=score,
                price=today_close,
                gap_percent=gap_percent,
                tightness=0,
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                risk_reward=round(rr, 2),
                rs_rating=rs,
                adr_percent=adr,
                volume_ratio=volume_ratio,
                above_sma50=today_close > sma50,
                notes=" | ".join(notes)
            )
            
        except Exception as e:
            return None
    
    def scan_vcp(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[SetupResult]:
        """
        Scan for Volatility Contraction Pattern (VCP)
        
        Minervini VCP Criteria:
        1. Price in Stage 2 uptrend (above rising 50/200 MA)
        2. Series of price contractions (T1 > T2 > T3...)
        3. Each contraction is smaller than previous
        4. Volume dries up during contraction
        5. Final contraction < 10-15%
        6. RS Rating > 70
        
        Entry: Buy on breakout above pivot with volume
        Stop: Below VCP low
        Target: Prior base depth added to pivot
        """
        if df is None or len(df) < 100:
            return None
        
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            curr_price = float(close.iloc[-1])
            
            # Check Stage 2 uptrend
            sma50 = close.rolling(50).mean()
            sma150 = close.rolling(150).mean() if len(close) >= 150 else sma50
            sma200 = close.rolling(200).mean() if len(close) >= 200 else sma150
            
            curr_sma50 = float(sma50.iloc[-1])
            curr_sma150 = float(sma150.iloc[-1])
            curr_sma200 = float(sma200.iloc[-1])
            
            # Stage 2 check
            stage2 = (curr_price > curr_sma50 > curr_sma150 > curr_sma200)
            
            if not stage2:
                return None
            
            # Find contractions in last 60 days
            recent = df.tail(60)
            
            # Calculate weekly ranges
            contractions = []
            for i in range(0, min(50, len(recent)-5), 5):
                week = recent.iloc[i:i+5]
                week_high = float(week['High'].max())
                week_low = float(week['Low'].min())
                week_range = (week_high - week_low) / week_low * 100
                week_vol = float(week['Volume'].mean())
                contractions.append({
                    'range': week_range,
                    'high': week_high,
                    'low': week_low,
                    'volume': week_vol
                })
            
            if len(contractions) < 3:
                return None
            
            # Check for contraction pattern
            contraction_count = 0
            for i in range(1, len(contractions)):
                if contractions[i]['range'] < contractions[i-1]['range']:
                    contraction_count += 1
            
            if contraction_count < CONFIG.VCP_MIN_CONTRACTIONS:
                return None
            
            # Final tightness
            final_tightness = contractions[-1]['range']
            
            if final_tightness > CONFIG.VCP_MAX_TIGHTNESS:
                return None
            
            # Volume dry up
            avg_vol_early = np.mean([c['volume'] for c in contractions[:2]])
            avg_vol_late = np.mean([c['volume'] for c in contractions[-2:]])
            vol_dry_up = avg_vol_late < avg_vol_early
            
            # Find pivot (highest high in recent consolidation)
            pivot = float(recent['High'].max())
            vcp_low = float(recent['Low'].min())
            
            # Calculate base depth
            base_depth = (pivot - vcp_low) / vcp_low * 100
            
            # RS Rating
            rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            
            # ADR%
            adr = float(self.ta.adr_percent(df).iloc[-1])
            
            # Current volume ratio
            curr_vol = float(volume.iloc[-1])
            avg_vol = float(volume.tail(50).mean())
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1
            
            # Calculate score
            score = 0
            notes = []
            
            # Tightness scoring
            if final_tightness <= 5:
                score += 30
                notes.append(f"極緊縮 {final_tightness:.1f}%")
            elif final_tightness <= 8:
                score += 25
                notes.append(f"良好緊縮 {final_tightness:.1f}%")
            else:
                score += 15
                notes.append(f"緊縮 {final_tightness:.1f}%")
            
            # Contraction count
            if contraction_count >= 4:
                score += 20
                notes.append(f"{contraction_count} 次收縮")
            elif contraction_count >= 3:
                score += 15
            else:
                score += 10
            
            # Volume dry up
            if vol_dry_up:
                score += 15
                notes.append("量縮")
            
            # RS scoring
            if rs >= 90:
                score += 20
                notes.append(f"RS 極強 {rs:.0f}")
            elif rs >= 80:
                score += 15
            elif rs >= 70:
                score += 10
            else:
                score -= 10
            
            # Stage 2 confirmation
            score += 10  # Already passed stage 2 check
            
            # Distance to pivot (closer = better)
            dist_to_pivot = (pivot - curr_price) / curr_price * 100
            if dist_to_pivot <= 2:
                score += 10
                notes.append("接近突破點")
            elif dist_to_pivot <= 5:
                score += 5
            
            # Determine quality
            if score >= 85:
                quality = 'A+'
            elif score >= 70:
                quality = 'A'
            elif score >= 55:
                quality = 'B'
            else:
                quality = 'C'
            
            # Calculate entry/exit
            atr = float(self.ta.atr(df).iloc[-1])
            
            entry = pivot * 1.001  # Buy just above pivot
            stop = vcp_low - atr * 0.5  # Stop below VCP low
            target_1 = entry + (pivot - vcp_low)  # Add base depth
            target_2 = entry + (pivot - vcp_low) * 1.5
            
            risk = entry - stop
            reward = target_1 - entry
            rr = reward / risk if risk > 0 else 0
            
            return SetupResult(
                ticker=ticker,
                setup_type='VCP',
                quality=quality,
                score=score,
                price=curr_price,
                gap_percent=0,
                tightness=final_tightness,
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target_1, 2),
                target_2=round(target_2, 2),
                risk_reward=round(rr, 2),
                rs_rating=rs,
                adr_percent=adr,
                volume_ratio=vol_ratio,
                above_sma50=curr_price > curr_sma50,
                notes=" | ".join(notes)
            )
            
        except Exception as e:
            return None
    
    def scan_all(self, stocks: List[str], spy_df: pd.DataFrame = None) -> Tuple[List[SetupResult], List[SetupResult]]:
        """Scan all stocks for both BGU and VCP setups"""
        bgu_results = []
        vcp_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(stocks):
            status_text.text(f"掃描 {ticker}...")
            progress_bar.progress((i + 1) / len(stocks))
            
            try:
                df = yf.download(ticker, period='6mo', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is not None and len(df) > 50:
                    # Scan for BGU
                    bgu = self.scan_bgu(df, ticker, spy_df)
                    if bgu and bgu.score >= 50:
                        bgu_results.append(bgu)
                    
                    # Scan for VCP
                    vcp = self.scan_vcp(df, ticker, spy_df)
                    if vcp and vcp.score >= 50:
                        vcp_results.append(vcp)
                        
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort by score
        bgu_results.sort(key=lambda x: x.score, reverse=True)
        vcp_results.sort(key=lambda x: x.score, reverse=True)
        
        return bgu_results, vcp_results


# ============================================
# 💰 POSITION CALCULATOR
# ============================================
class PositionCalculator:
    @staticmethod
    def calculate(account: float, entry: float, stop: float, risk_pct: float = 0.02) -> Dict:
        risk_amount = account * risk_pct
        risk_per_share = abs(entry - stop)
        
        if risk_per_share <= 0:
            return {'error': 'Invalid stop'}
        
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
# 🌡️ MARKET REGIME
# ============================================
class MarketRegime:
    @staticmethod
    @st.cache_data(ttl=900)
    def get_health() -> Dict:
        default = {'status': '❓ 未知', 'score': 50, 'vix': None, 'spy_price': None, 'advice': ''}
        
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
            if spy_close > sma200: score += 15
            if spy_close > sma50: score += 10
            if len(spy) >= 21:
                ret = (spy_close / float(spy['Close'].iloc[-21]) - 1) * 100
                if ret > 0: score += 10
                elif ret < -5: score -= 15
            
            if vix_val < 15: score += 10
            elif vix_val > 25: score -= 15
            
            if score >= 75:
                status, advice = "🟢 強勢", "全力進攻"
            elif score >= 60:
                status, advice = "🟡 謹慎樂觀", "正常交易"
            elif score >= 40:
                status, advice = "🟠 震盪", "減少倉位"
            else:
                status, advice = "🔴 弱勢", "防守為主"
            
            return {
                'status': status, 'score': score, 'advice': advice,
                'vix': round(vix_val, 1), 'spy_price': round(spy_close, 2)
            }
        except:
            return default


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
# 📊 CHART BUILDER
# ============================================
class ChartBuilder:
    @staticmethod
    def create_setup_chart(df: pd.DataFrame, ticker: str, setup: SetupResult = None) -> go.Figure:
        """Create chart for setup visualization"""
        
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price',
            increasing_line_color='#00CC96', decreasing_line_color='#EF553B'
        ), row=1, col=1)
        
        # MAs
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                                 line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                                 line=dict(color='blue', width=1)), row=1, col=1)
        
        # Entry/Stop lines
        if setup:
            fig.add_hline(y=setup.entry_price, line_dash="dash", line_color="green",
                         annotation_text=f"Entry ${setup.entry_price}", row=1, col=1)
            fig.add_hline(y=setup.stop_loss, line_dash="dash", line_color="red",
                         annotation_text=f"Stop ${setup.stop_loss}", row=1, col=1)
            fig.add_hline(y=setup.target_1, line_dash="dash", line_color="blue",
                         annotation_text=f"T1 ${setup.target_1}", row=1, col=1)
        
        # Volume
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors,
                            name='Volume'), row=2, col=1)
        
        fig.update_layout(
            height=500, showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            title=f"{ticker} - {setup.setup_type if setup else ''}"
        )
        
        return fig


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    # Header
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v6.0 Pro")
    st.caption("專業交易員版本 | BGU & VCP Setup Scanner | 財報日曆")
    
    # Market Health
    market = MarketRegime.get_health()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("市場狀態", market['status'])
    col2.metric("健康評分", f"{market['score']}/100")
    col3.metric("VIX", f"{market['vix']:.1f}" if market['vix'] else "N/A")
    col4.metric("SPY", f"${market['spy_price']:.2f}" if market['spy_price'] else "N/A")
    
    st.divider()
    
    # Tabs
    tabs = st.tabs([
        "🌪️ 板塊輪動",
        "🎯 狼群掃描",
        "📊 個股分析",
        "💰 倉位計算",
        "📋 Watchlist",
        "📈 風控儀表板",
        "📅 財報日曆",  # NEW
        "🎯 Setup 獵人"  # NEW
    ])
    
    # ===== TAB 1: Sector Rotation =====
    with tabs[0]:
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
    
    # ===== TAB 2: Wolf Pack =====
    with tabs[1]:
        st.header("🎯 狼群掃描")
        
        selected = st.selectbox("選擇板塊:", list(SECTORS.keys()))
        
        if st.button("掃描", type="primary", key="scan_sector"):
            with st.spinner("分析中..."):
                spy_df = DataFetcher.get_stock('SPY', '6mo')
                scanner = SetupScanner()
                
                tickers = SECTORS[selected]['holdings']
                bgu_results, vcp_results = scanner.scan_all(tickers, spy_df)
                
                st.session_state['bgu_results'] = bgu_results
                st.session_state['vcp_results'] = vcp_results
        
        # Display results
        if 'bgu_results' in st.session_state:
            col1, col2 = st.columns(2)
            col1.metric("🚀 BGU 發現", len(st.session_state['bgu_results']))
            col2.metric("🎯 VCP 發現", len(st.session_state['vcp_results']))
    
    # ===== TAB 3: Stock Analysis =====
    with tabs[2]:
        st.header("📊 個股分析")
        
        ticker = st.text_input("股票代碼", value="NVDA").upper()
        
        if st.button("分析", type="primary", key="analyze_stock"):
            df = DataFetcher.get_stock(ticker, "1y")
            spy_df = DataFetcher.get_stock('SPY', '1y')
            
            if df is not None:
                scanner = SetupScanner()
                
                # Check for setups
                bgu = scanner.scan_bgu(df, ticker, spy_df)
                vcp = scanner.scan_vcp(df, ticker, spy_df)
                
                # Display
                col1, col2, col3 = st.columns(3)
                col1.metric("價格", f"${float(df['Close'].iloc[-1]):.2f}")
                
                ta = TechnicalAnalysis()
                rs = ta.rs_rating(df, spy_df)
                col2.metric("RS Rating", f"{rs:.0f}")
                
                adr = float(ta.adr_percent(df).iloc[-1])
                col3.metric("ADR%", f"{adr:.1f}%")
                
                # Setup status
                if bgu:
                    st.success(f"🚀 BGU 信號! Quality: {bgu.quality}, Score: {bgu.score:.0f}")
                if vcp:
                    st.info(f"🎯 VCP 信號! Quality: {vcp.quality}, Score: {vcp.score:.0f}")
                
                # Chart
                setup = bgu or vcp
                fig = ChartBuilder.create_setup_chart(df, ticker, setup)
                st.plotly_chart(fig, use_container_width=True)
                
                if setup:
                    st.markdown(f"""
                    ### 交易計劃
                    | 項目 | 價格 |
                    |------|------|
                    | 入場 | ${setup.entry_price} |
                    | 止損 | ${setup.stop_loss} |
                    | 目標1 | ${setup.target_1} |
                    | 目標2 | ${setup.target_2} |
                    | R:R | {setup.risk_reward}:1 |
                    
                    **備註:** {setup.notes}
                    """)
    
    # ===== TAB 4: Position Calculator =====
    with tabs[3]:
        st.header("💰 倉位計算器")
        
        col1, col2 = st.columns(2)
        with col1:
            account = st.number_input("帳戶金額 ($)", value=100000, step=10000)
            risk_pct = st.slider("風險 (%)", 0.5, 3.0, 2.0, 0.5) / 100
        with col2:
            entry = st.number_input("入場價 ($)", value=150.0, step=1.0)
            stop = st.number_input("止損價 ($)", value=145.0, step=1.0)
        
        if st.button("計算", type="primary", key="calc_position"):
            result = PositionCalculator.calculate(account, entry, stop, risk_pct)
            
            if 'error' not in result:
                col1, col2, col3 = st.columns(3)
                col1.metric("股數", f"{result['shares']}")
                col2.metric("倉位金額", f"${result['position_value']:,.0f}")
                col3.metric("最大虧損", f"${result['max_loss']:,.0f}")
    
    # ===== TAB 5: Watchlist =====
    with tabs[4]:
        st.header("📋 Watchlist")
        
        if 'watchlist' not in st.session_state:
            st.session_state['watchlist'] = ['NVDA', 'TSLA', 'AMD']
        
        new_ticker = st.text_input("添加股票")
        if st.button("添加", key="add_watch"):
            if new_ticker and new_ticker.upper() not in st.session_state['watchlist']:
                st.session_state['watchlist'].append(new_ticker.upper())
        
        st.write("**當前 Watchlist:**", ", ".join(st.session_state['watchlist']))
    
    # ===== TAB 6: Risk Dashboard =====
    with tabs[5]:
        st.header("📈 風控儀表板")
        
        st.markdown("""
        ### 🎯 交易守則
        | 規則 | 設定 |
        |------|------|
        | 單筆風險 | ≤ 2% |
        | 最大持倉 | 5-8 個 |
        | 板塊暴露 | ≤ 40% |
        | 財報迴避 | 7 天 |
        """)
    
    # ===== TAB 7: EARNINGS CALENDAR (NEW!) =====
    with tabs[6]:
        st.header("📅 財報日曆 - 未來 20 天")
        
        st.info("""
        **為什麼追蹤財報很重要？**
        - 財報是最大的股價催化劑
        - BGU 通常發生在財報後
        - 持倉股票財報前需要決定是否持有
        - 可以提前佈局強勢股的財報
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            sector_filter = st.selectbox(
                "選擇板塊",
                ["全部"] + list(STOCK_UNIVERSE.keys()),
                key="earnings_sector"
            )
        with col2:
            days_ahead = st.slider("天數", 7, 30, 20)
        
        if st.button("🔍 掃描財報日期", type="primary", key="scan_earnings"):
            # Select stocks based on filter
            if sector_filter == "全部":
                stocks_to_scan = ALL_STOCKS[:50]  # Limit to 50 for speed
            else:
                stocks_to_scan = STOCK_UNIVERSE.get(sector_filter, [])
            
            earnings = EarningsCalendar.get_upcoming_earnings(stocks_to_scan, days_ahead)
            st.session_state['earnings_data'] = earnings
        
        # Display earnings
        if 'earnings_data' in st.session_state and st.session_state['earnings_data']:
            earnings = st.session_state['earnings_data']
            
            st.success(f"找到 {len(earnings)} 隻股票在未來 {days_ahead} 天內有財報")
            
            # Summary
            col1, col2, col3 = st.columns(3)
            urgent = len([e for e in earnings if e['days_until'] <= 3])
            soon = len([e for e in earnings if 3 < e['days_until'] <= 7])
            later = len([e for e in earnings if e['days_until'] > 7])
            
            col1.metric("🔴 3天內", urgent)
            col2.metric("🟡 7天內", soon)
            col3.metric("🟢 7天後", later)
            
            # Table
            st.markdown("### 📋 財報列表")
            
            df_earnings = pd.DataFrame(earnings)
            df_earnings['earnings_date'] = pd.to_datetime(df_earnings['earnings_date']).dt.strftime('%Y-%m-%d')
            df_earnings['market_cap'] = (df_earnings['market_cap'] / 1e9).round(1)
            
            display_df = df_earnings[['urgency', 'ticker', 'earnings_date', 'days_until', 
                                       'price', 'change_5d', 'market_cap', 'sector']].copy()
            display_df.columns = ['⚠️', 'Ticker', '財報日期', '天數', '價格', '5日%', '市值(B)', '板塊']
            
            st.dataframe(
                display_df.style.format({
                    '價格': '${:.2f}',
                    '5日%': '{:+.1f}%',
                    '市值(B)': '${:.1f}B'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Earnings trade ideas
            st.markdown("### 💡 財報交易想法")
            
            # Strong stocks with upcoming earnings (potential BGU candidates)
            strong_earnings = [e for e in earnings if e['change_5d'] > 2]
            if strong_earnings:
                st.success("**強勢股財報 (潛在 BGU):**")
                for e in strong_earnings[:5]:
                    st.write(f"• **{e['ticker']}** - {e['earnings_date']}, 5日漲 {e['change_5d']:.1f}%")
            
            # Weak stocks (avoid)
            weak_earnings = [e for e in earnings if e['change_5d'] < -3]
            if weak_earnings:
                st.warning("**弱勢股財報 (避開):**")
                for e in weak_earnings[:5]:
                    st.write(f"• **{e['ticker']}** - {e['earnings_date']}, 5日跌 {e['change_5d']:.1f}%")
        
        elif 'earnings_data' in st.session_state:
            st.warning("沒有找到符合條件的財報")
    
    # ===== TAB 8: SETUP HUNTER (NEW!) =====
    with tabs[7]:
        st.header("🎯 Setup 獵人 - BGU & VCP 專業掃描")
        
        # Education section
        with st.expander("📖 什麼是 BGU 和 VCP？點擊展開學習", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🚀 BGU (Buyable Gap Up)
                **Qullamaggie 風格跳空突破**
                
                **識別特徵：**
                - 跳空 ≥ 4% (越大越好)
                - 成交量 ≥ 2x 平均 (越大越好)
                - 收盤在日內高點附近
                - 價格在所有均線之上
                - RS Rating > 70
                
                **入場時機：**
                - 跳空當天直接買入
                - 或等待回調到跳空日低點
                
                **止損位置：**
                - 跳空日低點下方 0.5 ATR
                
                **獲利目標：**
                - T1: 10%
                - T2: 20%
                
                **最佳情況：**
                - 財報後跳空 (Earnings Gap)
                - 板塊輪動帶動
                - 突破長期整理區間
                """)
            
            with col2:
                st.markdown("""
                ### 🎯 VCP (Volatility Contraction Pattern)
                **Minervini 風格波動收縮**
                
                **識別特徵：**
                - Stage 2 上升趨勢
                - 2-6 次價格收縮
                - 每次收縮幅度遞減
                - 最後收縮 < 10%
                - 成交量逐漸萎縮
                - RS Rating > 70
                
                **入場時機：**
                - 突破 Pivot 高點時買入
                - 需要成交量確認
                
                **止損位置：**
                - VCP 低點下方
                
                **獲利目標：**
                - T1: 整理區間深度
                - T2: 1.5x 整理區間深度
                
                **最佳情況：**
                - 強勢板塊中的領頭羊
                - 多次測試支撐後突破
                - 機構資金參與
                """)
        
        st.divider()
        
        # Scanning section
        st.subheader("🔍 掃描 Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            scan_universe = st.selectbox(
                "掃描範圍",
                ["Growth 高成長股", "Mega Cap 大型股", "Semiconductors 半導體", 
                 "Software 軟件", "全部股票 (較慢)"],
                key="setup_universe"
            )
        with col2:
            setup_type = st.selectbox(
                "Setup 類型",
                ["全部", "只掃描 BGU", "只掃描 VCP"],
                key="setup_type"
            )
        
        if st.button("🎯 開始掃描", type="primary", key="scan_setups"):
            # Select stocks
            if "Growth" in scan_universe:
                stocks = STOCK_UNIVERSE['Growth']
            elif "Mega Cap" in scan_universe:
                stocks = STOCK_UNIVERSE['Mega Cap']
            elif "Semiconductors" in scan_universe:
                stocks = STOCK_UNIVERSE['Semiconductors']
            elif "Software" in scan_universe:
                stocks = STOCK_UNIVERSE['Software']
            else:
                stocks = ALL_STOCKS[:30]
            
            spy_df = DataFetcher.get_stock('SPY', '6mo')
            scanner = SetupScanner()
            
            with st.spinner("掃描中..."):
                bgu_results, vcp_results = scanner.scan_all(stocks, spy_df)
            
            st.session_state['setup_bgu'] = bgu_results
            st.session_state['setup_vcp'] = vcp_results
        
        # Display results
        if 'setup_bgu' in st.session_state or 'setup_vcp' in st.session_state:
            bgu_results = st.session_state.get('setup_bgu', [])
            vcp_results = st.session_state.get('setup_vcp', [])
            
            # Summary
            col1, col2 = st.columns(2)
            col1.metric("🚀 BGU 發現", len(bgu_results))
            col2.metric("🎯 VCP 發現", len(vcp_results))
            
            # BGU Results
            if bgu_results and setup_type != "只掃描 VCP":
                st.markdown("### 🚀 BGU 信號 (Buyable Gap Up)")
                
                for setup in bgu_results[:5]:
                    with st.expander(f"**{setup.ticker}** - Quality: {setup.quality} | Score: {setup.score:.0f}", expanded=(setup.quality == 'A+')):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **基本信息：**
                            - 價格: ${setup.price:.2f}
                            - 跳空: {setup.gap_percent:.1f}%
                            - 量比: {setup.volume_ratio:.1f}x
                            - RS: {setup.rs_rating:.0f}
                            - ADR%: {setup.adr_percent:.1f}%
                            
                            **交易計劃：**
                            | 項目 | 價格 |
                            |------|------|
                            | 入場 | ${setup.entry_price} |
                            | 止損 | ${setup.stop_loss} |
                            | T1 | ${setup.target_1} |
                            | T2 | ${setup.target_2} |
                            | R:R | {setup.risk_reward}:1 |
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **信號詳情：**
                            {setup.notes}
                            
                            **倉位建議 ($100K, 2% 風險)：**
                            """)
                            pos = PositionCalculator.calculate(100000, setup.entry_price, setup.stop_loss)
                            if 'error' not in pos:
                                st.write(f"- 股數: {pos['shares']}")
                                st.write(f"- 金額: ${pos['position_value']:,.0f}")
                        
                        # Chart button
                        if st.button(f"查看 {setup.ticker} 圖表", key=f"bgu_chart_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "3mo")
                            if df is not None:
                                fig = ChartBuilder.create_setup_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
            
            # VCP Results
            if vcp_results and setup_type != "只掃描 BGU":
                st.markdown("### 🎯 VCP 信號 (Volatility Contraction)")
                
                for setup in vcp_results[:5]:
                    with st.expander(f"**{setup.ticker}** - Quality: {setup.quality} | Score: {setup.score:.0f}", expanded=(setup.quality == 'A+')):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **基本信息：**
                            - 價格: ${setup.price:.2f}
                            - 緊縮度: {setup.tightness:.1f}%
                            - RS: {setup.rs_rating:.0f}
                            - ADR%: {setup.adr_percent:.1f}%
                            
                            **交易計劃：**
                            | 項目 | 價格 |
                            |------|------|
                            | 入場 (突破) | ${setup.entry_price} |
                            | 止損 | ${setup.stop_loss} |
                            | T1 | ${setup.target_1} |
                            | T2 | ${setup.target_2} |
                            | R:R | {setup.risk_reward}:1 |
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **信號詳情：**
                            {setup.notes}
                            
                            **倉位建議 ($100K, 2% 風險)：**
                            """)
                            pos = PositionCalculator.calculate(100000, setup.entry_price, setup.stop_loss)
                            if 'error' not in pos:
                                st.write(f"- 股數: {pos['shares']}")
                                st.write(f"- 金額: ${pos['position_value']:,.0f}")
                        
                        if st.button(f"查看 {setup.ticker} 圖表", key=f"vcp_chart_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "3mo")
                            if df is not None:
                                fig = ChartBuilder.create_setup_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
            
            # No results
            if not bgu_results and not vcp_results:
                st.info("沒有發現符合條件的 Setup。這可能表示：\n"
                       "1. 市場整體缺乏動能\n"
                       "2. 需要等待更好的機會\n"
                       "3. 嘗試掃描其他板塊")
        
        # Trading rules reminder
        st.divider()
        st.markdown("""
        ### ⚠️ Setup 交易守則
        
        1. **只交易 A+ 和 A 級別** - 不要為了交易而交易
        2. **確認大盤環境** - 熊市中 Setup 成功率大降
        3. **檢查財報日期** - 避開財報前 7 天開新倉
        4. **嚴格止損** - 止損觸發立即出場，不要期望
        5. **分批止盈** - T1 減半倉，剩餘追蹤止盈
        6. **控制倉位** - 單筆不超過帳戶 2% 風險
        """)
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v6.0 Pro 功能")
    st.sidebar.markdown("""
    - ✅ **財報日曆** - 20天財報追蹤
    - ✅ **BGU 掃描** - Qullamaggie 風格
    - ✅ **VCP 掃描** - Minervini 風格
    - ✅ **Setup 評分** - A+/A/B/C
    - ✅ **完整交易計劃**
    - ✅ **自動倉位計算**
    """)


if __name__ == "__main__":
    main()
