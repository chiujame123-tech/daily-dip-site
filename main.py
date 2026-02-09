# -*- coding: utf-8 -*-
"""
🌪️ Market Structure Radar - v5.0 (Pro Trader Edition)
=======================================================

Existing Features (v4.0):
✅ Tab 1-4: Sector Rotation, Wolf Pack, Temperature, Momentum
✅ Tab 5: Deep Analysis with interactive charts & Entry Points

New Pro Features (v5.0):
✅ 🚦 Market Regime (Traffic Light): Global trend filter
✅ 🐢 RS Rating: Relative Strength vs SPY
✅ 🚀 Pocket Pivot: Institutional buying detection
✅ 💣 Earnings Blackout: Risk avoidance
✅ 💰 Position Sizing: Risk management calculator

Author: AI Trading Assistant
Style: Vibe Coding - Clean, Readable, Modular
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
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ============================================
# ⚙️ CONFIGURATION
# ============================================
@dataclass
class Config:
    """Central configuration"""
    PAGE_TITLE: str = "Market Radar v5.0 Pro"
    PAGE_ICON: str = "🌪️"
    CACHE_TTL: int = 1800
    DATA_PERIOD: str = "1y"
    BENCHMARK: str = "SPY"
    
    # Technical thresholds
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    EXTENSION_DANGER: float = 25.0
    VCP_TIGHTNESS: float = 8.0
    BREAKOUT_VOLUME: float = 1.5
    HIGH_ADR: float = 5.0

CONFIG = Config()


# ============================================
# 📊 SECTOR DATA (EXPANDED v5.0)
# ============================================
# Added MU, WDC (SNDK), CIBR, IGV, XHB, XBI as requested
SECTORS: Dict[str, Dict] = {
    'SMH (半導體)': {
        'etf': 'SMH',
        'holdings': ['NVDA', 'TSM', 'AVGO', 'AMD', 'MU', 'WDC', 'QCOM', 'AMAT', 'LRCX', 'ARM', 'INTC'],
        'theme': '🔬 AI/芯片 & 記憶體'
    },
    'IGV (軟件 SaaS)': {
        'etf': 'IGV',
        'holdings': ['MSFT', 'CRM', 'NOW', 'ADBE', 'ORCL', 'PLTR', 'SNOW', 'DDOG', 'MDB', 'PANW'],
        'theme': '☁️ 雲端與軟件'
    },
    'CIBR (網絡安全)': {
        'etf': 'CIBR',
        'holdings': ['CRWD', 'PANW', 'FTNT', 'ZS', 'NET', 'CYBR', 'OKTA', 'SENT', 'CHKP'],
        'theme': '🛡️ 駭客防禦'
    },
    'XLK (科技)': {
        'etf': 'XLK',
        'holdings': ['MSFT', 'AAPL', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'QCOM', 'IBM'],
        'theme': '💻 大型科技'
    },
    'XLC (通訊)': {
        'etf': 'XLC',
        'holdings': ['META', 'GOOGL', 'NFLX', 'DIS', 'TMUS', 'VZ', 'CMCSA', 'T', 'WBD'],
        'theme': '📱 社交/媒體'
    },
    'XLF (金融)': {
        'etf': 'XLF',
        'holdings': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'V', 'MA'],
        'theme': '🏦 銀行/保險'
    },
    'XLY (消費)': {
        'etf': 'XLY',
        'holdings': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG'],
        'theme': '🛒 零售/消費'
    },
    'XLV (醫療)': {
        'etf': 'XLV',
        'holdings': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN'],
        'theme': '💊 製藥/醫療'
    },
    'XBI (生物科技)': {
        'etf': 'XBI',
        'holdings': ['VRTX', 'REGN', 'MRNA', 'BNTX', 'CRSP', 'EDIT', 'NTLA'],
        'theme': '🧬 基因與創新藥'
    },
    'XLE (能源)': {
        'etf': 'XLE',
        'holdings': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'OXY'],
        'theme': '⛽ 石油/天然氣'
    },
    'XHB (建築)': {
        'etf': 'XHB',
        'holdings': ['DHI', 'LEN', 'PHM', 'TOL', 'LOW', 'HD', 'SHW'],
        'theme': '🏠 房產與建材'
    },
    'IWM (小型股)': {
        'etf': 'IWM',
        'holdings': ['MSTR', 'SMCI', 'CELH', 'AFRM', 'SOFI', 'UPST', 'RIVN', 'COIN', 'DKNG'],
        'theme': '📈 高成長小型'
    },
    'KWEB (中概)': {
        'etf': 'KWEB',
        'holdings': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI'],
        'theme': '🇨🇳 中國互聯網'
    }
}

HOT_STOCKS = ['NVDA', 'TSLA', 'AMD', 'META', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 
              'SMCI', 'ARM', 'COIN', 'PLTR', 'MSTR', 'MU', 'WDC']


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
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0):
        sma = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def adr_percent(df: pd.DataFrame, period: int = 20) -> pd.Series:
        if 'High' not in df.columns or 'Low' not in df.columns:
            return pd.Series([0] * len(df))
        daily_range = (df['High'] / df['Low'] - 1) * 100
        return daily_range.rolling(period).mean()
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, lookback: int = 60) -> Dict:
        """Find key support and resistance levels"""
        recent = df.tail(lookback)
        close = recent['Close']
        high = recent['High']
        low = recent['Low']
        
        current = float(close.iloc[-1])
        
        levels = {
            'current': current,
            'high_52w': float(df['High'].tail(252).max()) if len(df) >= 252 else float(high.max()),
            'low_52w': float(df['Low'].tail(252).min()) if len(df) >= 252 else float(low.min()),
            'high_20d': float(high.max()),
            'low_20d': float(low.min()),
            'sma20': float(close.rolling(20).mean().iloc[-1]),
            'sma50': float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current,
            'sma200': float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else current,
        }
        
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(recent) - 5):
            if all(high.iloc[i] >= high.iloc[i-5:i]) and all(high.iloc[i] >= high.iloc[i+1:i+6]):
                swing_highs.append(float(high.iloc[i]))
            if all(low.iloc[i] <= low.iloc[i-5:i]) and all(low.iloc[i] <= low.iloc[i+1:i+6]):
                swing_lows.append(float(low.iloc[i]))
        
        resistances = [l for l in swing_highs if l > current]
        supports = [l for l in swing_lows if l < current]
        
        levels['nearest_resistance'] = min(resistances) if resistances else levels['high_20d']
        levels['nearest_support'] = max(supports) if supports else levels['low_20d']
        
        return levels


# ============================================
# 🎯 ENTRY POINT CALCULATOR
# ============================================
@dataclass
class EntryPoint:
    """Entry point recommendation"""
    entry_type: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    confidence: str
    notes: str


class EntryPointCalculator:
    """Calculate optimal entry points (Qullamaggie Style)"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def calculate(self, df: pd.DataFrame) -> List[EntryPoint]:
        entries = []
        if len(df) < 50: return entries
        
        close = df['Close']
        current = float(close.iloc[-1])
        atr = float(self.ta.atr(df).iloc[-1])
        levels = self.ta.find_support_resistance(df)
        
        # 1. Breakout
        breakout_entry = self._breakout_entry(df, current, atr, levels)
        if breakout_entry: entries.append(breakout_entry)
        
        # 2. Pullback
        pullback_entry = self._pullback_entry(df, current, atr, levels)
        if pullback_entry: entries.append(pullback_entry)
        
        # 3. VCP
        vcp_entry = self._vcp_entry(df, current, atr, levels)
        if vcp_entry: entries.append(vcp_entry)
        
        # 4. Support Bounce
        bounce_entry = self._bounce_entry(df, current, atr, levels)
        if bounce_entry: entries.append(bounce_entry)
        
        return entries
    
    def _breakout_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        high_20d = levels['high_20d']
        if current >= high_20d * 0.98:
            entry = high_20d * 1.001
            stop = entry - (atr * 2)
            target1 = entry + (atr * 3)
            target2 = entry + (atr * 5)
            rr = (target1 - entry) / (entry - stop)
            
            vol_avg = float(df['Volume'].tail(20).mean())
            vol_today = float(df['Volume'].iloc[-1])
            vol_ratio = vol_today / vol_avg
            confidence = 'high' if vol_ratio > 1.5 else 'medium' if vol_ratio > 1.0 else 'low'
            
            return EntryPoint('🚀 突破入場', round(entry, 2), round(stop, 2), round(target1, 2), 
                              round(target2, 2), round(rr, 2), confidence, 
                              f"突破 ${high_20d:.2f} 時買入，量比 {vol_ratio:.1f}x")
        return None
    
    def _pullback_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        sma20 = levels['sma20']
        sma50 = levels['sma50']
        
        if current <= sma20 * 1.02 and current >= sma20 * 0.98:
            entry = sma20
            stop = entry - (atr * 1.5)
            return EntryPoint('📉 回調到 EMA20', round(entry, 2), round(stop, 2), 
                              round(entry+(atr*2), 2), round(levels['high_20d'], 2), 
                              round((entry+(atr*2)-entry)/(entry-stop), 2), 'medium', 
                              f"回調到 20日均線 ${sma20:.2f} 支撐")
        
        if current <= sma50 * 1.02 and current >= sma50 * 0.98:
            entry = sma50
            stop = entry - (atr * 2)
            return EntryPoint('📉 回調到 SMA50', round(entry, 2), round(stop, 2), 
                              round(entry+(atr*3), 2), round(levels['high_20d'], 2), 
                              round((entry+(atr*3)-entry)/(entry-stop), 2), 'medium', 
                              f"回調到 50日均線 ${sma50:.2f} 支撐")
        return None
    
    def _vcp_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        recent_20 = df.tail(20)
        ranges = []
        for i in range(0, 20, 5):
            if i + 5 <= 20:
                period = recent_20.iloc[i:i+5]
                range_pct = (period['High'].max() - period['Low'].min()) / period['Low'].min() * 100
                ranges.append(range_pct)
        
        if len(ranges) >= 3:
            is_contracting = ranges[-1] < ranges[0] * 0.7
            if is_contracting and ranges[-1] < 8:
                pivot = float(df['High'].tail(10).max())
                entry = pivot * 1.001
                stop = current - (atr * 1.5)
                return EntryPoint('🎯 VCP 突破', round(entry, 2), round(stop, 2), 
                                  round(entry+(atr*3), 2), round(entry+(atr*5), 2), 
                                  round((entry+(atr*3)-entry)/(entry-stop), 2), 'high', 
                                  f"VCP 形態，緊縮度 {ranges[-1]:.1f}%")
        return None
    
    def _bounce_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        support = levels['nearest_support']
        if current <= support * 1.03 and current >= support * 0.99:
            entry = support
            stop = support - (atr * 1.5)
            return EntryPoint('💚 支撐反彈', round(entry, 2), round(stop, 2), 
                              round(entry+(atr*2), 2), round(levels['nearest_resistance'], 2), 
                              round((entry+(atr*2)-entry)/(entry-stop), 2), 'medium', 
                              f"從支撐位 ${support:.2f} 反彈")
        return None


# ============================================
# 📈 CHART BUILDER
# ============================================
class ChartBuilder:
    """Build interactive charts"""
    
    @staticmethod
    def create_stock_chart(df: pd.DataFrame, ticker: str, entries: List[EntryPoint] = None) -> go.Figure:
        ta = TechnicalAnalysis()
        df = df.copy()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['RSI'] = ta.rsi(df['Close'])
        macd, signal, hist = ta.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        bb_upper, bb_mid, bb_lower = ta.bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.5, 0.15, 0.15, 0.2], 
                            subplot_titles=(f'{ticker} 價格走勢', 'RSI', 'MACD', '成交量'))
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#00CC96', decreasing_line_color='#EF553B'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='blue', width=1)), row=1, col=1)
        if len(df) >= 200:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200', line=dict(color='purple', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        if entries:
            for entry in entries:
                fig.add_hline(y=entry.entry_price, line_dash="dash", line_color="green", annotation_text=f"入場 ${entry.entry_price}", row=1, col=1)
                fig.add_hline(y=entry.stop_loss, line_dash="dash", line_color="red", annotation_text=f"止損 ${entry.stop_loss}", row=1, col=1)
                fig.add_hline(y=entry.target_1, line_dash="dash", line_color="blue", annotation_text=f"目標1 ${entry.target_1}", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        colors = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=3, col=1)
        
        vol_colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=vol_colors), row=4, col=1)
        
        fig.update_layout(height=900, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False, template='plotly_dark')
        return fig


# ============================================
# 📡 DATA FETCHER (v5.0 UPDATED)
# ============================================
class DataFetcher:
    """Fetch market data"""
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if len(df) > 0 else None
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_sector_etfs() -> Optional[pd.DataFrame]:
        tickers = [s['etf'] for s in SECTORS.values()] + [CONFIG.BENCHMARK]
        try:
            data = yf.download(tickers, period="6mo", progress=False)['Close']
            return data
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_holdings(sector_name: str):
        if sector_name not in SECTORS: return None, []
        tickers = SECTORS[sector_name]['holdings']
        try:
            data = yf.download(tickers, period="6mo", group_by='ticker', progress=False)
            return data, tickers
        except:
            return None, []
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_vix() -> Optional[Dict]:
        try:
            vix = yf.download("^VIX", period="5d", progress=False)['Close']
            if len(vix) > 0:
                return {'value': float(vix.iloc[-1]), 'change': float(vix.iloc[-1] - vix.iloc[-2]) if len(vix) > 1 else 0}
        except: pass
        return None

    # --- NEW PRO FEATURES ---
    @staticmethod
    def get_earnings_date(ticker: str) -> str:
        """Get next earnings date to avoid binary events"""
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is not None and 'Earnings Date' in cal:
                # yfinance calendar structure varies, handle list or index
                dates = cal['Earnings Date']
                if len(dates) > 0:
                    next_date = dates[0]
                    # Ensure it's a date object
                    if hasattr(next_date, 'date'):
                        next_date = next_date.date()
                    
                    days = (next_date - datetime.now().date()).days
                    if 0 <= days <= 5:
                        return f"⚠️ 財報風險! ({days}天後)"
                    elif days < 0:
                        return "✅ 財報已過"
                    else:
                        return f"✅ 安全 ({days}天後)"
        except:
            pass
        return "📅 未知"

    @staticmethod
    def check_market_regime() -> Tuple[str, str]:
        """Check SPY status for Market Regime filter"""
        try:
            spy = yf.download("SPY", period="6mo", progress=False)
            if len(spy) > 50:
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                
                close = spy['Close'].iloc[-1]
                sma50 = spy['Close'].rolling(50).mean().iloc[-1]
                sma200 = spy['Close'].rolling(200).mean().iloc[-1]
                
                if close > sma50 and sma50 > sma200:
                    return "🟢 牛市強勢", "適合積極做多，提高倉位"
                elif close > sma50:
                    return "🟡 震盪偏強", "選擇性做多，關注優質股"
                elif close < sma50 and close > sma200:
                    return "🟠 調整中", "減少倉位，謹慎開倉"
                else:
                    return "🔴 熊市/崩盤", "現金為王 (Cash is King)，禁止做多"
        except:
            pass
        return "⚪ 未知", "數據不足"


# ============================================
# 📈 STOCK ANALYZER (v5.0 UPDATED)
# ============================================
@dataclass
class StockAnalysis:
    """Stock analysis result"""
    ticker: str
    price: float
    change_pct: float
    status: str
    status_code: int
    extension_pct: float
    rsi: float
    adr_pct: float
    dist_52w_high: float
    volume_ratio: float
    above_sma50: bool
    is_vcp: bool
    is_breakout: bool
    is_overheated: bool
    # New fields
    rs_rating: float = 0.0
    is_pocket_pivot: bool = False
    earnings_status: str = ""
    sector: str = ""


class StockAnalyzer:
    """Analyze stocks"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def analyze(self, df: pd.DataFrame, ticker: str = "", spy_df: pd.DataFrame = None) -> Optional[StockAnalysis]:
        """Analyze a stock"""
        if df is None or len(df) < 50: return None
        
        try:
            close = df['Close']
            volume = df['Volume']
            high = df['High']
            
            # Indicators
            sma50 = close.rolling(50).mean()
            rsi = self.ta.rsi(close)
            adr = self.ta.adr_percent(df)
            
            # Current values
            curr_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            curr_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else curr_price
            curr_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            curr_adr = float(adr.iloc[-1]) if not pd.isna(adr.iloc[-1]) else 3
            
            # 52w high
            high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
            
            # Volume ratio
            vol_avg = float(volume.tail(50).mean())
            vol_curr = float(volume.iloc[-1])
            vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1
            
            # Derived
            change_pct = (curr_price - prev_price) / prev_price * 100
            extension = (curr_price - curr_sma50) / curr_sma50 * 100
            dist_high = (curr_price / high_52w - 1) * 100
            above_sma50 = curr_price > curr_sma50
            
            # Pattern detection
            is_vcp = self._detect_vcp(df)
            is_breakout = self._detect_breakout(df)
            
            # --- PRO FEATURES ---
            # 1. Pocket Pivot (Volume > largest down volume in last 10 days)
            is_pocket_pivot = False
            if change_pct > 0:
                down_days = df[df['Close'] < df['Open']].tail(10)
                if not down_days.empty:
                    max_down_vol = down_days['Volume'].max()
                    if vol_curr > max_down_vol:
                        is_pocket_pivot = True
            
            # 2. RS Rating (Relative Strength vs SPY over 63 days)
            rs_rating = 0.0
            if spy_df is not None and len(spy_df) >= 63 and len(close) >= 63:
                # Align dates simply by tail
                stock_perf = (close.iloc[-1] / close.iloc[-63] - 1) * 100
                spy_perf = (spy_df.iloc[-1] / spy_df.iloc[-63] - 1) * 100
                rs_rating = stock_perf - spy_perf

            # 3. Earnings Check
            earnings_msg = DataFetcher.get_earnings_date(ticker)
            
            # Status Logic
            if is_breakout: status, code = "🚀 突破", 3
            elif is_pocket_pivot and above_sma50: status, code = "💎 口袋支點", 2.5
            elif is_vcp and above_sma50: status, code = "🎯 VCP", 2
            elif above_sma50 and vol_ratio < 0.8: status, code = "📊 整理", 1
            elif not above_sma50: status, code = "⚠️ 弱勢", -1
            else: status, code = "🧘 盤整", 0
            
            is_overheated = extension > 25 or curr_rsi > 80
            
            return StockAnalysis(
                ticker=ticker, price=curr_price, change_pct=change_pct,
                status=status, status_code=code, extension_pct=extension,
                rsi=curr_rsi, adr_pct=curr_adr, dist_52w_high=dist_high,
                volume_ratio=vol_ratio, above_sma50=above_sma50,
                is_vcp=is_vcp, is_breakout=is_breakout, is_overheated=is_overheated,
                rs_rating=rs_rating, is_pocket_pivot=is_pocket_pivot, earnings_status=earnings_msg
            )
        except:
            return None
    
    def _detect_vcp(self, df: pd.DataFrame) -> bool:
        if len(df) < 30: return False
        recent = df.tail(20)
        ranges = []
        for i in range(0, 20, 5):
            if i + 5 <= 20:
                period = recent.iloc[i:i+5]
                range_pct = (period['High'].max() - period['Low'].min()) / period['Low'].min() * 100
                ranges.append(range_pct)
        if len(ranges) >= 3:
            return ranges[-1] < ranges[0] * 0.7 and ranges[-1] < 8
        return False
    
    def _detect_breakout(self, df: pd.DataFrame) -> bool:
        if len(df) < 25: return False
        current = float(df['Close'].iloc[-1])
        high_20 = float(df['High'].iloc[-20:-1].max())
        vol_avg = float(df['Volume'].iloc[-20:-1].mean())
        vol_curr = float(df['Volume'].iloc[-1])
        return current > high_20 and vol_curr > vol_avg * 1.3


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    # Header & Market Regime
    st.title(f"{CONFIG.PAGE_ICON} Market Structure Radar v5.0 Pro")
    
    # 🚥 Market Traffic Light
    regime, advice = DataFetcher.check_market_regime()
    vix_data = DataFetcher.get_vix()
    
    col_m1, col_m2, col_m3 = st.columns([2, 4, 2])
    with col_m1:
        st.metric("市場狀態", regime)
    with col_m2:
        if "🟢" in regime: st.success(f"💡 {advice}")
        elif "🔴" in regime: st.error(f"💡 {advice}")
        else: st.warning(f"💡 {advice}")
    with col_m3:
        if vix_data:
            st.metric("VIX", f"{vix_data['value']:.1f}", f"{vix_data['change']:+.1f}")
            
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌪️ 板塊輪動", "🎯 狼群戰術", "🔥 溫度計", "🏆 動能排行", "📊 個股深度分析"
    ])
    
    # ===== TAB 1: Sector Rotation =====
    with tab1:
        st.header("板塊相對強度")
        df_etf = DataFetcher.get_sector_etfs()
        if df_etf is not None:
            timeframe = st.selectbox("時間軸", [5, 21, 63], format_func=lambda x: f"{x} 天", index=1)
            returns = df_etf.pct_change(periods=timeframe).iloc[-1] * 100
            spy_return = returns.get(CONFIG.BENCHMARK, 0)
            
            rs_data = []
            for name, info in SECTORS.items():
                if info['etf'] in returns:
                    rs_data.append({
                        '板塊': name, '主題': info['theme'],
                        'RS Rating': returns[info['etf']] - spy_return,
                        'Return %': returns[info['etf']]
                    })
            
            if rs_data:
                df_rs = pd.DataFrame(rs_data).sort_values('RS Rating', ascending=False)
                fig = px.bar(df_rs, x='RS Rating', y='板塊', orientation='h', color='RS Rating',
                             color_continuous_scale=['#FF4B4B', '#F0F2F6', '#00CC96'], range_color=[-15, 15])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=550)
                st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: Wolf Pack =====
    with tab2:
        st.header("🎯 狼群戰術")
        selected = st.selectbox("選擇板塊:", list(SECTORS.keys()))
        if st.button(f"掃描 {selected}", type="primary"):
            with st.spinner("分析中..."):
                raw_data, tickers = DataFetcher.get_holdings(selected)
                analyzer = StockAnalyzer()
                results = []
                if raw_data is not None:
                    for t in tickers:
                        try:
                            df_t = raw_data[t] if len(tickers) > 1 else raw_data
                            res = analyzer.analyze(df_t, t)
                            if res: results.append(res.__dict__)
                        except: continue
                st.session_state['sector_results'] = results
                st.session_state['selected_sector'] = selected
        
        if 'sector_results' in st.session_state and st.session_state.get('selected_sector') == selected:
            results = st.session_state['sector_results']
            if results:
                df = pd.DataFrame(results)
                col1, col2, col3 = st.columns(3)
                col1.metric("🚀 突破/口袋", f"{df['is_breakout'].sum() + df['is_pocket_pivot'].sum()}")
                col2.metric("🎯 VCP", f"{df['is_vcp'].sum()}")
                col3.metric("📈 > SMA50", f"{df['above_sma50'].sum()}/{len(df)}")
                
                st.dataframe(
                    df[['ticker', 'price', 'change_pct', 'status', 'adr_pct', 'rsi', 'volume_ratio']]
                    .rename(columns={'ticker': 'Ticker', 'price': 'Price', 'change_pct': 'Chg%', 'status': 'Status'})
                    .sort_values('Status', ascending=False)
                    .style.format({'Price': '${:.2f}', 'Chg%': '{:+.2f}%', 'adr_pct': '{:.1f}%', 'rsi': '{:.0f}'}),
                    use_container_width=True, hide_index=True
                )
    
    # ===== TAB 3: Temperature =====
    with tab3:
        st.header("🔥 過熱偵測")
        if 'sector_results' in st.session_state and st.session_state['sector_results']:
            df = pd.DataFrame(st.session_state['sector_results'])
            heat_ratio = df['is_overheated'].sum() / len(df) * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ 過熱比例", f"{heat_ratio:.0f}%")
            col2.metric("平均乖離", f"{df['extension_pct'].mean():.1f}%")
            col3.metric("平均 RSI", f"{df['rsi'].mean():.0f}")
            st.progress(min(int(heat_ratio), 100))
            if heat_ratio > 50: st.error("🚨 極度過熱!")
            elif heat_ratio > 30: st.warning("⚠️ 過熱警告")
            else: st.success("✅ 溫度正常")
        else:
            st.warning("請先在狼群戰術掃描板塊")
    
    # ===== TAB 4: Momentum Leaders =====
    with tab4:
        st.header("🏆 動能排行")
        selected_hot = st.multiselect("選擇股票", HOT_STOCKS, default=['NVDA', 'TSLA', 'AMD', 'PLTR'])
        if selected_hot:
            analyzer = StockAnalyzer()
            hot_results = []
            for t in selected_hot:
                df = DataFetcher.get_stock_data(t, "6mo")
                if df is not None:
                    res = analyzer.analyze(df, t)
                    if res: hot_results.append(res.__dict__)
            
            if hot_results:
                df_hot = pd.DataFrame(hot_results)
                df_hot['score'] = df_hot['adr_pct'] * 2 + (100 + df_hot['dist_52w_high']) / 10
                st.dataframe(
                    df_hot[['ticker', 'price', 'change_pct', 'status', 'adr_pct', 'rsi', 'dist_52w_high']]
                    .sort_values('status_code', ascending=False)
                    .style.format({'price': '${:.2f}', 'change_pct': '{:+.2f}%', 'adr_pct': '{:.1f}%', 'dist_52w_high': '{:+.1f}%'}),
                    use_container_width=True, hide_index=True
                )

    # ===== TAB 5: Stock Analysis (PRO UPGRADE) =====
    with tab5:
        st.header("📊 Pro 個股深度分析")
        
        c1, c2, c3 = st.columns([1, 1, 2])
        ticker = c1.text_input("股票代碼", value="NVDA").upper()
        account_size = c2.number_input("總資金 ($)", value=10000, step=1000)
        risk_pct = c3.slider("單筆風險 (%)", 0.5, 5.0, 1.0)
        
        if st.button("🚀 專業分析", type="primary"):
            with st.spinner(f"正在分析 {ticker}..."):
                df = DataFetcher.get_stock_data(ticker)
                spy_df = DataFetcher.get_stock_data("SPY") # Fetch SPY for RS calculation
                
                if df is None:
                    st.error(f"無法獲取 {ticker} 的數據")
                else:
                    analyzer = StockAnalyzer()
                    ta = TechnicalAnalysis()
                    entry_calc = EntryPointCalculator()
                    chart_builder = ChartBuilder()
                    
                    analysis = analyzer.analyze(df, ticker, spy_df)
                    entries = entry_calc.calculate(df)
                    levels = ta.find_support_resistance(df)
                    
                    if analysis:
                        # 1. Pro Overview
                        st.subheader(f"📈 {ticker} - {analysis.status}")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("價格", f"${analysis.price:.2f}", f"{analysis.change_pct:+.2f}%")
                        m2.metric("RS Rating (vs SPY)", f"{analysis.rs_rating:+.1f}", delta_color="normal")
                        m3.metric("口袋支點", "✅ 是" if analysis.is_pocket_pivot else "❌ 否")
                        m4.metric("財報狀態", analysis.earnings_status)
                        
                        if "⚠️" in analysis.earnings_status:
                            st.error(f"🚨 注意：{analysis.earnings_status}，建議避開或減倉！")
                        
                        # 2. Position Calculator
                        st.subheader("💰 倉位管理 & 交易計劃")
                        
                        # Calculate ATR Stop
                        atr_val = ta.atr(df).iloc[-1]
                        stop_loss = analysis.price - (atr_val * 2)
                        risk_per_share = analysis.price - stop_loss
                        
                        if risk_per_share > 0:
                            total_risk_amount = account_size * (risk_pct / 100)
                            shares = int(total_risk_amount / risk_per_share)
                            position_size = shares * analysis.price
                            
                            p1, p2, p3 = st.columns(3)
                            with p1:
                                st.info(f"**🛡️ 止損設置 (2ATR)**\n\n Price: ${stop_loss:.2f}")
                            with p2:
                                st.warning(f"**💵 建議倉位**\n\n Buy: **{shares}** 股 (${position_size:,.0f})")
                            with p3:
                                st.success(f"**🎯 風險控制**\n\n Risk: ${total_risk_amount:.0f} ({risk_pct}%)")
                        
                        # 3. Chart
                        st.subheader("📈 互動圖表")
                        fig = chart_builder.create_stock_chart(df, ticker, entries)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 4. Entry Points
                        if entries:
                            st.subheader("🎯 技術入場點")
                            for i, entry in enumerate(entries):
                                with st.expander(f"{entry.entry_type} - {entry.confidence.upper()}", expanded=(i==0)):
                                    st.write(f"**Price:** ${entry.entry_price} | **Stop:** ${entry.stop_loss} | **Target:** ${entry.target_1}")
                                    st.caption(entry.notes)


if __name__ == "__main__":
    main()
