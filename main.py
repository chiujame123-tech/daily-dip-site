# -*- coding: utf-8 -*-
"""
🌪️ Market Structure Radar - v4.0 (Entry Point Edition)
=======================================================

New Features:
✅ Tab 5: 個股深度分析 - 互動圖表 + 最佳入場點
✅ 支撐/阻力位計算
✅ 買入區間建議 (Qullamaggie Style)
✅ 風險/回報計算
✅ 多時間框架分析

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
    PAGE_TITLE: str = "Market Structure Radar v4.0"
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
# 📊 SECTOR DATA
# ============================================
SECTORS: Dict[str, Dict] = {
    'SMH (半導體)': {
        'etf': 'SMH',
        'holdings': ['NVDA', 'TSM', 'AVGO', 'AMD', 'MU', 'QCOM', 'AMAT', 'LRCX', 'TXN', 'INTC', 'ADI', 'KLAC', 'MRVL', 'ARM'],
        'theme': '🔬 AI/芯片'
    },
    'XLK (科技)': {
        'etf': 'XLK',
        'holdings': ['MSFT', 'AAPL', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'QCOM', 'IBM', 'NOW', 'INTU'],
        'theme': '💻 大型科技'
    },
    'XLC (通訊)': {
        'etf': 'XLC',
        'holdings': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'TMUS', 'VZ', 'CMCSA', 'T', 'WBD'],
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
    'XLE (能源)': {
        'etf': 'XLE',
        'holdings': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY'],
        'theme': '⛽ 石油/天然氣'
    },
    'ARKK (創新)': {
        'etf': 'ARKK',
        'holdings': ['TSLA', 'COIN', 'ROKU', 'SQ', 'PATH', 'HOOD', 'RBLX', 'U', 'DKNG'],
        'theme': '🚀 顛覆創新'
    },
    'KWEB (中概)': {
        'etf': 'KWEB',
        'holdings': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME'],
        'theme': '🇨🇳 中國互聯網'
    },
    'IWM (小型股)': {
        'etf': 'IWM',
        'holdings': ['MSTR', 'SMCI', 'CELH', 'AFRM', 'SOFI', 'UPST', 'RIVN', 'PLUG', 'FSLR'],
        'theme': '📈 高成長小型'
    }
}

# 熱門股票快速選擇
HOT_STOCKS = ['NVDA', 'TSLA', 'AMD', 'META', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 
              'SMCI', 'ARM', 'COIN', 'PLTR', 'SOFI', 'NIO', 'BABA', 'MSTR']


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
        
        # Current price
        current = float(close.iloc[-1])
        
        # Key levels
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
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(recent) - 5):
            # Swing high: higher than 5 bars before and after
            if all(high.iloc[i] >= high.iloc[i-5:i]) and all(high.iloc[i] >= high.iloc[i+1:i+6]):
                swing_highs.append(float(high.iloc[i]))
            # Swing low
            if all(low.iloc[i] <= low.iloc[i-5:i]) and all(low.iloc[i] <= low.iloc[i+1:i+6]):
                swing_lows.append(float(low.iloc[i]))
        
        # Get nearest support/resistance
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
    entry_type: str  # 'breakout', 'pullback', 'vcp', 'bounce'
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    confidence: str  # 'high', 'medium', 'low'
    notes: str


class EntryPointCalculator:
    """Calculate optimal entry points (Qullamaggie Style)"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def calculate(self, df: pd.DataFrame) -> List[EntryPoint]:
        """Calculate all possible entry points"""
        entries = []
        
        if len(df) < 50:
            return entries
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        current = float(close.iloc[-1])
        atr = float(self.ta.atr(df).iloc[-1])
        levels = self.ta.find_support_resistance(df)
        
        # 1. Breakout Entry (突破入場)
        breakout_entry = self._breakout_entry(df, current, atr, levels)
        if breakout_entry:
            entries.append(breakout_entry)
        
        # 2. Pullback Entry (回調入場)
        pullback_entry = self._pullback_entry(df, current, atr, levels)
        if pullback_entry:
            entries.append(pullback_entry)
        
        # 3. VCP Entry (波動收縮入場)
        vcp_entry = self._vcp_entry(df, current, atr, levels)
        if vcp_entry:
            entries.append(vcp_entry)
        
        # 4. Support Bounce Entry (支撐反彈)
        bounce_entry = self._bounce_entry(df, current, atr, levels)
        if bounce_entry:
            entries.append(bounce_entry)
        
        return entries
    
    def _breakout_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        """Breakout above 20-day high"""
        high_20d = levels['high_20d']
        
        # Check if near breakout point
        if current >= high_20d * 0.98:
            entry = high_20d * 1.001  # Slightly above breakout
            stop = entry - (atr * 2)
            target1 = entry + (atr * 3)
            target2 = entry + (atr * 5)
            rr = (target1 - entry) / (entry - stop)
            
            # Check volume
            vol_avg = float(df['Volume'].tail(20).mean())
            vol_today = float(df['Volume'].iloc[-1])
            vol_ratio = vol_today / vol_avg
            
            confidence = 'high' if vol_ratio > 1.5 else 'medium' if vol_ratio > 1.0 else 'low'
            
            return EntryPoint(
                entry_type='🚀 突破入場',
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target1, 2),
                target_2=round(target2, 2),
                risk_reward=round(rr, 2),
                confidence=confidence,
                notes=f"突破 ${high_20d:.2f} 時買入，量比 {vol_ratio:.1f}x"
            )
        return None
    
    def _pullback_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        """Pullback to moving average"""
        sma20 = levels['sma20']
        sma50 = levels['sma50']
        
        # Price near SMA20 or SMA50
        if current <= sma20 * 1.02 and current >= sma20 * 0.98:
            entry = sma20
            stop = entry - (atr * 1.5)
            target1 = entry + (atr * 2)
            target2 = levels['high_20d']
            rr = (target1 - entry) / (entry - stop)
            
            return EntryPoint(
                entry_type='📉 回調到 EMA20',
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target1, 2),
                target_2=round(target2, 2),
                risk_reward=round(rr, 2),
                confidence='medium',
                notes=f"回調到 20日均線 ${sma20:.2f} 支撐"
            )
        
        if current <= sma50 * 1.02 and current >= sma50 * 0.98:
            entry = sma50
            stop = entry - (atr * 2)
            target1 = entry + (atr * 3)
            target2 = levels['high_20d']
            rr = (target1 - entry) / (entry - stop)
            
            return EntryPoint(
                entry_type='📉 回調到 SMA50',
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target1, 2),
                target_2=round(target2, 2),
                risk_reward=round(rr, 2),
                confidence='medium',
                notes=f"回調到 50日均線 ${sma50:.2f} 支撐"
            )
        
        return None
    
    def _vcp_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        """VCP pattern entry"""
        # Check for volatility contraction
        recent_20 = df.tail(20)
        ranges = []
        
        for i in range(0, 20, 5):
            if i + 5 <= 20:
                period = recent_20.iloc[i:i+5]
                range_pct = (period['High'].max() - period['Low'].min()) / period['Low'].min() * 100
                ranges.append(range_pct)
        
        if len(ranges) >= 3:
            is_contracting = ranges[-1] < ranges[0] * 0.7  # Last range is 70% of first
            
            if is_contracting and ranges[-1] < 8:
                pivot = float(df['High'].tail(10).max())
                entry = pivot * 1.001
                stop = current - (atr * 1.5)
                target1 = entry + (atr * 3)
                target2 = entry + (atr * 5)
                rr = (target1 - entry) / (entry - stop)
                
                return EntryPoint(
                    entry_type='🎯 VCP 突破',
                    entry_price=round(entry, 2),
                    stop_loss=round(stop, 2),
                    target_1=round(target1, 2),
                    target_2=round(target2, 2),
                    risk_reward=round(rr, 2),
                    confidence='high',
                    notes=f"VCP 形態，緊縮度 {ranges[-1]:.1f}%，突破 ${pivot:.2f} 買入"
                )
        
        return None
    
    def _bounce_entry(self, df, current, atr, levels) -> Optional[EntryPoint]:
        """Bounce from support"""
        support = levels['nearest_support']
        
        if current <= support * 1.03 and current >= support * 0.99:
            entry = support
            stop = support - (atr * 1.5)
            target1 = entry + (atr * 2)
            target2 = levels['nearest_resistance']
            rr = (target1 - entry) / (entry - stop)
            
            return EntryPoint(
                entry_type='💚 支撐反彈',
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                target_1=round(target1, 2),
                target_2=round(target2, 2),
                risk_reward=round(rr, 2),
                confidence='medium',
                notes=f"從支撐位 ${support:.2f} 反彈"
            )
        
        return None


# ============================================
# 📈 CHART BUILDER
# ============================================
class ChartBuilder:
    """Build interactive charts"""
    
    @staticmethod
    def create_stock_chart(df: pd.DataFrame, ticker: str, entries: List[EntryPoint] = None) -> go.Figure:
        """Create comprehensive stock analysis chart"""
        
        ta = TechnicalAnalysis()
        
        # Calculate indicators
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
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{ticker} 價格走勢', 'RSI', 'MACD', '成交量')
        )
        
        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#00CC96',
                decreasing_line_color='#EF553B'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', 
                                  line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', 
                                  line=dict(color='blue', width=1)), row=1, col=1)
        if len(df) >= 200:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200', 
                                      line=dict(color='purple', width=1)), row=1, col=1)
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                                  line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                                  line=dict(color='gray', width=1, dash='dash'),
                                  fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        # Add entry points if available
        if entries:
            for entry in entries:
                # Entry line
                fig.add_hline(y=entry.entry_price, line_dash="dash", line_color="green",
                             annotation_text=f"入場 ${entry.entry_price}", row=1, col=1)
                # Stop loss line
                fig.add_hline(y=entry.stop_loss, line_dash="dash", line_color="red",
                             annotation_text=f"止損 ${entry.stop_loss}", row=1, col=1)
                # Target line
                fig.add_hline(y=entry.target_1, line_dash="dash", line_color="blue",
                             annotation_text=f"目標1 ${entry.target_1}", row=1, col=1)
        
        # 2. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                  line=dict(color='purple', width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. MACD
        colors = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
                             marker_color=colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                  line=dict(color='blue', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                                  line=dict(color='orange', width=1)), row=3, col=1)
        
        # 4. Volume
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                             marker_color=colors), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    @staticmethod
    def create_mini_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
        """Create a mini chart for quick view"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#00CC96' if df['Close'].iloc[-1] >= df['Close'].iloc[0] else '#EF553B', width=2)
        ))
        
        # SMA20
        sma20 = df['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma20,
            mode='lines',
            name='SMA20',
            line=dict(color='orange', width=1, dash='dash')
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            title=dict(text=ticker, x=0.5),
            template='plotly_dark'
        )
        
        return fig


# ============================================
# 📡 DATA FETCHER
# ============================================
class DataFetcher:
    """Fetch market data"""
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch single stock data"""
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
        """Fetch sector ETF prices"""
        tickers = [s['etf'] for s in SECTORS.values()] + [CONFIG.BENCHMARK]
        try:
            data = yf.download(tickers, period="6mo", progress=False)['Close']
            return data
        except:
            return None
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_holdings(sector_name: str):
        """Fetch sector holdings"""
        if sector_name not in SECTORS:
            return None, []
        tickers = SECTORS[sector_name]['holdings']
        try:
            data = yf.download(tickers, period="6mo", group_by='ticker', progress=False)
            return data, tickers
        except:
            return None, []
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_vix() -> Optional[Dict]:
        """Fetch VIX"""
        try:
            vix = yf.download("^VIX", period="5d", progress=False)['Close']
            if len(vix) > 0:
                return {'value': float(vix.iloc[-1]), 
                        'change': float(vix.iloc[-1] - vix.iloc[-2]) if len(vix) > 1 else 0}
        except:
            pass
        return None


# ============================================
# 📈 STOCK ANALYZER
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
    sector: str = ""


class StockAnalyzer:
    """Analyze stocks"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def analyze(self, df: pd.DataFrame, ticker: str = "") -> Optional[StockAnalysis]:
        """Analyze a stock"""
        if df is None or len(df) < 50:
            return None
        
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
            vol_ratio = float(volume.iloc[-1]) / vol_avg if vol_avg > 0 else 1
            
            # Derived
            change_pct = (curr_price - prev_price) / prev_price * 100
            extension = (curr_price - curr_sma50) / curr_sma50 * 100
            dist_high = (curr_price / high_52w - 1) * 100
            above_sma50 = curr_price > curr_sma50
            
            # Pattern detection
            is_vcp = self._detect_vcp(df)
            is_breakout = self._detect_breakout(df)
            
            # Status
            if is_breakout:
                status, code = "🚀 突破", 3
            elif is_vcp and above_sma50:
                status, code = "🎯 VCP", 2
            elif above_sma50 and vol_ratio < 0.8:
                status, code = "📊 整理", 1
            elif not above_sma50:
                status, code = "⚠️ 弱勢", -1
            else:
                status, code = "🧘 盤整", 0
            
            # Overheated
            is_overheated = extension > 25 or curr_rsi > 80
            
            return StockAnalysis(
                ticker=ticker,
                price=curr_price,
                change_pct=change_pct,
                status=status,
                status_code=code,
                extension_pct=extension,
                rsi=curr_rsi,
                adr_pct=curr_adr,
                dist_52w_high=dist_high,
                volume_ratio=vol_ratio,
                above_sma50=above_sma50,
                is_vcp=is_vcp,
                is_breakout=is_breakout,
                is_overheated=is_overheated
            )
        except:
            return None
    
    def _detect_vcp(self, df: pd.DataFrame) -> bool:
        """Detect VCP pattern"""
        if len(df) < 30:
            return False
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
        """Detect breakout"""
        if len(df) < 25:
            return False
        current = float(df['Close'].iloc[-1])
        high_20 = float(df['High'].iloc[-20:-1].max())
        vol_avg = float(df['Volume'].iloc[-20:-1].mean())
        vol_curr = float(df['Volume'].iloc[-1])
        return current > high_20 and vol_curr > vol_avg * 1.3


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    """Main application"""
    
    # Page config
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    # Header
    st.title(f"{CONFIG.PAGE_ICON} Market Structure Radar v4.0")
    st.caption("Entry Point Edition | Gil Morales + Qullamaggie Style")
    
    # VIX
    vix_data = DataFetcher.get_vix()
    if vix_data:
        col1, col2, col3 = st.columns([1, 1, 3])
        col1.metric("VIX", f"{vix_data['value']:.1f}", f"{vix_data['change']:+.1f}")
        if vix_data['value'] >= 30:
            col2.error("🔴 高風險")
        elif vix_data['value'] >= 20:
            col2.warning("🟡 警戒")
        else:
            col2.success("🟢 正常")
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌪️ 板塊輪動", 
        "🎯 狼群戰術", 
        "🔥 溫度計",
        "🏆 動能排行",
        "📊 個股分析"  # NEW!
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
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
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
                            if res:
                                results.append(res.__dict__)
                        except:
                            continue
                
                st.session_state['sector_results'] = results
                st.session_state['selected_sector'] = selected
        
        if 'sector_results' in st.session_state and st.session_state.get('selected_sector') == selected:
            results = st.session_state['sector_results']
            if results:
                df = pd.DataFrame(results)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🚀 突破", f"{df['is_breakout'].sum()}")
                col2.metric("🎯 VCP", f"{df['is_vcp'].sum()}")
                col3.metric("📈 > SMA50", f"{df['above_sma50'].sum()}/{len(df)}")
                
                st.dataframe(
                    df[['ticker', 'price', 'change_pct', 'status', 'adr_pct', 'rsi']]
                    .rename(columns={'ticker': 'Ticker', 'price': 'Price', 'change_pct': 'Change%',
                                    'status': 'Status', 'adr_pct': 'ADR%', 'rsi': 'RSI'})
                    .sort_values('Status', ascending=False)
                    .style.format({'Price': '${:.2f}', 'Change%': '{:+.2f}%', 'ADR%': '{:.1f}%', 'RSI': '{:.0f}'}),
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
            
            if heat_ratio > 50:
                st.error("🚨 極度過熱!")
            elif heat_ratio > 30:
                st.warning("⚠️ 過熱警告")
            else:
                st.success("✅ 溫度正常")
        else:
            st.warning("請先在狼群戰術掃描板塊")
    
    # ===== TAB 4: Momentum Leaders =====
    with tab4:
        st.header("🏆 動能排行")
        st.info("選擇熱門股票快速查看動能")
        
        # Quick view of hot stocks
        selected_hot = st.multiselect("選擇股票", HOT_STOCKS, default=['NVDA', 'TSLA', 'AMD'])
        
        if selected_hot:
            analyzer = StockAnalyzer()
            hot_results = []
            
            for t in selected_hot:
                df = DataFetcher.get_stock_data(t, "6mo")
                if df is not None:
                    res = analyzer.analyze(df, t)
                    if res:
                        hot_results.append(res.__dict__)
            
            if hot_results:
                df_hot = pd.DataFrame(hot_results)
                df_hot['score'] = df_hot['adr_pct'] * 2 + (100 + df_hot['dist_52w_high']) / 10
                df_hot = df_hot.sort_values('score', ascending=False)
                
                st.dataframe(
                    df_hot[['ticker', 'price', 'change_pct', 'status', 'adr_pct', 'rsi', 'dist_52w_high']]
                    .rename(columns={'ticker': 'Ticker', 'price': 'Price', 'change_pct': 'Change%',
                                    'status': 'Status', 'adr_pct': 'ADR%', 'rsi': 'RSI', 
                                    'dist_52w_high': '52W High%'})
                    .style.format({'Price': '${:.2f}', 'Change%': '{:+.2f}%', 'ADR%': '{:.1f}%', 
                                  'RSI': '{:.0f}', '52W High%': '{:+.1f}%'}),
                    use_container_width=True, hide_index=True
                )
    
    # ===== TAB 5: Stock Analysis (NEW!) =====
    with tab5:
        st.header("📊 個股深度分析")
        st.info("輸入股票代碼，獲取詳細技術分析和最佳入場點建議")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_input = st.text_input("股票代碼", value="NVDA", placeholder="輸入如 NVDA, TSLA, AAPL")
        with col2:
            period = st.selectbox("時間範圍", ["6mo", "1y", "2y"], index=1)
        
        if st.button("🔍 分析股票", type="primary"):
            ticker = ticker_input.upper().strip()
            
            with st.spinner(f"正在分析 {ticker}..."):
                df = DataFetcher.get_stock_data(ticker, period)
                
                if df is None or len(df) == 0:
                    st.error(f"無法獲取 {ticker} 的數據")
                else:
                    # Analyze
                    analyzer = StockAnalyzer()
                    ta = TechnicalAnalysis()
                    entry_calc = EntryPointCalculator()
                    chart_builder = ChartBuilder()
                    
                    analysis = analyzer.analyze(df, ticker)
                    entries = entry_calc.calculate(df)
                    levels = ta.find_support_resistance(df)
                    
                    # ===== 1. Overview Metrics =====
                    st.subheader(f"📈 {ticker} 概覽")
                    
                    if analysis:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("價格", f"${analysis.price:.2f}", f"{analysis.change_pct:+.2f}%")
                        col2.metric("狀態", analysis.status)
                        col3.metric("RSI", f"{analysis.rsi:.0f}")
                        col4.metric("ADR%", f"{analysis.adr_pct:.1f}%")
                        col5.metric("52W High", f"{analysis.dist_52w_high:+.1f}%")
                    
                    # ===== 2. Chart =====
                    st.subheader("📊 技術圖表")
                    fig = chart_builder.create_stock_chart(df, ticker, entries)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ===== 3. Key Levels =====
                    st.subheader("🎯 關鍵價位")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**📈 阻力位**")
                        st.write(f"• 52週高點: ${levels['high_52w']:.2f}")
                        st.write(f"• 20日高點: ${levels['high_20d']:.2f}")
                        st.write(f"• 最近阻力: ${levels['nearest_resistance']:.2f}")
                    
                    with col2:
                        st.markdown("**📉 支撐位**")
                        st.write(f"• SMA20: ${levels['sma20']:.2f}")
                        st.write(f"• SMA50: ${levels['sma50']:.2f}")
                        st.write(f"• 最近支撐: ${levels['nearest_support']:.2f}")
                    
                    with col3:
                        st.markdown("**📊 均線**")
                        st.write(f"• SMA20: ${levels['sma20']:.2f}")
                        st.write(f"• SMA50: ${levels['sma50']:.2f}")
                        st.write(f"• SMA200: ${levels['sma200']:.2f}")
                    
                    # ===== 4. Entry Points =====
                    st.subheader("🎯 最佳入場點建議")
                    
                    if entries:
                        for i, entry in enumerate(entries):
                            with st.expander(f"{entry.entry_type} - 信心: {entry.confidence.upper()}", expanded=(i==0)):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    | 項目 | 價格 |
                                    |------|------|
                                    | **入場價** | ${entry.entry_price} |
                                    | **止損** | ${entry.stop_loss} |
                                    | **目標1** | ${entry.target_1} |
                                    | **目標2** | ${entry.target_2} |
                                    | **風險回報比** | {entry.risk_reward}:1 |
                                    """)
                                
                                with col2:
                                    # Risk calculation
                                    risk_per_share = entry.entry_price - entry.stop_loss
                                    reward_per_share = entry.target_1 - entry.entry_price
                                    
                                    st.markdown(f"""
                                    **📝 說明:**
                                    {entry.notes}
                                    
                                    **💰 風險計算 (以 $10,000 為例):**
                                    - 風險 2%: 最多可買 {int(200 / risk_per_share)} 股
                                    - 每股風險: ${risk_per_share:.2f}
                                    - 每股潛在收益: ${reward_per_share:.2f}
                                    """)
                    else:
                        st.warning("目前沒有明確的入場信號，建議等待更好的機會")
                        st.markdown("""
                        **可能原因:**
                        - 股價不在關鍵位置附近
                        - 沒有明顯的技術形態
                        - 建議等待回調到支撐位或突破阻力位
                        """)
                    
                    # ===== 5. Trading Plan =====
                    st.subheader("📋 交易計劃建議")
                    
                    if analysis:
                        if analysis.is_breakout:
                            st.success("""
                            **🚀 突破中 - 可考慮入場**
                            1. 等待回測突破位確認
                            2. 設置止損在突破位下方
                            3. 分批止盈
                            """)
                        elif analysis.is_vcp:
                            st.info("""
                            **🎯 VCP 形態 - 準備入場**
                            1. 等待突破近期高點
                            2. 需要成交量放大確認
                            3. 止損設在 VCP 低點
                            """)
                        elif analysis.is_overheated:
                            st.error("""
                            **🔥 過熱警告 - 不建議追高**
                            1. 等待回調到均線支撐
                            2. 或等待整理形態形成
                            3. 勿在高位追入
                            """)
                        elif not analysis.above_sma50:
                            st.warning("""
                            **⚠️ 弱勢 - 觀望**
                            1. 等待站回 SMA50 之上
                            2. 或等待在支撐位形成反轉
                            3. 不建議現在入場
                            """)
                        else:
                            st.info("""
                            **🧘 盤整中 - 等待機會**
                            1. 設置突破提醒
                            2. 觀察成交量變化
                            3. 等待明確信號
                            """)
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v4.0 新功能")
    st.sidebar.markdown("""
    - ✅ **個股深度分析**
    - ✅ **最佳入場點計算**
    - ✅ **支撐/阻力位**
    - ✅ **風險回報計算**
    - ✅ **互動技術圖表**
    - ✅ **交易計劃建議**
    """)
    
    st.sidebar.divider()
    st.sidebar.info("""
    **入場類型說明:**
    - 🚀 **突破**: 價格創新高
    - 📉 **回調**: 回到均線支撐
    - 🎯 **VCP**: 波動收縮突破
    - 💚 **反彈**: 支撐位反彈
    """)


if __name__ == "__main__":
    main()
