# -*- coding: utf-8 -*-
"""
🌪️ Market Structure Radar - v3.1 (Vibe Coding Edition)
=======================================================

A clean, modular market analysis tool combining:
- Gil Morales: Sector rotation, climax detection
- Qullamaggie: ADR%, VCP, momentum breakouts

Author: AI Trading Assistant
Style: Vibe Coding - Clean, Readable, Modular

Usage:
    streamlit run market_radar_v3.py

Required packages:
    pip install streamlit yfinance pandas numpy plotly
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
    """Central configuration for the app"""
    # Page settings
    PAGE_TITLE: str = "Market Structure Radar v3.1"
    PAGE_ICON: str = "🌪️"
    
    # Data settings
    CACHE_TTL: int = 1800  # 30 minutes
    DATA_PERIOD: str = "6mo"
    BENCHMARK: str = "SPY"
    
    # Technical thresholds
    RSI_OVERBOUGHT: int = 80
    RSI_OVERSOLD: int = 30
    EXTENSION_DANGER: float = 25.0  # % above SMA50
    VCP_TIGHTNESS_THRESHOLD: float = 8.0
    BREAKOUT_VOLUME_RATIO: float = 1.3
    HIGH_ADR_THRESHOLD: float = 5.0

CONFIG = Config()


# ============================================
# 📊 SECTOR DATA
# ============================================
SECTORS: Dict[str, Dict] = {
    'SMH (半導體)': {
        'etf': 'SMH',
        'holdings': ['NVDA', 'TSM', 'AVGO', 'AMD', 'MU', 'QCOM', 'AMAT', 'LRCX', 'TXN', 'INTC', 'ADI', 'KLAC', 'MRVL', 'ARM'],
        'theme': '🔬 AI/芯片',
        'color': '#00D4FF'
    },
    'XLK (科技)': {
        'etf': 'XLK',
        'holdings': ['MSFT', 'AAPL', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'AMD', 'QCOM', 'IBM', 'NOW', 'INTU'],
        'theme': '💻 大型科技',
        'color': '#7B68EE'
    },
    'XLC (通訊)': {
        'etf': 'XLC',
        'holdings': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'TMUS', 'VZ', 'CMCSA', 'T', 'WBD', 'TTWO', 'EA'],
        'theme': '📱 社交/媒體',
        'color': '#FF6B6B'
    },
    'XLF (金融)': {
        'etf': 'XLF',
        'holdings': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'V', 'MA', 'SCHW', 'CB'],
        'theme': '🏦 銀行/保險',
        'color': '#4ECDC4'
    },
    'XLY (消費)': {
        'etf': 'XLY',
        'holdings': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'ORLY'],
        'theme': '🛒 零售/消費',
        'color': '#FFE66D'
    },
    'XLV (醫療)': {
        'etf': 'XLV',
        'holdings': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'ABT', 'PFE', 'AMGN', 'BMY', 'GILD'],
        'theme': '💊 製藥/醫療',
        'color': '#95E1D3'
    },
    'XLE (能源)': {
        'etf': 'XLE',
        'holdings': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'DVN'],
        'theme': '⛽ 石油/天然氣',
        'color': '#F38181'
    },
    'XLI (工業)': {
        'etf': 'XLI',
        'holdings': ['GE', 'CAT', 'UNP', 'HON', 'UPS', 'BA', 'RTX', 'DE', 'LMT', 'MMM'],
        'theme': '🏭 製造/國防',
        'color': '#AA96DA'
    },
    'ARKK (創新)': {
        'etf': 'ARKK',
        'holdings': ['TSLA', 'COIN', 'ROKU', 'SQ', 'PATH', 'HOOD', 'RBLX', 'U', 'DKNG', 'CRSP'],
        'theme': '🚀 顛覆創新',
        'color': '#FF9F43'
    },
    'KWEB (中概)': {
        'etf': 'KWEB',
        'holdings': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES'],
        'theme': '🇨🇳 中國互聯網',
        'color': '#EE5A24'
    },
    'IWM (小型股)': {
        'etf': 'IWM',
        'holdings': ['MSTR', 'SMCI', 'CELH', 'AFRM', 'SOFI', 'UPST', 'RIVN', 'LCID', 'PLUG', 'FSLR'],
        'theme': '📈 高成長小型',
        'color': '#A3CB38'
    }
}


# ============================================
# 🧮 TECHNICAL INDICATORS
# ============================================
class TechnicalAnalysis:
    """Technical indicator calculations"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def adr_percent(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Average Daily Range % (Qullamaggie Style)
        ADR% = average of (High - Low) / Low * 100
        """
        if 'High' not in df.columns or 'Low' not in df.columns:
            return pd.Series([0] * len(df))
        daily_range = (df['High'] / df['Low'] - 1) * 100
        return daily_range.rolling(period).mean()
    
    @staticmethod
    def extension_percent(price: float, sma: float) -> float:
        """Calculate price extension from SMA"""
        if sma == 0:
            return 0
        return (price - sma) / sma * 100


# ============================================
# 🔍 PATTERN DETECTION
# ============================================
class PatternDetector:
    """Detect chart patterns"""
    
    @staticmethod
    def detect_vcp(df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, float, str]:
        """
        Detect Volatility Contraction Pattern (VCP)
        Returns: (is_vcp, tightness, message)
        """
        if len(df) < lookback + 10:
            return False, 0, ""
        
        recent = df.tail(lookback)
        
        # Calculate range for each 5-day period
        ranges = []
        for i in range(0, lookback, 5):
            if i + 5 <= lookback:
                period = recent.iloc[i:i+5]
                if 'High' in period.columns and 'Low' in period.columns:
                    range_pct = (period['High'].max() - period['Low'].min()) / period['Low'].min() * 100
                    ranges.append(range_pct)
        
        if len(ranges) < 2:
            return False, 0, ""
        
        # Check if contracting (each range <= previous * 1.1)
        is_contracting = all(ranges[i] >= ranges[i+1] * 0.9 for i in range(len(ranges)-1))
        tightness = ranges[-1] if ranges else 100
        
        if is_contracting and tightness < CONFIG.VCP_TIGHTNESS_THRESHOLD:
            return True, tightness, f"VCP 形成 (緊縮: {tightness:.1f}%)"
        
        return False, tightness, ""
    
    @staticmethod
    def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> Tuple[bool, float, str]:
        """
        Detect price breakout with volume confirmation
        Returns: (is_breakout, volume_ratio, message)
        """
        if len(df) < lookback + 5:
            return False, 1.0, ""
        
        current_price = float(df['Close'].iloc[-1])
        prev_high = float(df['High'].iloc[-lookback:-1].max())
        
        current_vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-lookback:-1].mean())
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        if current_price > prev_high and vol_ratio > CONFIG.BREAKOUT_VOLUME_RATIO:
            return True, vol_ratio, f"🚀 突破! (量比: {vol_ratio:.1f}x)"
        elif current_price > prev_high * 0.98:
            return False, vol_ratio, f"接近突破 (量比: {vol_ratio:.1f}x)"
        
        return False, vol_ratio, ""


# ============================================
# 📈 STOCK ANALYZER
# ============================================
@dataclass
class StockAnalysis:
    """Data class for stock analysis results"""
    ticker: str
    price: float
    change_pct: float
    status: str
    status_code: int
    extension_pct: float
    rsi: float
    adr_pct: float
    dist_52w_high: float
    dist_52w_low: float
    volume_ratio: float
    above_sma50: bool
    above_sma200: bool
    is_vcp: bool
    is_breakout: bool
    is_overheated: bool
    overheat_reasons: str
    sector: str = ""


class StockAnalyzer:
    """Analyze individual stocks"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.patterns = PatternDetector()
    
    def analyze(self, df: pd.DataFrame, ticker: str = "") -> Optional[StockAnalysis]:
        """
        Perform comprehensive technical analysis on a stock
        """
        if df is None or len(df) < 50:
            return None
        
        try:
            # Extract price data
            close = df['Close']
            volume = df['Volume']
            high = df['High']
            low = df['Low']
            
            # Calculate indicators
            sma50 = self.ta.sma(close, 50)
            sma200 = self.ta.sma(close, 200) if len(close) >= 200 else sma50
            rsi = self.ta.rsi(close, 14)
            adr_pct = self.ta.adr_percent(df)
            
            # Get current values
            curr = self._get_current_values(df, sma50, sma200, rsi, adr_pct)
            
            # Detect patterns
            is_vcp, vcp_tightness, _ = self.patterns.detect_vcp(df)
            is_breakout, vol_ratio, _ = self.patterns.detect_breakout(df)
            
            # Determine status
            status, status_code = self._determine_status(
                curr, is_vcp, is_breakout, vol_ratio
            )
            
            # Check overheating
            is_overheated, overheat_reasons = self._check_overheating(curr)
            
            return StockAnalysis(
                ticker=ticker,
                price=curr['price'],
                change_pct=curr['change_pct'],
                status=status,
                status_code=status_code,
                extension_pct=curr['extension'],
                rsi=curr['rsi'],
                adr_pct=curr['adr'],
                dist_52w_high=curr['dist_high'],
                dist_52w_low=curr['dist_low'],
                volume_ratio=vol_ratio,
                above_sma50=curr['above_sma50'],
                above_sma200=curr['above_sma200'],
                is_vcp=is_vcp,
                is_breakout=is_breakout,
                is_overheated=is_overheated,
                overheat_reasons=overheat_reasons
            )
            
        except Exception as e:
            return None
    
    def _get_current_values(self, df, sma50, sma200, rsi, adr_pct) -> Dict:
        """Extract current values from dataframes"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        curr_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        curr_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else curr_price
        curr_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else curr_price
        
        # 52-week high/low
        high_52w = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
        low_52w = float(low.tail(252).min()) if len(low) >= 252 else float(low.min())
        
        return {
            'price': curr_price,
            'change_pct': (curr_price - prev_price) / prev_price * 100,
            'sma50': curr_sma50,
            'sma200': curr_sma200,
            'extension': self.ta.extension_percent(curr_price, curr_sma50),
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
            'adr': float(adr_pct.iloc[-1]) if not pd.isna(adr_pct.iloc[-1]) else 3,
            'dist_high': (curr_price / high_52w - 1) * 100,
            'dist_low': (curr_price / low_52w - 1) * 100,
            'above_sma50': curr_price > curr_sma50,
            'above_sma200': curr_price > curr_sma200
        }
    
    def _determine_status(self, curr: Dict, is_vcp: bool, is_breakout: bool, vol_ratio: float) -> Tuple[str, int]:
        """Determine stock status"""
        if is_breakout:
            return "🚀 突破", 3
        elif is_vcp and curr['above_sma50']:
            return "🎯 VCP蓄勢", 2
        elif curr['above_sma50'] and vol_ratio < 0.8:
            return "📊 收縮整理", 1
        elif not curr['above_sma50']:
            return "⚠️ 弱勢", -1
        else:
            return "🧘 盤整", 0
    
    def _check_overheating(self, curr: Dict) -> Tuple[bool, str]:
        """Check if stock is overheated (Gil Morales Style)"""
        reasons = []
        
        if curr['extension'] > CONFIG.EXTENSION_DANGER:
            reasons.append(f"乖離 {curr['extension']:.0f}%")
        
        if curr['rsi'] > CONFIG.RSI_OVERBOUGHT:
            reasons.append(f"RSI {curr['rsi']:.0f}")
        
        if curr['dist_high'] > -2 and curr['change_pct'] > 5:
            reasons.append("高潮頂")
        
        is_overheated = len(reasons) > 0
        return is_overheated, ", ".join(reasons)


# ============================================
# 📡 DATA FETCHER
# ============================================
class DataFetcher:
    """Fetch market data from Yahoo Finance"""
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_sector_etfs() -> Optional[pd.DataFrame]:
        """Fetch all sector ETF prices"""
        tickers = [s['etf'] for s in SECTORS.values()] + [CONFIG.BENCHMARK]
        try:
            data = yf.download(tickers, period=CONFIG.DATA_PERIOD, progress=False)['Close']
            return data
        except Exception as e:
            st.error(f"獲取 ETF 數據失敗: {e}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_holdings(sector_name: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Fetch holdings data for a sector"""
        if sector_name not in SECTORS:
            return None, []
        
        tickers = SECTORS[sector_name]['holdings']
        try:
            data = yf.download(tickers, period=CONFIG.DATA_PERIOD, group_by='ticker', progress=False)
            return data, tickers
        except Exception as e:
            st.error(f"獲取成分股數據失敗: {e}")
            return None, []
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def get_all_holdings() -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Fetch all holdings across all sectors"""
        all_tickers = set()
        for sector in SECTORS.values():
            all_tickers.update(sector['holdings'])
        
        try:
            data = yf.download(list(all_tickers), period=CONFIG.DATA_PERIOD, group_by='ticker', progress=False)
            return data, list(all_tickers)
        except Exception as e:
            return None, []
    
    @staticmethod
    @st.cache_data(ttl=300)  # 5 minutes for VIX
    def get_vix() -> Optional[Dict]:
        """Fetch VIX data"""
        try:
            vix = yf.download("^VIX", period="5d", progress=False)['Close']
            if len(vix) > 0:
                return {
                    'value': float(vix.iloc[-1]),
                    'change': float(vix.iloc[-1] - vix.iloc[-2]) if len(vix) > 1 else 0
                }
        except:
            pass
        return None


# ============================================
# 🖥️ UI COMPONENTS
# ============================================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def show_vix_header():
        """Display VIX in header"""
        vix_data = DataFetcher.get_vix()
        if vix_data:
            col1, col2, col3 = st.columns([1, 1, 2])
            col1.metric("VIX", f"{vix_data['value']:.1f}", f"{vix_data['change']:+.1f}")
            
            if vix_data['value'] >= 30:
                col2.error("🔴 高風險")
            elif vix_data['value'] >= 20:
                col2.warning("🟡 警戒")
            else:
                col2.success("🟢 正常")
    
    @staticmethod
    def show_sector_strength_chart(df_rs: pd.DataFrame, timeframe: int):
        """Display sector strength bar chart"""
        fig = px.bar(
            df_rs, 
            x='RS Rating', 
            y='板塊', 
            orientation='h', 
            color='RS Rating',
            color_continuous_scale=['#FF4B4B', '#F0F2F6', '#00CC96'], 
            range_color=[-15, 15],
            title=f"相對於 SPY 的強度 ({timeframe} 天)"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def show_stock_table(df: pd.DataFrame, columns: List[str]):
        """Display formatted stock table"""
        format_dict = {
            'Price': '${:.2f}',
            'Change%': '{:+.2f}%',
            'ADR%': '{:.1f}%',
            'RSI': '{:.0f}',
            'Dist_52W_High%': '{:+.1f}%',
            'Extension%': '{:.1f}%',
            'Volume_Rel': '{:.2f}x',
            'Momentum_Score': '{:.1f}'
        }
        
        # Filter format dict to only include columns in df
        active_formats = {k: v for k, v in format_dict.items() if k in columns}
        
        styled = df[columns].style
        
        # Apply status coloring if Status column exists
        if 'Status' in columns:
            styled = styled.applymap(
                lambda v: 'color: green; font-weight: bold' if '突破' in str(v) else 
                         ('color: blue' if 'VCP' in str(v) or '蓄勢' in str(v) else 
                          ('color: red' if '弱勢' in str(v) else '')),
                subset=['Status']
            )
        
        styled = styled.format(active_formats)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    
    @staticmethod
    def show_health_meter(ratio: float, label: str = "健康度"):
        """Display a health/temperature meter"""
        st.progress(min(int(ratio), 100))
        st.caption(f"{label}: {ratio:.0f}%")


# ============================================
# 📱 MAIN APPLICATION
# ============================================
class MarketRadarApp:
    """Main application class"""
    
    def __init__(self):
        self.analyzer = StockAnalyzer()
        self.data = DataFetcher()
        self.ui = UIComponents()
    
    def run(self):
        """Run the application"""
        self._setup_page()
        self._show_header()
        self._show_tabs()
        self._show_sidebar()
    
    def _setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title=CONFIG.PAGE_TITLE,
            page_icon=CONFIG.PAGE_ICON,
            layout="wide"
        )
    
    def _show_header(self):
        """Show app header"""
        st.title(f"{CONFIG.PAGE_ICON} Market Structure Radar v3.1")
        st.caption("Gil Morales & Qullamaggie Style | Vibe Coding Edition")
        self.ui.show_vix_header()
        st.divider()
    
    def _show_tabs(self):
        """Show main tabs"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌪️ 板塊戰國策", 
            "🎯 狼群戰術", 
            "🔥 溫度計", 
            "🏆 動能排行"
        ])
        
        with tab1:
            self._tab_sector_rotation()
        with tab2:
            self._tab_group_action()
        with tab3:
            self._tab_temperature()
        with tab4:
            self._tab_momentum_leaders()
    
    def _tab_sector_rotation(self):
        """Tab 1: Sector Rotation Analysis"""
        st.header("板塊相對強度 (Relative Strength)")
        
        with st.spinner("掃描市場..."):
            df_etf = self.data.get_sector_etfs()
        
        if df_etf is None or len(df_etf) == 0:
            st.error("無法獲取數據")
            return
        
        # Timeframe selector
        timeframe = st.selectbox(
            "時間軸", 
            [5, 21, 63, 126],
            format_func=lambda x: f"{x} 天" + (f" ({x//21}M)" if x >= 21 else ""),
            index=1
        )
        
        # Calculate returns
        returns = df_etf.pct_change(periods=timeframe).iloc[-1] * 100
        spy_return = returns.get(CONFIG.BENCHMARK, 0)
        
        # Build RS data
        rs_data = []
        for name, info in SECTORS.items():
            ticker = info['etf']
            if ticker in returns:
                rs_data.append({
                    '板塊': name,
                    '主題': info.get('theme', ''),
                    'RS Rating': returns[ticker] - spy_return,
                    'Return %': returns[ticker],
                    'ETF': ticker
                })
        
        if not rs_data:
            st.warning("沒有數據")
            return
        
        df_rs = pd.DataFrame(rs_data).sort_values('RS Rating', ascending=False)
        
        # Show strong vs weak sectors
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 強勢板塊")
            strong = df_rs[df_rs['RS Rating'] > 0]
            for _, row in strong.iterrows():
                st.write(f"**{row['板塊']}** {row['主題']}: RS {row['RS Rating']:+.1f}%")
        
        with col2:
            st.subheader("🔴 弱勢板塊")
            weak = df_rs[df_rs['RS Rating'] <= 0]
            for _, row in weak.iterrows():
                st.write(f"**{row['板塊']}** {row['主題']}: RS {row['RS Rating']:+.1f}%")
        
        # Chart
        self.ui.show_sector_strength_chart(df_rs, timeframe)
        
        # Recommendations
        st.markdown("### 💡 輪動建議")
        if len(strong) >= 2:
            top = strong.head(2)['板塊'].tolist()
            st.success(f"**強勢主線**: {', '.join(top)}")
        if len(weak) >= 2:
            bottom = weak.tail(2)['板塊'].tolist()
            st.warning(f"**避開**: {', '.join(bottom)}")
    
    def _tab_group_action(self):
        """Tab 2: Group Action / Wolf Pack Analysis"""
        st.header("🎯 尋找集體行動 (Group Action)")
        
        # Sector selector
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox("選擇板塊:", list(SECTORS.keys()))
        with col2:
            st.write(f"**主題**: {SECTORS[selected].get('theme', '')}")
            st.write(f"**成分股**: {len(SECTORS[selected]['holdings'])} 隻")
        
        # Scan button
        if st.button(f"🔍 掃描 {selected}", type="primary"):
            self._scan_sector(selected)
        
        # Show results if available
        if 'sector_results' in st.session_state and st.session_state.get('selected_sector') == selected:
            self._display_sector_results()
    
    def _scan_sector(self, sector_name: str):
        """Scan a sector for setups"""
        with st.spinner(f"正在分析 {sector_name}..."):
            raw_data, tickers = self.data.get_holdings(sector_name)
            
            results = []
            if raw_data is not None:
                for t in tickers:
                    try:
                        df_t = self._extract_ticker_data(raw_data, t, len(tickers))
                        if df_t is not None:
                            analysis = self.analyzer.analyze(df_t, t)
                            if analysis:
                                results.append(analysis.__dict__)
                    except:
                        continue
            
            st.session_state['sector_results'] = results
            st.session_state['selected_sector'] = sector_name
    
    def _extract_ticker_data(self, raw_data, ticker: str, total_tickers: int):
        """Extract single ticker data from multi-ticker download"""
        if total_tickers == 1:
            return raw_data
        try:
            if ticker in raw_data.columns.get_level_values(0):
                return raw_data[ticker]
        except:
            pass
        return None
    
    def _display_sector_results(self):
        """Display sector scan results"""
        results = st.session_state['sector_results']
        
        if not results:
            st.warning("沒有數據")
            return
        
        df = pd.DataFrame(results)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚀 突破", f"{df['is_breakout'].sum()} 隻")
        col2.metric("🎯 VCP", f"{df['is_vcp'].sum()} 隻")
        col3.metric("📈 > SMA50", f"{df['above_sma50'].sum()}/{len(df)}")
        col4.metric("平均 ADR%", f"{df['adr_pct'].mean():.1f}%")
        
        # Health meter
        health = df['above_sma50'].sum() / len(df) * 100
        self.ui.show_health_meter(health, "板塊健康度")
        
        # Filter
        st.markdown("### 📋 成分股清單")
        filter_opt = st.radio("篩選", ["全部", "🚀 突破", "🎯 VCP", "⚠️ 弱勢"], horizontal=True)
        
        if filter_opt == "🚀 突破":
            df_show = df[df['is_breakout'] == True]
        elif filter_opt == "🎯 VCP":
            df_show = df[df['is_vcp'] == True]
        elif filter_opt == "⚠️ 弱勢":
            df_show = df[df['above_sma50'] == False]
        else:
            df_show = df
        
        df_show = df_show.sort_values('status_code', ascending=False)
        
        # Rename columns for display
        df_display = df_show.rename(columns={
            'ticker': 'Ticker',
            'price': 'Price',
            'change_pct': 'Change%',
            'status': 'Status',
            'adr_pct': 'ADR%',
            'rsi': 'RSI',
            'dist_52w_high': 'Dist_52W_High%',
            'volume_ratio': 'Volume_Rel'
        })
        
        columns = ['Ticker', 'Price', 'Change%', 'Status', 'ADR%', 'RSI', 'Dist_52W_High%', 'Volume_Rel']
        self.ui.show_stock_table(df_display, columns)
    
    def _tab_temperature(self):
        """Tab 3: Market Temperature / Overheating Detection"""
        st.header("🔥 過熱偵測 (Climax Run)")
        
        st.info("""
        **過熱判斷邏輯 (Gil Morales Style):**
        - 乖離率 > 25% | RSI > 80 | 高潮頂 (接近高點 + 大漲)
        """)
        
        if 'sector_results' not in st.session_state or not st.session_state['sector_results']:
            st.warning("👈 請先在 **狼群戰術** 掃描一個板塊")
            return
        
        results = st.session_state['sector_results']
        sector = st.session_state.get('selected_sector', '')
        
        df = pd.DataFrame(results)
        df_hot = df[df['is_overheated'] == True]
        
        # Temperature metrics
        heat_ratio = len(df_hot) / len(df) * 100 if len(df) > 0 else 0
        avg_ext = df['extension_pct'].mean()
        avg_rsi = df['rsi'].mean()
        
        st.write(f"### {sector} 板塊溫度")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ 過熱比例", f"{heat_ratio:.0f}%")
        col2.metric("📏 平均乖離", f"{avg_ext:.1f}%")
        col3.metric("📊 平均 RSI", f"{avg_rsi:.0f}")
        
        self.ui.show_health_meter(heat_ratio, "過熱程度")
        
        # Warning levels
        if heat_ratio > 50:
            st.error("🚨 **極度過熱** - 高潮頂風險！")
        elif heat_ratio > 30:
            st.warning("⚠️ **過熱警告** - 勿追高！")
        elif heat_ratio > 15:
            st.info("📢 **溫和過熱** - 選擇性參與")
        else:
            st.success("✅ **溫度正常** - 可正常操作")
        
        # Overheated list
        st.write("### 🔥 過熱名單")
        if not df_hot.empty:
            df_hot_display = df_hot.rename(columns={
                'ticker': 'Ticker',
                'price': 'Price',
                'change_pct': 'Change%',
                'extension_pct': 'Extension%',
                'rsi': 'RSI',
                'overheat_reasons': 'Reason'
            })
            st.dataframe(
                df_hot_display[['Ticker', 'Price', 'Change%', 'Extension%', 'RSI', 'Reason']]
                .sort_values('Extension%', ascending=False)
                .style.background_gradient(subset=['Extension%', 'RSI'], cmap='Reds')
                .format({'Price': '${:.2f}', 'Change%': '{:+.2f}%', 'Extension%': '{:.1f}%', 'RSI': '{:.0f}'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("✅ 沒有過熱股票")
    
    def _tab_momentum_leaders(self):
        """Tab 4: Momentum Leaders Ranking"""
        st.header("🏆 動能排行榜")
        st.info("全市場掃描 - Qullamaggie Style")
        
        if st.button("🔍 全市場掃描", type="primary"):
            self._scan_all_sectors()
        
        if 'all_results' not in st.session_state or not st.session_state['all_results']:
            return
        
        df = pd.DataFrame(st.session_state['all_results'])
        
        # Calculate momentum score
        df['momentum_score'] = (
            df['adr_pct'] * 2 +
            (100 + df['dist_52w_high']) / 10 +
            df['above_sma50'].astype(int) * 10 +
            df['is_breakout'].astype(int) * 20 +
            df['is_vcp'].astype(int) * 15
        )
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_adr = st.slider("最低 ADR%", 0.0, 10.0, 3.0, 0.5)
        with col2:
            only_above = st.checkbox("只顯示 > SMA50", value=True)
        
        df_filtered = df[df['adr_pct'] >= min_adr]
        if only_above:
            df_filtered = df_filtered[df_filtered['above_sma50'] == True]
        
        df_filtered = df_filtered.sort_values('momentum_score', ascending=False)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("符合條件", f"{len(df_filtered)} 隻")
        col2.metric("突破中", f"{df_filtered['is_breakout'].sum()} 隻")
        col3.metric("VCP 蓄勢", f"{df_filtered['is_vcp'].sum()} 隻")
        
        # Top 20
        st.write("### 🥇 Top 20 動能股")
        top20 = df_filtered.head(20).rename(columns={
            'ticker': 'Ticker',
            'sector': 'Sector',
            'price': 'Price',
            'status': 'Status',
            'adr_pct': 'ADR%',
            'rsi': 'RSI',
            'dist_52w_high': 'Dist_52W_High%',
            'momentum_score': 'Momentum_Score'
        })
        
        columns = ['Ticker', 'Sector', 'Price', 'Status', 'ADR%', 'RSI', 'Dist_52W_High%', 'Momentum_Score']
        self.ui.show_stock_table(top20, columns)
        
        # Sector distribution
        st.write("### 📊 板塊分布")
        sector_counts = top20['Sector'].value_counts()
        fig = px.pie(values=sector_counts.values, names=sector_counts.index, title="Top 20 板塊分布")
        st.plotly_chart(fig, use_container_width=True)
    
    def _scan_all_sectors(self):
        """Scan all sectors for momentum leaders"""
        with st.spinner("掃描中 (約 1-2 分鐘)..."):
            raw_data, all_tickers = self.data.get_all_holdings()
            
            results = []
            progress = st.progress(0)
            
            if raw_data is not None:
                for idx, t in enumerate(all_tickers):
                    try:
                        if t in raw_data.columns.get_level_values(0):
                            df_t = raw_data[t]
                            analysis = self.analyzer.analyze(df_t, t)
                            if analysis:
                                # Find sector
                                for sector_name, info in SECTORS.items():
                                    if t in info['holdings']:
                                        analysis.sector = sector_name
                                        break
                                results.append(analysis.__dict__)
                    except:
                        continue
                    progress.progress((idx + 1) / len(all_tickers))
            
            progress.empty()
            st.session_state['all_results'] = results
    
    def _show_sidebar(self):
        """Show sidebar"""
        st.sidebar.divider()
        st.sidebar.markdown("### 📖 v3.1 特色")
        st.sidebar.markdown("""
        **Vibe Coding Style:**
        - ✅ 模組化設計
        - ✅ 類型提示
        - ✅ Dataclass
        - ✅ 清晰命名
        
        **功能:**
        - 🌪️ 板塊輪動
        - 🎯 VCP/突破檢測
        - 🔥 過熱偵測
        - 🏆 動能排行
        """)
        
        st.sidebar.divider()
        st.sidebar.info("""
        **Gil Morales:**
        - RS 相對強度
        - Climax 高潮頂
        
        **Qullamaggie:**
        - ADR% 動能
        - VCP 收縮形態
        """)


# ============================================
# 🚀 ENTRY POINT
# ============================================
if __name__ == "__main__":
    app = MarketRadarApp()
    app.run()
