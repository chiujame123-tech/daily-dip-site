# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v7.5 Pro Trader Edition
===================================================

重大改進：
✅ 圖表上標註 VCP/BGU 關鍵點位
✅ 優化止損位置 (更緊的止損)
✅ 期權策略建議 (賣PUT/買CALL/價差組合)
✅ A級 Setup 詳細買入理由

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
    PAGE_TITLE: str = "Market Radar v7.5 Pro"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800
    
    # 優化後的止損參數 (更緊)
    BGU_STOP_ATR_MULT: float = 0.3  # 從 0.5 改為 0.3
    VCP_STOP_ATR_MULT: float = 0.2  # 從 0.3 改為 0.2
    PP_STOP_ATR_MULT: float = 0.25

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
    'Financials': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'SCHW',
        'COIN', 'HOOD', 'SOFI', 'AFRM', 'PYPL', 'SQ', 'ICE', 'CME', 'SPGI', 'MCO'
    ],
    'Healthcare': [
        'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
        'GILD', 'VRTX', 'REGN', 'MRNA', 'ISRG', 'BSX', 'EW', 'SYK', 'MDT', 'ZTS'
    ],
    'China ADR': [
        'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES',
        'BEKE', 'FUTU', 'TIGR', 'TAL', 'EDU', 'YUMC', 'ZTO', 'QFIN', 'VIPS', 'HTHT'
    ],
    'Clean Energy & EV': [
        'TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'ENPH', 'SEDG', 'FSLR', 'RUN',
        'PLUG', 'BE', 'CHPT', 'QS', 'F', 'GM', 'BLNK', 'EVGO', 'LEA', 'ALB'
    ],
}

ALL_STOCKS = list(set([s for stocks in STOCK_UNIVERSE.values() for s in stocks]))
ALL_STOCKS.sort()

SECTORS = {
    'SMH (半導體)': {'etf': 'SMH', 'holdings': STOCK_UNIVERSE['Semiconductors'][:12]},
    'XLK (科技)': {'etf': 'XLK', 'holdings': STOCK_UNIVERSE['Software & Cloud'][:12]},
    'XLF (金融)': {'etf': 'XLF', 'holdings': STOCK_UNIVERSE['Financials'][:12]},
    'XLV (醫療)': {'etf': 'XLV', 'holdings': STOCK_UNIVERSE['Healthcare'][:12]},
    'ARKK (成長)': {'etf': 'ARKK', 'holdings': STOCK_UNIVERSE['High Growth'][:12]},
    'KWEB (中概)': {'etf': 'KWEB', 'holdings': STOCK_UNIVERSE['China ADR'][:12]},
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
        if len(df) < 200:
            return {'passed': False, 'score': 0, 'checks': []}
        
        close = float(df['Close'].iloc[-1])
        sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
        sma150 = float(df['Close'].rolling(150).mean().iloc[-1])
        sma200 = float(df['Close'].rolling(200).mean().iloc[-1])
        
        checks = []
        passed = 0
        
        if close > sma50: checks.append(('✅', 'Price > SMA50')); passed += 1
        else: checks.append(('❌', 'Price < SMA50'))
        
        if close > sma150: checks.append(('✅', 'Price > SMA150')); passed += 1
        else: checks.append(('❌', 'Price < SMA150'))
        
        if close > sma200: checks.append(('✅', 'Price > SMA200')); passed += 1
        else: checks.append(('❌', 'Price < SMA200'))
        
        if sma50 > sma150: checks.append(('✅', 'SMA50 > SMA150')); passed += 1
        else: checks.append(('❌', 'SMA50 < SMA150'))
        
        if sma150 > sma200: checks.append(('✅', 'SMA150 > SMA200')); passed += 1
        else: checks.append(('❌', 'SMA150 < SMA200'))
        
        return {'passed': passed >= 4, 'score': passed, 'total': 5, 'checks': checks}


# ============================================
# 🎯 SETUP RESULT (擴展)
# ============================================
@dataclass
class SetupResult:
    ticker: str
    setup_type: str  # 'BGU', 'VCP', 'PP'
    quality: str  # 'A+', 'A', 'B', 'C'
    score: float
    price: float
    
    # 入場/止損
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float  # 新增第三目標
    risk_reward: float
    
    # 技術指標
    rs_rating: float
    adr_percent: float
    volume_ratio: float
    
    # Setup 特定數據
    gap_percent: float = 0  # BGU
    tightness: float = 0  # VCP
    contractions: int = 0  # VCP
    
    # VCP 標註數據 (用於圖表)
    pivot_price: float = 0
    pivot_date: str = ""
    vcp_low: float = 0
    vcp_low_date: str = ""
    contraction_levels: List[Dict] = None  # 每次收縮的高低點
    
    # BGU 標註數據
    gap_day_date: str = ""
    gap_day_open: float = 0
    gap_day_low: float = 0
    gap_day_high: float = 0
    prev_close: float = 0
    
    # 解釋
    notes: str = ""
    buy_reasons: List[str] = None  # 詳細買入理由
    risk_factors: List[str] = None  # 風險因素
    
    # 期權建議
    options_strategy: Dict = None


# ============================================
# 📈 期權策略生成器
# ============================================
class OptionsStrategyGenerator:
    """根據 Setup 生成期權策略建議"""
    
    @staticmethod
    def generate_strategy(setup: SetupResult) -> Dict:
        """
        根據 Setup 類型和質量生成期權策略
        """
        price = setup.price
        entry = setup.entry_price
        stop = setup.stop_loss
        target1 = setup.target_1
        target2 = setup.target_2
        
        risk_pct = abs(entry - stop) / entry * 100
        reward_pct = (target1 - entry) / entry * 100
        
        strategies = []
        
        # ===== 策略1: 賣出認沽期權 (Cash Secured Put) =====
        # 適合: 想要在回調時買入的情況
        put_strike = round(stop * 1.02, 0)  # 略高於止損位
        
        strategies.append({
            'name': '💰 賣出認沽期權 (Cash Secured Put)',
            'description': f'在支撐位賣 PUT，收取權利金或以更低價買入',
            'strike': put_strike,
            'type': 'SELL PUT',
            'expiry': '30-45天',
            'details': f"""
**策略邏輯:**
- 賣出行使價 ${put_strike:.0f} 的認沽期權
- 如果股價保持在 ${put_strike:.0f} 以上，賺取全部權利金
- 如果被行使，以 ${put_strike:.0f} 買入股票 (接近你的止損位)

**適合情況:**
- 你願意在 ${put_strike:.0f} 買入該股票
- 預期股價會上漲或橫盤

**風險管理:**
- 確保帳戶有足夠現金支持 (${put_strike * 100:,.0f}/合約)
- 如果股價跌破 ${stop:.2f}，考慮平倉止損

**預期回報:** 年化 15-30% (視乎波動率)
""",
            'risk_level': '中等',
            'capital_required': put_strike * 100
        })
        
        # ===== 策略2: 買入認購期權 (Long Call) =====
        # 適合: 看好突破的情況
        call_strike = round(entry * 1.02, 0)  # 略高於入場價 (輕度價外)
        
        strategies.append({
            'name': '🚀 買入認購期權 (Long Call)',
            'description': f'用有限資金捕捉上漲潛力',
            'strike': call_strike,
            'type': 'BUY CALL',
            'expiry': '45-60天',
            'details': f"""
**策略邏輯:**
- 買入行使價 ${call_strike:.0f} 的認購期權
- 最大虧損 = 權利金 (有限)
- 潛在收益 = 無限 (股價上漲越多賺越多)

**選擇建議:**
- 行使價: ${call_strike:.0f} (輕度價外)
- 到期日: 45-60天 (給時間讓 Setup 發展)
- Delta: 0.40-0.50

**盈虧平衡:**
- 需要股價漲到 ${call_strike:.0f} + 權利金

**風險管理:**
- 投入不超過帳戶 5% 買期權
- 設定心理止損: 權利金虧損 50% 離場
""",
            'risk_level': '較高',
            'capital_required': price * 5  # 估計權利金
        })
        
        # ===== 策略3: 牛市認沽價差 (Bull Put Spread) =====
        # 適合: 想要限制風險的賣 PUT
        short_put = round(entry * 0.95, 0)
        long_put = round(stop * 0.98, 0)
        
        strategies.append({
            'name': '📊 牛市認沽價差 (Bull Put Spread)',
            'description': f'有限風險的方向性策略',
            'strike': f"${short_put:.0f} / ${long_put:.0f}",
            'type': 'SPREAD',
            'expiry': '30-45天',
            'details': f"""
**策略邏輯:**
- 賣出 ${short_put:.0f} PUT (收取權利金)
- 買入 ${long_put:.0f} PUT (保護下行)
- 淨收入 = 收到的權利金差額

**最大收益:** 淨權利金收入
**最大虧損:** (${short_put:.0f} - ${long_put:.0f}) × 100 - 淨權利金

**適合情況:**
- 看漲但想限制風險
- 預期股價會保持在 ${short_put:.0f} 以上

**優點:**
- 風險有限，明確知道最大虧損
- 不需要大量資金
- 時間衰減對你有利
""",
            'risk_level': '中低',
            'capital_required': (short_put - long_put) * 100
        })
        
        # ===== 策略4: 買入認購價差 (Bull Call Spread) =====
        # 適合: 想要降低買 CALL 成本
        long_call = round(entry, 0)
        short_call = round(target1, 0)
        
        strategies.append({
            'name': '📈 牛市認購價差 (Bull Call Spread)',
            'description': f'降低成本的看漲策略',
            'strike': f"${long_call:.0f} / ${short_call:.0f}",
            'type': 'SPREAD',
            'expiry': '45-60天',
            'details': f"""
**策略邏輯:**
- 買入 ${long_call:.0f} CALL
- 賣出 ${short_call:.0f} CALL (降低成本)
- 淨支出 = 買入權利金 - 賣出權利金

**最大收益:** (${short_call:.0f} - ${long_call:.0f}) × 100 - 淨支出
**最大虧損:** 淨支出

**盈虧平衡:** ${long_call:.0f} + 淨支出

**適合情況:**
- 看漲但目標明確 (T1 = ${target1:.2f})
- 想要降低期權成本

**優點:**
- 成本比單純買 CALL 低
- 風險有限
""",
            'risk_level': '中等',
            'capital_required': (short_call - long_call) * 20  # 估計淨支出
        })
        
        # 根據 Setup 類型推薦最佳策略
        if setup.setup_type == 'VCP':
            # VCP 突破前適合賣 PUT
            recommended = 0 if setup.quality in ['A+', 'A'] else 2
        elif setup.setup_type == 'BGU':
            # BGU 已經突破，適合買 CALL 或價差
            recommended = 1 if risk_pct < 5 else 3
        else:
            # Power Play 適合價差
            recommended = 2
        
        return {
            'strategies': strategies,
            'recommended_index': recommended,
            'summary': f"""
### 期權策略總結

**當前股價:** ${price:.2f}
**入場價:** ${entry:.2f}
**止損價:** ${stop:.2f}
**風險:** {risk_pct:.1f}%
**目標收益:** {reward_pct:.1f}%

**推薦策略:** {strategies[recommended]['name']}

**選擇依據:**
- Setup 類型: {setup.setup_type}
- Setup 質量: {setup.quality}
- 風險回報比: {setup.risk_reward}:1
"""
        }


# ============================================
# 🎯 SETUP SCANNER (改進版)
# ============================================
class SetupScanner:
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def scan_bgu(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None, lookback_days: int = 5) -> Optional[SetupResult]:
        """掃描 BGU - 改進版，包含圖表標註數據"""
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
                
                if gap_percent < 3.0:
                    continue
                
                avg_volume = float(df['Volume'].iloc[:-lookback_days].tail(50).mean())
                volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio < 1.5:
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
                buy_reasons = []
                risk_factors = []
                
                if gap_percent >= 8:
                    score += 30
                    notes.append(f"強跳空{gap_percent:.1f}%")
                    buy_reasons.append(f"🚀 強勁跳空 {gap_percent:.1f}% - 代表機構大量買入")
                elif gap_percent >= 5:
                    score += 25
                    notes.append(f"跳空{gap_percent:.1f}%")
                    buy_reasons.append(f"📈 良好跳空 {gap_percent:.1f}% - 明確的買入信號")
                else:
                    score += 15
                    buy_reasons.append(f"⚠️ 跳空 {gap_percent:.1f}% - 力度一般")
                
                if volume_ratio >= 3:
                    score += 25
                    notes.append(f"爆量{volume_ratio:.1f}x")
                    buy_reasons.append(f"🔥 成交量爆發 {volume_ratio:.1f}x - 機構資金參與")
                elif volume_ratio >= 2:
                    score += 20
                    buy_reasons.append(f"📊 放量 {volume_ratio:.1f}x - 有資金支持")
                else:
                    score += 10
                    risk_factors.append(f"量能一般 {volume_ratio:.1f}x")
                
                if close_position >= 0.8:
                    score += 20
                    notes.append("收強")
                    buy_reasons.append(f"💪 收盤極強 ({close_position*100:.0f}%) - 買盤持續到收盤")
                elif close_position >= 0.6:
                    score += 15
                    buy_reasons.append(f"✅ 收盤偏強 ({close_position*100:.0f}%)")
                else:
                    score += 8
                    risk_factors.append(f"收盤位置偏低 ({close_position*100:.0f}%)")
                
                if rs >= 90:
                    score += 15
                    notes.append(f"RS{rs:.0f}")
                    buy_reasons.append(f"⭐ RS Rating 極強 ({rs:.0f}) - 市場最強股之一")
                elif rs >= 80:
                    score += 12
                    buy_reasons.append(f"✅ RS Rating 強勢 ({rs:.0f})")
                elif rs >= 70:
                    score += 8
                else:
                    score -= 5
                    risk_factors.append(f"RS Rating 偏弱 ({rs:.0f})")
                
                if above_mas:
                    score += 10
                    buy_reasons.append("✅ 價格在所有均線之上 - 趨勢健康")
                else:
                    score -= 10
                    risk_factors.append("⚠️ 價格在均線之下")
                
                if day_offset > 0:
                    penalty = day_offset * 5
                    score -= penalty
                    risk_factors.append(f"⏰ {day_offset}天前的跳空，最佳時機已過")
                
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
                    
                    atr_val = float(self.ta.atr(df).iloc[idx])
                    current_price = float(df['Close'].iloc[-1])
                    
                    # 改進的止損計算 (更緊)
                    entry = today_low  # 回調到跳空日低點入場
                    stop = today_low - atr_val * CONFIG.BGU_STOP_ATR_MULT  # 更緊的止損
                    
                    # 確保止損不會太遠
                    max_stop_distance = entry * 0.05  # 最大 5% 止損
                    if entry - stop > max_stop_distance:
                        stop = entry - max_stop_distance
                    
                    risk = entry - stop
                    target_1 = entry + risk * 2  # 2R
                    target_2 = entry + risk * 3  # 3R
                    target_3 = entry + risk * 5  # 5R
                    
                    rr = (target_1 - entry) / risk if risk > 0 else 0
                    
                    gap_day_date = df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else str(df.index[idx])
                    
                    best_bgu = SetupResult(
                        ticker=ticker, setup_type='BGU', quality=quality, score=score,
                        price=current_price,
                        entry_price=round(entry, 2), stop_loss=round(stop, 2),
                        target_1=round(target_1, 2), target_2=round(target_2, 2), target_3=round(target_3, 2),
                        risk_reward=round(rr, 2),
                        rs_rating=rs, adr_percent=adr, volume_ratio=volume_ratio,
                        gap_percent=gap_percent,
                        gap_day_date=gap_day_date,
                        gap_day_open=today_open,
                        gap_day_low=today_low,
                        gap_day_high=today_high,
                        prev_close=yesterday_close,
                        notes=" | ".join(notes),
                        buy_reasons=buy_reasons,
                        risk_factors=risk_factors
                    )
            
            return best_bgu
            
        except:
            return None
    
    def scan_vcp(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[SetupResult]:
        """掃描 VCP - 改進版，包含收縮層級數據用於圖表標註"""
        if df is None or len(df) < 100:
            return None
        
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            curr_price = float(close.iloc[-1])
            
            sma50 = close.rolling(50).mean()
            curr_sma50 = float(sma50.iloc[-1])
            above_sma50 = curr_price > curr_sma50 * 0.98
            
            if not above_sma50:
                return None
            
            # 計算收縮 - 保存每個收縮的詳細數據
            recent = df.tail(60)
            contraction_levels = []
            
            for i in range(0, min(50, len(recent)-5), 5):
                week = recent.iloc[i:i+5]
                week_high = float(week['High'].max())
                week_low = float(week['Low'].min())
                week_range = (week_high - week_low) / week_low * 100
                
                # 找到高點和低點的日期
                high_idx = week['High'].idxmax()
                low_idx = week['Low'].idxmin()
                
                contraction_levels.append({
                    'high': week_high,
                    'low': week_low,
                    'range': week_range,
                    'high_date': str(high_idx)[:10] if high_idx else '',
                    'low_date': str(low_idx)[:10] if low_idx else ''
                })
            
            if len(contraction_levels) < 3:
                return None
            
            contraction_count = sum(1 for i in range(1, len(contraction_levels)) 
                                   if contraction_levels[i]['range'] < contraction_levels[i-1]['range'] * 1.1)
            
            if contraction_count < 2:
                return None
            
            final_tightness = contraction_levels[-1]['range']
            
            if final_tightness > 12.0:
                return None
            
            # Pivot 和 VCP Low
            pivot_price = float(recent['High'].max())
            pivot_idx = recent['High'].idxmax()
            pivot_date = str(pivot_idx)[:10] if pivot_idx else ''
            
            vcp_low = float(recent['Low'].min())
            vcp_low_idx = recent['Low'].idxmin()
            vcp_low_date = str(vcp_low_idx)[:10] if vcp_low_idx else ''
            
            rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            adr = float(self.ta.adr_percent(df).iloc[-1])
            vol_ratio = float(volume.iloc[-1]) / float(volume.tail(50).mean())
            
            # 計算分數
            score = 0
            notes = []
            buy_reasons = []
            risk_factors = []
            
            if final_tightness <= 5:
                score += 30
                notes.append(f"極緊{final_tightness:.1f}%")
                buy_reasons.append(f"🎯 極度緊縮 {final_tightness:.1f}% - 賣壓幾乎消失，隨時可能爆發")
            elif final_tightness <= 8:
                score += 25
                notes.append(f"緊縮{final_tightness:.1f}%")
                buy_reasons.append(f"✅ 良好緊縮 {final_tightness:.1f}% - 籌碼集中")
            else:
                score += 15
                buy_reasons.append(f"⚠️ 緊縮度 {final_tightness:.1f}% - 還可以更緊")
            
            if contraction_count >= 4:
                score += 20
                notes.append(f"{contraction_count}次收縮")
                buy_reasons.append(f"💪 {contraction_count} 次波動收縮 - 多次洗盤，籌碼非常穩定")
            elif contraction_count >= 3:
                score += 15
                buy_reasons.append(f"✅ {contraction_count} 次收縮 - 標準 VCP 形態")
            else:
                score += 10
                risk_factors.append(f"收縮次數偏少 ({contraction_count}次)")
            
            # 成交量萎縮
            vol_early = np.mean([c['range'] for c in contraction_levels[:2]])
            vol_late = np.mean([c['range'] for c in contraction_levels[-2:]])
            if vol_late < vol_early * 0.7:
                score += 15
                notes.append("量縮")
                buy_reasons.append("📉 成交量萎縮 - 賣壓減少，突破更容易")
            else:
                score += 5
            
            if rs >= 90:
                score += 20
                notes.append(f"RS{rs:.0f}")
                buy_reasons.append(f"⭐ RS Rating 極強 ({rs:.0f}) - 領漲股票")
            elif rs >= 80:
                score += 15
                buy_reasons.append(f"✅ RS Rating 強勢 ({rs:.0f})")
            elif rs >= 70:
                score += 10
            else:
                score -= 5
                risk_factors.append(f"RS Rating 偏弱 ({rs:.0f})")
            
            # 距離 Pivot
            dist_pivot = (pivot_price - curr_price) / curr_price * 100
            if dist_pivot <= 3:
                score += 15
                notes.append("近突破")
                buy_reasons.append(f"🔥 距離突破點僅 {dist_pivot:.1f}% - 隨時可能突破")
            elif dist_pivot <= 5:
                score += 10
                buy_reasons.append(f"✅ 距離突破點 {dist_pivot:.1f}%")
            else:
                score += 5
                risk_factors.append(f"距離突破點較遠 ({dist_pivot:.1f}%)")
            
            score += 10  # 基礎分 (已在 SMA50 之上)
            buy_reasons.append("✅ 價格在 SMA50 之上 - Stage 2 上升趨勢")
            
            if score >= 85:
                quality = 'A+'
            elif score >= 70:
                quality = 'A'
            elif score >= 55:
                quality = 'B'
            else:
                quality = 'C'
            
            atr_val = float(self.ta.atr(df).iloc[-1])
            
            # 改進的止損計算 (更緊)
            entry = pivot_price * 1.001
            
            # 止損: VCP 低點或最近收縮低點，取較近者
            recent_contraction_low = contraction_levels[-1]['low']
            stop = max(vcp_low, recent_contraction_low) - atr_val * CONFIG.VCP_STOP_ATR_MULT
            
            # 確保止損不會太遠
            max_stop_distance = entry * 0.06  # 最大 6% 止損
            if entry - stop > max_stop_distance:
                stop = entry - max_stop_distance
            
            risk = entry - stop
            target_1 = entry + risk * 2
            target_2 = entry + risk * 3
            target_3 = entry + risk * 5
            
            rr = (target_1 - entry) / risk if risk > 0 else 0
            
            return SetupResult(
                ticker=ticker, setup_type='VCP', quality=quality, score=score,
                price=curr_price,
                entry_price=round(entry, 2), stop_loss=round(stop, 2),
                target_1=round(target_1, 2), target_2=round(target_2, 2), target_3=round(target_3, 2),
                risk_reward=round(rr, 2),
                rs_rating=rs, adr_percent=adr, volume_ratio=vol_ratio,
                tightness=final_tightness,
                contractions=contraction_count,
                pivot_price=pivot_price,
                pivot_date=pivot_date,
                vcp_low=vcp_low,
                vcp_low_date=vcp_low_date,
                contraction_levels=contraction_levels,
                notes=" | ".join(notes),
                buy_reasons=buy_reasons,
                risk_factors=risk_factors
            )
            
        except:
            return None
    
    def scan_all(self, stocks: List[str], spy_df: pd.DataFrame = None, 
                 progress_callback=None) -> Tuple[List[SetupResult], List[SetupResult]]:
        """掃描所有股票"""
        bgu_results = []
        vcp_results = []
        
        for i, ticker in enumerate(stocks):
            if progress_callback:
                progress_callback(i, len(stocks), ticker)
            
            try:
                df = yf.download(ticker, period='6mo', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is None or len(df) < 50:
                    continue
                
                bgu = self.scan_bgu(df, ticker, spy_df, lookback_days=5)
                if bgu and bgu.score >= 45:
                    # 生成期權策略
                    bgu.options_strategy = OptionsStrategyGenerator.generate_strategy(bgu)
                    bgu_results.append(bgu)
                
                vcp = self.scan_vcp(df, ticker, spy_df)
                if vcp and vcp.score >= 45:
                    vcp.options_strategy = OptionsStrategyGenerator.generate_strategy(vcp)
                    vcp_results.append(vcp)
                    
            except:
                continue
        
        bgu_results.sort(key=lambda x: x.score, reverse=True)
        vcp_results.sort(key=lambda x: x.score, reverse=True)
        
        return bgu_results, vcp_results


# ============================================
# 📊 CHART BUILDER (大幅改進 - 加入標註)
# ============================================
class ChartBuilder:
    """改進版圖表 - 顯示 Setup 關鍵點位"""
    
    @staticmethod
    def create_annotated_chart(df: pd.DataFrame, ticker: str, setup: SetupResult = None) -> go.Figure:
        """創建帶有 Setup 標註的圖表"""
        
        df = df.copy()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        # 創建圖表
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.75, 0.25],
            subplot_titles=(f'{ticker} - {setup.setup_type} ({setup.quality})' if setup else ticker, 'Volume')
        )
        
        # K線圖
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price',
            increasing_line_color='#00CC96', decreasing_line_color='#EF553B'
        ), row=1, col=1)
        
        # 均線
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA10'], name='SMA10',
                                 line=dict(color='yellow', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                                 line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                                 line=dict(color='blue', width=1.5)), row=1, col=1)
        
        # ===== VCP 標註 =====
        if setup and setup.setup_type == 'VCP':
            # 標註 Pivot 高點
            if setup.pivot_price > 0:
                fig.add_hline(
                    y=setup.pivot_price, 
                    line_dash="dash", 
                    line_color="cyan",
                    annotation_text=f"📍 Pivot ${setup.pivot_price:.2f}",
                    annotation_position="right",
                    row=1, col=1
                )
            
            # 標註 VCP 低點
            if setup.vcp_low > 0:
                fig.add_hline(
                    y=setup.vcp_low, 
                    line_dash="dot", 
                    line_color="yellow",
                    annotation_text=f"📍 VCP Low ${setup.vcp_low:.2f}",
                    annotation_position="right",
                    row=1, col=1
                )
            
            # 標註每次收縮區間 (用矩形)
            if setup.contraction_levels:
                colors = ['rgba(255,255,0,0.1)', 'rgba(0,255,255,0.1)', 'rgba(255,0,255,0.1)', 
                         'rgba(0,255,0,0.1)', 'rgba(255,165,0,0.1)']
                
                for i, level in enumerate(setup.contraction_levels[-4:]):  # 最後4個收縮
                    color = colors[i % len(colors)]
                    
                    # 添加收縮區間標籤
                    fig.add_annotation(
                        x=df.index[-30 + i*7] if len(df) > 30 else df.index[i*7],
                        y=level['high'],
                        text=f"T{i+1}: {level['range']:.1f}%",
                        showarrow=False,
                        font=dict(color='white', size=10),
                        bgcolor=color.replace('0.1', '0.7')
                    )
        
        # ===== BGU 標註 =====
        if setup and setup.setup_type == 'BGU':
            # 標註跳空日
            if setup.gap_day_date:
                # 跳空缺口區域
                fig.add_shape(
                    type="rect",
                    x0=setup.gap_day_date,
                    x1=setup.gap_day_date,
                    y0=setup.prev_close,
                    y1=setup.gap_day_open,
                    fillcolor="rgba(0, 255, 0, 0.3)",
                    line=dict(width=0),
                    row=1, col=1
                )
                
                # 跳空日高點
                fig.add_hline(
                    y=setup.gap_day_high, 
                    line_dash="dot", 
                    line_color="lime",
                    annotation_text=f"📍 Gap High ${setup.gap_day_high:.2f}",
                    row=1, col=1
                )
                
                # 跳空日低點 (入場點)
                fig.add_hline(
                    y=setup.gap_day_low, 
                    line_dash="dash", 
                    line_color="cyan",
                    annotation_text=f"📍 Gap Low (Entry) ${setup.gap_day_low:.2f}",
                    row=1, col=1
                )
                
                # 前一日收盤
                fig.add_hline(
                    y=setup.prev_close, 
                    line_dash="dot", 
                    line_color="gray",
                    annotation_text=f"Prev Close ${setup.prev_close:.2f}",
                    row=1, col=1
                )
                
                # 跳空標籤
                fig.add_annotation(
                    x=setup.gap_day_date,
                    y=setup.gap_day_high * 1.02,
                    text=f"🚀 GAP +{setup.gap_percent:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="lime",
                    font=dict(color='lime', size=12, family='Arial Black'),
                    bgcolor="rgba(0,0,0,0.7)"
                )
        
        # ===== 交易計劃標註 =====
        if setup:
            # 入場線
            fig.add_hline(
                y=setup.entry_price, 
                line_dash="dash", 
                line_color="green",
                line_width=2,
                annotation_text=f"🎯 Entry ${setup.entry_price}",
                annotation_position="left",
                row=1, col=1
            )
            
            # 止損線
            fig.add_hline(
                y=setup.stop_loss, 
                line_dash="dash", 
                line_color="red",
                line_width=2,
                annotation_text=f"🛑 Stop ${setup.stop_loss}",
                annotation_position="left",
                row=1, col=1
            )
            
            # 目標線
            fig.add_hline(
                y=setup.target_1, 
                line_dash="dot", 
                line_color="#00BFFF",
                annotation_text=f"T1 ${setup.target_1} (2R)",
                annotation_position="left",
                row=1, col=1
            )
            
            fig.add_hline(
                y=setup.target_2, 
                line_dash="dot", 
                line_color="#1E90FF",
                annotation_text=f"T2 ${setup.target_2} (3R)",
                annotation_position="left",
                row=1, col=1
            )
        
        # 成交量
        colors = ['#00CC96' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#EF553B' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors,
                            name='Volume', showlegend=False), row=2, col=1)
        
        # 平均成交量線
        avg_vol = df['Volume'].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=avg_vol, name='Avg Vol',
                                 line=dict(color='yellow', width=1, dash='dash')), row=2, col=1)
        
        # 布局
        fig.update_layout(
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            margin=dict(l=60, r=60, t=80, b=40)
        )
        
        # 隱藏非交易日
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])]
        )
        
        return fig


# ============================================
# 🌡️ MARKET REGIME
# ============================================
class MarketRegime:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_health() -> Dict:
        default = {'status': '❓', 'score': 50, 'vix': None, 'spy_price': None, 'advice': ''}
        
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
                status, advice = "🟡 謹慎", "正常交易"
            elif score >= 40:
                status, advice = "🟠 震盪", "減少倉位"
            else:
                status, advice = "🔴 弱勢", "防守"
            
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
            'risk_per_share': risk_per_share,
            'max_loss': shares * risk_per_share
        }


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v7.5 Pro")
    st.caption("視覺化 Setup 標註 | 優化止損 | 期權策略 | 詳細買入理由")
    
    # Market Health
    market = MarketRegime.get_health()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("市場狀態", market['status'])
    col2.metric("健康評分", f"{market['score']}/100")
    col3.metric("VIX", f"{market['vix']:.1f}" if market['vix'] else "N/A")
    col4.metric("SPY", f"${market['spy_price']:.2f}" if market['spy_price'] else "N/A")
    col5.metric("建議", market['advice'])
    
    st.divider()
    
    # Tabs
    tabs = st.tabs([
        "🌪️ 板塊輪動",
        "📊 個股分析",
        "🎯 Setup 獵人",
        "💰 倉位計算"
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
        st.header("📊 個股深度分析")
        
        ticker = st.text_input("股票代碼", value="NVDA").upper()
        
        if st.button("🔍 分析股票", type="primary", key="analyze"):
            df = DataFetcher.get_stock(ticker, "1y")
            spy_df = DataFetcher.get_stock('SPY', '1y')
            
            if df is not None:
                scanner = SetupScanner()
                ta = TechnicalAnalysis()
                
                bgu = scanner.scan_bgu(df, ticker, spy_df)
                vcp = scanner.scan_vcp(df, ticker, spy_df)
                
                # 選擇最佳 Setup
                setup = None
                if bgu and vcp:
                    setup = bgu if bgu.score > vcp.score else vcp
                elif bgu:
                    setup = bgu
                elif vcp:
                    setup = vcp
                
                # 生成期權策略
                if setup:
                    setup.options_strategy = OptionsStrategyGenerator.generate_strategy(setup)
                
                # 基本信息
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("價格", f"${float(df['Close'].iloc[-1]):.2f}")
                col2.metric("RS Rating", f"{ta.rs_rating(df, spy_df):.0f}")
                col3.metric("ADR%", f"{float(ta.adr_percent(df).iloc[-1]):.1f}%")
                trend = ta.check_trend_template(df)
                col4.metric("趨勢模板", f"{trend['score']}/{trend['total']}")
                
                # Setup 狀態
                if setup:
                    if setup.quality in ['A+', 'A']:
                        st.success(f"✅ **{setup.setup_type} - {setup.quality} 級 Setup** (Score: {setup.score:.0f})")
                    else:
                        st.warning(f"⚠️ **{setup.setup_type} - {setup.quality} 級 Setup** (Score: {setup.score:.0f})")
                else:
                    st.info("未發現明確的 Setup 信號")
                
                # ===== 圖表 (帶標註) =====
                st.subheader("📈 技術圖表")
                fig = ChartBuilder.create_annotated_chart(df, ticker, setup)
                st.plotly_chart(fig, use_container_width=True)
                
                # ===== A/A+ 級 Setup 詳細解釋 =====
                if setup and setup.quality in ['A+', 'A']:
                    st.subheader(f"🎯 為什麼 {ticker} 值得買入？")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ✅ 買入理由")
                        if setup.buy_reasons:
                            for reason in setup.buy_reasons:
                                st.markdown(f"- {reason}")
                    
                    with col2:
                        st.markdown("### ⚠️ 風險因素")
                        if setup.risk_factors:
                            for risk in setup.risk_factors:
                                st.markdown(f"- {risk}")
                        else:
                            st.markdown("- 暫無明顯風險")
                    
                    # 交易計劃
                    st.subheader("📋 交易計劃")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_pct = (setup.entry_price - setup.stop_loss) / setup.entry_price * 100
                        st.markdown(f"""
                        **入場策略:**
                        - 入場價: **${setup.entry_price}**
                        - 止損價: **${setup.stop_loss}**
                        - 風險: **{risk_pct:.1f}%**
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **目標價位:**
                        - T1 (2R): **${setup.target_1}**
                        - T2 (3R): **${setup.target_2}**
                        - T3 (5R): **${setup.target_3}**
                        """)
                    
                    with col3:
                        st.markdown(f"""
                        **風險回報:**
                        - R:R = **{setup.risk_reward}:1**
                        - 質量: **{setup.quality}**
                        - 評分: **{setup.score:.0f}**
                        """)
                    
                    # 倉位建議
                    st.subheader("💰 倉位建議 ($100,000 帳戶)")
                    pos = PositionCalculator.calculate(100000, setup.entry_price, setup.stop_loss)
                    if 'error' not in pos:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("建議股數", f"{pos['shares']}")
                        col2.metric("倉位金額", f"${pos['position_value']:,.0f}")
                        col3.metric("倉位比例", f"{pos['position_pct']:.1f}%")
                        col4.metric("最大虧損", f"${pos['max_loss']:,.0f}")
                    
                    # ===== 期權策略 =====
                    st.subheader("📊 期權策略建議")
                    
                    if setup.options_strategy:
                        strategies = setup.options_strategy['strategies']
                        recommended = setup.options_strategy['recommended_index']
                        
                        st.info(f"**推薦策略:** {strategies[recommended]['name']}")
                        
                        for i, strategy in enumerate(strategies):
                            is_recommended = (i == recommended)
                            expander_title = f"{'⭐ ' if is_recommended else ''}{strategy['name']} {'(推薦)' if is_recommended else ''}"
                            
                            with st.expander(expander_title, expanded=is_recommended):
                                st.markdown(strategy['details'])
                                
                                col1, col2 = st.columns(2)
                                col1.metric("風險等級", strategy['risk_level'])
                                col2.metric("所需資金", f"${strategy['capital_required']:,.0f}")
                
                # 非 A 級 Setup
                elif setup:
                    st.subheader("📋 交易計劃")
                    st.warning(f"這是 {setup.quality} 級 Setup，建議等待更好的機會或僅用小倉位測試")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        | 項目 | 價格 |
                        |------|------|
                        | 入場 | ${setup.entry_price} |
                        | 止損 | ${setup.stop_loss} |
                        | T1 | ${setup.target_1} |
                        | R:R | {setup.risk_reward}:1 |
                        """)
    
    # ===== TAB 3: Setup Hunter =====
    with tabs[2]:
        st.header("🎯 Setup 獵人")
        
        st.info(f"掃描 {len(ALL_STOCKS)}+ 股票，尋找 A/A+ 級 Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            scan_scope = st.selectbox(
                "掃描範圍",
                [
                    "🔥 熱門領導股 (20隻)",
                    "🔬 半導體 (20隻)",
                    "💻 軟件雲端 (20隻)",
                    "🚀 高成長股 (20隻)",
                    "🏦 金融股 (20隻)",
                    "💊 醫療股 (20隻)",
                    "🇨🇳 中概股 (20隻)",
                    "⚡ 新能源/EV (20隻)",
                    "📊 全部股票 (150+)"
                ],
                key="scan_scope"
            )
        with col2:
            min_quality = st.selectbox(
                "最低質量",
                ["全部", "只看 A+ 和 A", "只看 A+"],
                key="min_quality"
            )
        
        if st.button("🎯 開始掃描", type="primary", key="scan_all"):
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
            else:
                stocks = ALL_STOCKS
            
            spy_df = DataFetcher.get_stock('SPY', '6mo')
            scanner = SetupScanner()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(i, total, ticker):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"掃描 {ticker} ({i+1}/{total})...")
            
            bgu_results, vcp_results = scanner.scan_all(stocks, spy_df, update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            # 過濾質量
            if "只看 A+" in min_quality:
                bgu_results = [r for r in bgu_results if r.quality == 'A+']
                vcp_results = [r for r in vcp_results if r.quality == 'A+']
            elif "只看 A+ 和 A" in min_quality:
                bgu_results = [r for r in bgu_results if r.quality in ['A+', 'A']]
                vcp_results = [r for r in vcp_results if r.quality in ['A+', 'A']]
            
            st.session_state['bgu_results'] = bgu_results
            st.session_state['vcp_results'] = vcp_results
        
        # 顯示結果
        if 'bgu_results' in st.session_state:
            bgu_results = st.session_state.get('bgu_results', [])
            vcp_results = st.session_state.get('vcp_results', [])
            
            col1, col2 = st.columns(2)
            col1.metric("🚀 BGU 發現", len(bgu_results))
            col2.metric("🎯 VCP 發現", len(vcp_results))
            
            # BGU Results
            if bgu_results:
                st.markdown("### 🚀 BGU (跳空突破)")
                
                for setup in bgu_results[:6]:
                    quality_emoji = "⭐" if setup.quality == 'A+' else "✅" if setup.quality == 'A' else "⚠️"
                    
                    with st.expander(f"{quality_emoji} **{setup.ticker}** | {setup.quality} | {setup.score:.0f}分 | 跳空 {setup.gap_percent:.1f}%"):
                        
                        # 買入理由
                        if setup.quality in ['A+', 'A'] and setup.buy_reasons:
                            st.markdown("**✅ 為什麼值得買入:**")
                            for reason in setup.buy_reasons[:5]:
                                st.markdown(f"- {reason}")
                            st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            **交易計劃:**
                            - 入場: ${setup.entry_price}
                            - 止損: ${setup.stop_loss}
                            - T1: ${setup.target_1} (2R)
                            - T2: ${setup.target_2} (3R)
                            - R:R: {setup.risk_reward}:1
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **技術數據:**
                            - RS Rating: {setup.rs_rating:.0f}
                            - ADR%: {setup.adr_percent:.1f}%
                            - 量比: {setup.volume_ratio:.1f}x
                            """)
                        
                        # 圖表按鈕
                        if st.button(f"📈 查看 {setup.ticker} 圖表", key=f"bgu_chart_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "6mo")
                            if df is not None:
                                fig = ChartBuilder.create_annotated_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # 期權策略
                        if setup.options_strategy and setup.quality in ['A+', 'A']:
                            with st.expander("📊 期權策略"):
                                rec_idx = setup.options_strategy['recommended_index']
                                rec_strategy = setup.options_strategy['strategies'][rec_idx]
                                st.markdown(f"**推薦:** {rec_strategy['name']}")
                                st.markdown(rec_strategy['details'])
            
            # VCP Results
            if vcp_results:
                st.markdown("### 🎯 VCP (波動收縮)")
                
                for setup in vcp_results[:6]:
                    quality_emoji = "⭐" if setup.quality == 'A+' else "✅" if setup.quality == 'A' else "⚠️"
                    
                    with st.expander(f"{quality_emoji} **{setup.ticker}** | {setup.quality} | {setup.score:.0f}分 | 緊縮 {setup.tightness:.1f}%"):
                        
                        # 買入理由
                        if setup.quality in ['A+', 'A'] and setup.buy_reasons:
                            st.markdown("**✅ 為什麼值得買入:**")
                            for reason in setup.buy_reasons[:5]:
                                st.markdown(f"- {reason}")
                            st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            **交易計劃:**
                            - Pivot: ${setup.pivot_price:.2f}
                            - 入場: ${setup.entry_price}
                            - 止損: ${setup.stop_loss}
                            - T1: ${setup.target_1} (2R)
                            - R:R: {setup.risk_reward}:1
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **VCP 數據:**
                            - 收縮次數: {setup.contractions}
                            - 最終緊縮: {setup.tightness:.1f}%
                            - RS Rating: {setup.rs_rating:.0f}
                            """)
                        
                        if st.button(f"📈 查看 {setup.ticker} 圖表", key=f"vcp_chart_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "6mo")
                            if df is not None:
                                fig = ChartBuilder.create_annotated_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        if setup.options_strategy and setup.quality in ['A+', 'A']:
                            with st.expander("📊 期權策略"):
                                rec_idx = setup.options_strategy['recommended_index']
                                rec_strategy = setup.options_strategy['strategies'][rec_idx]
                                st.markdown(f"**推薦:** {rec_strategy['name']}")
                                st.markdown(rec_strategy['details'])
            
            if not bgu_results and not vcp_results:
                st.info("沒有發現符合條件的 Setup，嘗試其他板塊或降低質量要求")
    
    # ===== TAB 4: Position Calculator =====
    with tabs[3]:
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
                
                st.markdown("### 止盈目標")
                risk = entry - stop
                for r in [2, 3, 5]:
                    target = entry + risk * r
                    profit = result['shares'] * risk * r
                    st.write(f"• {r}R: ${target:.2f} (盈利 ${profit:,.0f})")
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v7.5 Pro")
    st.sidebar.markdown(f"""
    **新功能:**
    - ✅ 圖表標註 Setup 關鍵點
    - ✅ 優化止損 (更緊)
    - ✅ 期權策略建議
    - ✅ 詳細買入理由
    
    **期權策略:**
    - 賣 PUT (收權利金)
    - 買 CALL (看漲)
    - 牛市 PUT 價差
    - 牛市 CALL 價差
    """)


if __name__ == "__main__":
    main()
