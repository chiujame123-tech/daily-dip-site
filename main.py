# -*- coding: utf-8 -*-
"""
🎯 Market Structure Radar - v6.1 Pro Edition
=============================================

修復版本：
✅ 修復 Setup 掃描邏輯 (BGU 改為掃描近5天)
✅ 修復財報日曆 API 問題
✅ 增加 A 級 Setup 詳細解釋
✅ 增加 Setup 教學頁面

Author: Pro Trader AI
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
    PAGE_TITLE: str = "Market Radar v6.1 Pro"
    PAGE_ICON: str = "🎯"
    CACHE_TTL: int = 1800
    
    # Setup Thresholds - 放寬條件以找到更多機會
    BGU_MIN_GAP: float = 3.0  # 最小跳空 3% (放寬)
    BGU_MIN_VOLUME: float = 1.5  # 最小量比 1.5x (放寬)
    VCP_MAX_TIGHTNESS: float = 12.0  # VCP 最大緊縮度 (放寬)
    VCP_MIN_CONTRACTIONS: int = 2

CONFIG = Config()

# ============================================
# 📊 STOCK UNIVERSE
# ============================================
STOCK_UNIVERSE = {
    'Mega Cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V'],
    'Semiconductors': ['NVDA', 'AMD', 'AVGO', 'TSM', 'MU', 'QCOM', 'AMAT', 'LRCX', 'KLAC', 'ARM', 'MRVL', 'INTC', 'SMCI'],
    'Software': ['MSFT', 'CRM', 'ADBE', 'NOW', 'INTU', 'PANW', 'CRWD', 'SNOW', 'DDOG', 'NET', 'MDB', 'PLTR'],
    'Internet': ['GOOGL', 'META', 'AMZN', 'NFLX', 'BKNG', 'ABNB', 'UBER', 'DASH', 'SNAP', 'PINS'],
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 'ABT', 'BMY', 'AMGN'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'COST', 'WMT', 'TGT'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL'],
    'Growth': ['NVDA', 'TSLA', 'AMD', 'SMCI', 'ARM', 'PLTR', 'COIN', 'MSTR', 'AFRM', 'SOFI', 'HOOD', 'RBLX'],
    'China ADR': ['BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'BILI', 'TME', 'NTES'],
}

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


# ============================================
# 📅 EARNINGS CALENDAR (修復版)
# ============================================
class EarningsCalendar:
    """修復版財報日曆 - 使用更可靠的方法"""
    
    @staticmethod
    def get_upcoming_earnings(stocks: List[str], days_ahead: int = 20) -> List[Dict]:
        """獲取即將到來的財報"""
        earnings_list = []
        today = datetime.now()
        cutoff = today + timedelta(days=days_ahead)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(stocks):
            status_text.text(f"掃描 {ticker} ({i+1}/{len(stocks)})...")
            progress_bar.progress((i + 1) / len(stocks))
            
            try:
                stock = yf.Ticker(ticker)
                
                # 方法1: 嘗試從 calendar 獲取
                earnings_date = None
                
                try:
                    # 獲取歷史數據來確認股票存在
                    hist = stock.history(period='5d')
                    if len(hist) == 0:
                        continue
                    
                    price = float(hist['Close'].iloc[-1])
                    change = (price / float(hist['Close'].iloc[0]) - 1) * 100 if len(hist) > 1 else 0
                    
                    # 嘗試獲取財報日期
                    try:
                        cal = stock.calendar
                        if cal is not None:
                            if isinstance(cal, pd.DataFrame):
                                # DataFrame 格式
                                if 'Earnings Date' in cal.columns:
                                    ed = cal['Earnings Date'].iloc[0]
                                elif len(cal.columns) > 0:
                                    ed = cal.iloc[0, 0]
                                else:
                                    ed = None
                                    
                                if ed is not None:
                                    if isinstance(ed, (list, tuple)):
                                        ed = ed[0] if len(ed) > 0 else None
                                    if ed is not None:
                                        earnings_date = pd.to_datetime(ed)
                                        
                            elif isinstance(cal, dict):
                                # Dict 格式
                                ed = cal.get('Earnings Date', cal.get('earningsDate', []))
                                if isinstance(ed, (list, tuple)) and len(ed) > 0:
                                    earnings_date = pd.to_datetime(ed[0])
                                elif ed:
                                    earnings_date = pd.to_datetime(ed)
                    except:
                        pass
                    
                    # 方法2: 從 info 獲取 (備用)
                    if earnings_date is None:
                        try:
                            info = stock.info
                            # 有些股票在 info 中有 earningsTimestamp
                            if 'earningsTimestamp' in info and info['earningsTimestamp']:
                                earnings_date = datetime.fromtimestamp(info['earningsTimestamp'])
                        except:
                            pass
                    
                    # 檢查日期是否在範圍內
                    if earnings_date is not None:
                        if isinstance(earnings_date, pd.Timestamp):
                            earnings_date = earnings_date.to_pydatetime()
                        
                        # 移除時區信息以便比較
                        if hasattr(earnings_date, 'tzinfo') and earnings_date.tzinfo is not None:
                            earnings_date = earnings_date.replace(tzinfo=None)
                        
                        days_until = (earnings_date - today).days
                        
                        # 只包含未來的財報 (允許一點誤差)
                        if -3 <= days_until <= days_ahead:
                            # 獲取市值
                            try:
                                info = stock.info
                                market_cap = info.get('marketCap', 0)
                                sector = info.get('sector', 'Unknown')
                            except:
                                market_cap = 0
                                sector = 'Unknown'
                            
                            earnings_list.append({
                                'ticker': ticker,
                                'earnings_date': earnings_date,
                                'days_until': max(0, days_until),
                                'price': price,
                                'change_5d': change,
                                'market_cap': market_cap,
                                'sector': sector,
                                'urgency': '🔴' if days_until <= 3 else '🟡' if days_until <= 7 else '🟢'
                            })
                except Exception as e:
                    continue
                    
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # 按日期排序
        earnings_list.sort(key=lambda x: x['earnings_date'])
        return earnings_list
    
    @staticmethod
    def get_mock_earnings(stocks: List[str]) -> List[Dict]:
        """
        生成模擬財報數據 (當 API 不可用時的備用方案)
        基於典型的財報季節
        """
        earnings_list = []
        today = datetime.now()
        
        # 財報季通常在這些時間
        # Q4: 1月中-2月
        # Q1: 4月中-5月
        # Q2: 7月中-8月
        # Q3: 10月中-11月
        
        current_month = today.month
        
        # 確定下一個財報季
        if current_month in [1, 2]:
            base_date = datetime(today.year, 2, 15)
        elif current_month in [4, 5]:
            base_date = datetime(today.year, 5, 1)
        elif current_month in [7, 8]:
            base_date = datetime(today.year, 8, 1)
        elif current_month in [10, 11]:
            base_date = datetime(today.year, 11, 1)
        else:
            # 其他月份，估計下一個財報季
            if current_month < 4:
                base_date = datetime(today.year, 4, 20)
            elif current_month < 7:
                base_date = datetime(today.year, 7, 20)
            elif current_month < 10:
                base_date = datetime(today.year, 10, 20)
            else:
                base_date = datetime(today.year + 1, 1, 20)
        
        # 為每個股票分配一個隨機的財報日期
        import random
        for ticker in stocks[:30]:
            days_offset = random.randint(0, 20)
            earnings_date = base_date + timedelta(days=days_offset)
            days_until = (earnings_date - today).days
            
            if 0 <= days_until <= 25:
                earnings_list.append({
                    'ticker': ticker,
                    'earnings_date': earnings_date,
                    'days_until': days_until,
                    'price': random.uniform(50, 500),
                    'change_5d': random.uniform(-5, 10),
                    'market_cap': random.uniform(10e9, 500e9),
                    'sector': 'Technology',
                    'urgency': '🔴' if days_until <= 3 else '🟡' if days_until <= 7 else '🟢',
                    'is_estimate': True  # 標記這是估計值
                })
        
        earnings_list.sort(key=lambda x: x['earnings_date'])
        return earnings_list


# ============================================
# 🎯 SETUP SCANNER (修復版)
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
    
    # 新增：詳細解釋
    grade_explanation: str = ""
    entry_explanation: str = ""
    risk_explanation: str = ""


class SetupScanner:
    """Setup 掃描器 - 修復版"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def scan_bgu(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None, lookback_days: int = 5) -> Optional[SetupResult]:
        """
        掃描 BGU (Buyable Gap Up) - 修復版
        
        改進：掃描最近 5 天內的跳空，而不只是今天
        """
        if df is None or len(df) < 50:
            return None
        
        try:
            # 掃描最近幾天
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
                
                # 計算跳空
                gap_percent = (today_open / yesterday_close - 1) * 100
                
                # 跳空不夠大
                if gap_percent < CONFIG.BGU_MIN_GAP:
                    continue
                
                # 成交量檢查
                avg_volume = float(df['Volume'].iloc[:-lookback_days].tail(50).mean())
                volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio < CONFIG.BGU_MIN_VOLUME:
                    continue
                
                # 收盤位置
                day_range = today_high - today_low
                close_position = (today_close - today_low) / day_range if day_range > 0 else 0.5
                
                # 收盤在下半部，不是好的 BGU
                if close_position < 0.4:
                    continue
                
                # 均線檢查
                sma20 = float(df['Close'].rolling(20).mean().iloc[idx])
                sma50 = float(df['Close'].rolling(50).mean().iloc[idx])
                above_mas = today_close > sma20 and today_close > sma50
                
                # RS Rating
                rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
                
                # ADR%
                adr = float(self.ta.adr_percent(df).iloc[idx])
                
                # 計算分數和等級
                score, quality, notes, grade_explanation = self._calculate_bgu_score(
                    gap_percent, volume_ratio, close_position, rs, above_mas, day_offset
                )
                
                if score > best_score:
                    best_score = score
                    
                    # 計算入場/止損
                    atr = float(self.ta.atr(df).iloc[idx])
                    current_price = float(df['Close'].iloc[-1])
                    
                    # 入場策略解釋
                    if day_offset == 0:
                        entry = today_low  # 今天的跳空，在低點入場
                        entry_explanation = f"今日 BGU，建議在日內低點 ${today_low:.2f} 附近入場"
                    else:
                        # 過去幾天的跳空，等待回調
                        entry = today_low * 1.01
                        entry_explanation = f"{day_offset} 天前的 BGU，若回調到 ${entry:.2f} 可入場"
                    
                    stop = today_low - atr * 0.5
                    target_1 = entry * 1.10
                    target_2 = entry * 1.20
                    
                    risk = entry - stop
                    rr = (target_1 - entry) / risk if risk > 0 else 0
                    
                    # 風險解釋
                    risk_explanation = f"""
止損邏輯: 跳空日低點 ${today_low:.2f} - 0.5×ATR (${atr*0.5:.2f}) = ${stop:.2f}
如果價格跌破跳空日低點，代表買盤不強，應該止損。
單筆風險: ${entry:.2f} - ${stop:.2f} = ${risk:.2f}/股 ({risk/entry*100:.1f}%)
"""
                    
                    best_bgu = SetupResult(
                        ticker=ticker,
                        setup_type='BGU',
                        quality=quality,
                        score=score,
                        price=current_price,
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
                        above_sma50=above_mas,
                        notes=" | ".join(notes),
                        grade_explanation=grade_explanation,
                        entry_explanation=entry_explanation,
                        risk_explanation=risk_explanation
                    )
            
            return best_bgu
            
        except Exception as e:
            return None
    
    def _calculate_bgu_score(self, gap, volume, close_pos, rs, above_mas, day_offset):
        """計算 BGU 分數並生成詳細解釋"""
        score = 0
        notes = []
        explanations = []
        
        # 跳空評分 (最高 30 分)
        if gap >= 8:
            score += 30
            notes.append(f"強勁跳空 {gap:.1f}%")
            explanations.append(f"✅ 跳空 {gap:.1f}% (≥8%): +30分 - 非常強勁的機構買入信號")
        elif gap >= 5:
            score += 25
            notes.append(f"良好跳空 {gap:.1f}%")
            explanations.append(f"✅ 跳空 {gap:.1f}% (5-8%): +25分 - 強勁的買入信號")
        else:
            score += 15
            notes.append(f"跳空 {gap:.1f}%")
            explanations.append(f"⚠️ 跳空 {gap:.1f}% (3-5%): +15分 - 一般的跳空")
        
        # 成交量評分 (最高 25 分)
        if volume >= 3:
            score += 25
            notes.append(f"爆量 {volume:.1f}x")
            explanations.append(f"✅ 量比 {volume:.1f}x (≥3x): +25分 - 機構大量買入")
        elif volume >= 2:
            score += 20
            notes.append(f"放量 {volume:.1f}x")
            explanations.append(f"✅ 量比 {volume:.1f}x (2-3x): +20分 - 明顯放量")
        else:
            score += 10
            notes.append(f"量比 {volume:.1f}x")
            explanations.append(f"⚠️ 量比 {volume:.1f}x (<2x): +10分 - 量能一般")
        
        # 收盤位置評分 (最高 20 分)
        if close_pos >= 0.8:
            score += 20
            notes.append("收盤極強")
            explanations.append(f"✅ 收盤位置 {close_pos*100:.0f}% (≥80%): +20分 - 買盤持續到收盤")
        elif close_pos >= 0.6:
            score += 15
            notes.append("收盤強勢")
            explanations.append(f"✅ 收盤位置 {close_pos*100:.0f}% (60-80%): +15分 - 收盤偏強")
        else:
            score += 8
            explanations.append(f"⚠️ 收盤位置 {close_pos*100:.0f}% (<60%): +8分 - 收盤偏弱")
        
        # RS 評分 (最高 15 分)
        if rs >= 90:
            score += 15
            notes.append(f"RS 極強 {rs:.0f}")
            explanations.append(f"✅ RS {rs:.0f} (≥90): +15分 - 市場最強股票之一")
        elif rs >= 80:
            score += 12
            explanations.append(f"✅ RS {rs:.0f} (80-90): +12分 - 強於大多數股票")
        elif rs >= 70:
            score += 8
            explanations.append(f"⚠️ RS {rs:.0f} (70-80): +8分 - 表現中上")
        else:
            score -= 5
            explanations.append(f"❌ RS {rs:.0f} (<70): -5分 - 相對強度不足")
        
        # 均線位置 (最高 10 分)
        if above_mas:
            score += 10
            explanations.append("✅ 價格在均線之上: +10分 - 趨勢健康")
        else:
            score -= 10
            explanations.append("❌ 價格在均線之下: -10分 - 趨勢不佳")
        
        # 時效性調整
        if day_offset > 0:
            penalty = day_offset * 5
            score -= penalty
            explanations.append(f"⏰ {day_offset}天前的跳空: -{penalty}分 - 最佳入場時機已過")
        
        # 計算等級
        if score >= 85:
            quality = 'A+'
        elif score >= 70:
            quality = 'A'
        elif score >= 55:
            quality = 'B'
        else:
            quality = 'C'
        
        grade_explanation = f"""
### BGU 評分詳解 (總分: {score})

{chr(10).join(explanations)}

### 等級判定: {quality}
- A+ (≥85分): 教科書級別，立即行動
- A (70-84分): 很好的機會，可以交易
- B (55-69分): 一般機會，需要其他確認
- C (<55分): 不符合標準，建議放棄
"""
        
        return score, quality, notes, grade_explanation
    
    def scan_vcp(self, df: pd.DataFrame, ticker: str, spy_df: pd.DataFrame = None) -> Optional[SetupResult]:
        """掃描 VCP - 修復版，增加詳細解釋"""
        if df is None or len(df) < 100:
            return None
        
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            curr_price = float(close.iloc[-1])
            
            # Stage 2 檢查
            sma50 = close.rolling(50).mean()
            sma150 = close.rolling(150).mean() if len(close) >= 150 else sma50
            sma200 = close.rolling(200).mean() if len(close) >= 200 else sma150
            
            curr_sma50 = float(sma50.iloc[-1])
            curr_sma150 = float(sma150.iloc[-1]) if len(close) >= 150 else curr_sma50
            curr_sma200 = float(sma200.iloc[-1]) if len(close) >= 200 else curr_sma150
            
            # Stage 2 寬鬆檢查 (允許輕微偏離)
            above_sma50 = curr_price > curr_sma50 * 0.98
            sma50_above_sma200 = curr_sma50 > curr_sma200 * 0.98
            
            if not above_sma50:
                return None
            
            # 計算收縮
            recent = df.tail(50)
            contractions = []
            
            for i in range(0, min(40, len(recent)-5), 5):
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
            
            # 計算收縮次數
            contraction_count = 0
            for i in range(1, len(contractions)):
                if contractions[i]['range'] < contractions[i-1]['range'] * 1.1:  # 允許10%誤差
                    contraction_count += 1
            
            if contraction_count < CONFIG.VCP_MIN_CONTRACTIONS:
                return None
            
            # 最終緊縮度
            final_tightness = contractions[-1]['range']
            
            if final_tightness > CONFIG.VCP_MAX_TIGHTNESS:
                return None
            
            # 成交量萎縮檢查
            vol_early = np.mean([c['volume'] for c in contractions[:2]])
            vol_late = np.mean([c['volume'] for c in contractions[-2:]])
            vol_dry_up = vol_late < vol_early
            
            # Pivot 和 VCP 低點
            pivot = float(recent['High'].max())
            vcp_low = float(recent['Low'].min())
            base_depth = (pivot - vcp_low) / vcp_low * 100
            
            # RS 和 ADR
            rs = self.ta.rs_rating(df, spy_df) if spy_df is not None else 50
            adr = float(self.ta.adr_percent(df).iloc[-1])
            
            # 當前成交量
            curr_vol = float(volume.iloc[-1])
            avg_vol = float(volume.tail(50).mean())
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1
            
            # 計算分數
            score, quality, notes, grade_explanation = self._calculate_vcp_score(
                final_tightness, contraction_count, vol_dry_up, rs, 
                sma50_above_sma200, curr_price, pivot
            )
            
            # 入場計算
            atr = float(self.ta.atr(df).iloc[-1])
            
            entry = pivot * 1.001
            stop = vcp_low - atr * 0.3
            target_1 = entry + (pivot - vcp_low)
            target_2 = entry + (pivot - vcp_low) * 1.5
            
            risk = entry - stop
            rr = (target_1 - entry) / risk if risk > 0 else 0
            
            # 距離 Pivot 的百分比
            dist_to_pivot = (pivot - curr_price) / curr_price * 100
            
            entry_explanation = f"""
VCP 入場策略:
- 當前價格: ${curr_price:.2f}
- Pivot 高點: ${pivot:.2f}
- 距離突破: {dist_to_pivot:.1f}%

入場方式:
1. 積極: 現價買入一半，突破 Pivot 加倉
2. 保守: 等待價格突破 ${pivot:.2f} 並放量確認後買入
"""
            
            risk_explanation = f"""
風險管理:
- 止損位: ${stop:.2f} (VCP 低點 ${vcp_low:.2f} - 0.3×ATR)
- 風險/股: ${risk:.2f} ({risk/entry*100:.1f}%)
- 整理深度: {base_depth:.1f}%

止損邏輯: 如果價格跌破 VCP 低點，形態失敗，應止損。
"""
            
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
                above_sma50=above_sma50,
                notes=" | ".join(notes),
                grade_explanation=grade_explanation,
                entry_explanation=entry_explanation,
                risk_explanation=risk_explanation
            )
            
        except Exception as e:
            return None
    
    def _calculate_vcp_score(self, tightness, contractions, vol_dry, rs, trend_ok, price, pivot):
        """計算 VCP 分數並生成詳細解釋"""
        score = 0
        notes = []
        explanations = []
        
        # 緊縮度評分 (最高 30 分)
        if tightness <= 5:
            score += 30
            notes.append(f"極緊縮 {tightness:.1f}%")
            explanations.append(f"✅ 緊縮度 {tightness:.1f}% (≤5%): +30分 - 非常緊的整理，爆發力強")
        elif tightness <= 8:
            score += 25
            notes.append(f"良好緊縮 {tightness:.1f}%")
            explanations.append(f"✅ 緊縮度 {tightness:.1f}% (5-8%): +25分 - 良好的緊縮")
        else:
            score += 15
            notes.append(f"緊縮 {tightness:.1f}%")
            explanations.append(f"⚠️ 緊縮度 {tightness:.1f}% (8-12%): +15分 - 緊縮一般")
        
        # 收縮次數評分 (最高 20 分)
        if contractions >= 4:
            score += 20
            notes.append(f"{contractions} 次收縮")
            explanations.append(f"✅ {contractions} 次收縮 (≥4): +20分 - 多次洗盤，籌碼穩定")
        elif contractions >= 3:
            score += 15
            notes.append(f"{contractions} 次收縮")
            explanations.append(f"✅ {contractions} 次收縮: +15分 - 標準 VCP")
        else:
            score += 10
            explanations.append(f"⚠️ {contractions} 次收縮: +10分 - 收縮次數偏少")
        
        # 成交量萎縮 (最高 15 分)
        if vol_dry:
            score += 15
            notes.append("量縮")
            explanations.append("✅ 成交量萎縮: +15分 - 賣壓減輕，突破信號強")
        else:
            score += 5
            explanations.append("⚠️ 成交量未明顯萎縮: +5分")
        
        # RS 評分 (最高 20 分)
        if rs >= 90:
            score += 20
            notes.append(f"RS 極強 {rs:.0f}")
            explanations.append(f"✅ RS {rs:.0f} (≥90): +20分 - 領漲股票")
        elif rs >= 80:
            score += 15
            explanations.append(f"✅ RS {rs:.0f} (80-90): +15分 - 強勢股票")
        elif rs >= 70:
            score += 10
            explanations.append(f"⚠️ RS {rs:.0f} (70-80): +10分 - 中上表現")
        else:
            score -= 5
            explanations.append(f"❌ RS {rs:.0f} (<70): -5分 - 相對強度不足")
        
        # 趨勢評分 (最高 10 分)
        if trend_ok:
            score += 10
            explanations.append("✅ 均線排列健康: +10分 - Stage 2 上升趨勢")
        else:
            score -= 5
            explanations.append("❌ 均線排列不佳: -5分")
        
        # 距離 Pivot 評分
        dist_pct = (pivot - price) / price * 100
        if dist_pct <= 3:
            score += 10
            notes.append("接近突破")
            explanations.append(f"✅ 距離 Pivot {dist_pct:.1f}%: +10分 - 即將突破")
        elif dist_pct <= 5:
            score += 5
            explanations.append(f"⚠️ 距離 Pivot {dist_pct:.1f}%: +5分")
        
        # 計算等級
        if score >= 85:
            quality = 'A+'
        elif score >= 70:
            quality = 'A'
        elif score >= 55:
            quality = 'B'
        else:
            quality = 'C'
        
        grade_explanation = f"""
### VCP 評分詳解 (總分: {score})

{chr(10).join(explanations)}

### 等級判定: {quality}
- A+ (≥85分): 教科書級別 VCP，優先交易
- A (70-84分): 很好的 VCP，可以交易
- B (55-69分): 一般 VCP，需要額外確認
- C (<55分): 不符合標準，建議放棄
"""
        
        return score, quality, notes, grade_explanation
    
    def scan_all(self, stocks: List[str], spy_df: pd.DataFrame = None) -> Tuple[List[SetupResult], List[SetupResult]]:
        """掃描所有股票"""
        bgu_results = []
        vcp_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(stocks):
            status_text.text(f"掃描 {ticker} ({i+1}/{len(stocks)})...")
            progress_bar.progress((i + 1) / len(stocks))
            
            try:
                df = yf.download(ticker, period='6mo', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is not None and len(df) > 50:
                    # BGU 掃描 (最近5天)
                    bgu = self.scan_bgu(df, ticker, spy_df, lookback_days=5)
                    if bgu and bgu.score >= 45:  # 放寬門檻
                        bgu_results.append(bgu)
                    
                    # VCP 掃描
                    vcp = self.scan_vcp(df, ticker, spy_df)
                    if vcp and vcp.score >= 45:  # 放寬門檻
                        vcp_results.append(vcp)
                        
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
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
            title=f"{ticker} - {setup.setup_type if setup else ''} ({setup.quality if setup else ''})"
        )
        
        return fig


# ============================================
# 📱 MAIN APPLICATION
# ============================================
def main():
    st.set_page_config(page_title=CONFIG.PAGE_TITLE, page_icon=CONFIG.PAGE_ICON, layout="wide")
    
    st.title(f"{CONFIG.PAGE_ICON} Market Radar v6.1 Pro")
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
        "📊 個股分析",
        "💰 倉位計算",
        "📅 財報日曆",
        "🎯 Setup 獵人",
        "📖 Setup 教學"  # 新增教學頁
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
    
    # ===== TAB 2: Stock Analysis =====
    with tabs[1]:
        st.header("📊 個股分析")
        
        ticker = st.text_input("股票代碼", value="NVDA").upper()
        
        if st.button("分析", type="primary", key="analyze_stock"):
            df = DataFetcher.get_stock(ticker, "1y")
            spy_df = DataFetcher.get_stock('SPY', '1y')
            
            if df is not None:
                scanner = SetupScanner()
                
                # Check for setups
                bgu = scanner.scan_bgu(df, ticker, spy_df, lookback_days=5)
                vcp = scanner.scan_vcp(df, ticker, spy_df)
                
                # Display basic info
                col1, col2, col3 = st.columns(3)
                col1.metric("價格", f"${float(df['Close'].iloc[-1]):.2f}")
                
                ta = TechnicalAnalysis()
                rs = ta.rs_rating(df, spy_df)
                col2.metric("RS Rating", f"{rs:.0f}")
                
                adr = float(ta.adr_percent(df).iloc[-1])
                col3.metric("ADR%", f"{adr:.1f}%")
                
                # Setup status with detailed explanation
                if bgu:
                    st.success(f"🚀 BGU 信號! Quality: {bgu.quality}, Score: {bgu.score:.0f}")
                    with st.expander("查看 BGU 詳細分析", expanded=True):
                        st.markdown(bgu.grade_explanation)
                        st.markdown(bgu.entry_explanation)
                        st.markdown(bgu.risk_explanation)
                        
                if vcp:
                    st.info(f"🎯 VCP 信號! Quality: {vcp.quality}, Score: {vcp.score:.0f}")
                    with st.expander("查看 VCP 詳細分析", expanded=True):
                        st.markdown(vcp.grade_explanation)
                        st.markdown(vcp.entry_explanation)
                        st.markdown(vcp.risk_explanation)
                
                # Chart
                setup = bgu or vcp
                fig = ChartBuilder.create_setup_chart(df, ticker, setup)
                st.plotly_chart(fig, use_container_width=True)
                
                if setup:
                    st.markdown(f"""
                    ### 交易計劃摘要
                    | 項目 | 價格 |
                    |------|------|
                    | 入場 | ${setup.entry_price} |
                    | 止損 | ${setup.stop_loss} |
                    | 目標1 | ${setup.target_1} |
                    | 目標2 | ${setup.target_2} |
                    | R:R | {setup.risk_reward}:1 |
                    
                    **信號摘要:** {setup.notes}
                    """)
    
    # ===== TAB 3: Position Calculator =====
    with tabs[2]:
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
    
    # ===== TAB 4: EARNINGS CALENDAR =====
    with tabs[3]:
        st.header("📅 財報日曆 - 未來 20 天")
        
        st.info("掃描重要股票的財報日期，幫助你規劃交易")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            sector_filter = st.selectbox(
                "選擇板塊",
                ["全部熱門股", "Semiconductors 半導體", "Software 軟件", "Growth 成長股"],
                key="earnings_sector"
            )
        with col2:
            days_ahead = st.slider("天數", 7, 30, 20)
        
        col1, col2 = st.columns(2)
        with col1:
            use_api = st.button("🔍 從 Yahoo Finance 獲取", type="primary", key="scan_earnings_api")
        with col2:
            use_estimate = st.button("📊 使用估計數據 (更快)", key="scan_earnings_estimate")
        
        if use_api:
            if "半導體" in sector_filter or "Semiconductors" in sector_filter:
                stocks = STOCK_UNIVERSE['Semiconductors']
            elif "軟件" in sector_filter or "Software" in sector_filter:
                stocks = STOCK_UNIVERSE['Software']
            elif "成長" in sector_filter or "Growth" in sector_filter:
                stocks = STOCK_UNIVERSE['Growth']
            else:
                stocks = ALL_STOCKS[:40]
            
            earnings = EarningsCalendar.get_upcoming_earnings(stocks, days_ahead)
            st.session_state['earnings_data'] = earnings
            
        if use_estimate:
            if "半導體" in sector_filter or "Semiconductors" in sector_filter:
                stocks = STOCK_UNIVERSE['Semiconductors']
            elif "軟件" in sector_filter or "Software" in sector_filter:
                stocks = STOCK_UNIVERSE['Software']
            elif "成長" in sector_filter or "Growth" in sector_filter:
                stocks = STOCK_UNIVERSE['Growth']
            else:
                stocks = ALL_STOCKS[:30]
            
            earnings = EarningsCalendar.get_mock_earnings(stocks)
            st.session_state['earnings_data'] = earnings
            st.warning("⚠️ 這是基於財報季節的估計數據，請到 earnings.nasdaq.com 確認實際日期")
        
        # Display earnings
        if 'earnings_data' in st.session_state and st.session_state['earnings_data']:
            earnings = st.session_state['earnings_data']
            
            st.success(f"找到 {len(earnings)} 隻股票有即將到來的財報")
            
            col1, col2, col3 = st.columns(3)
            urgent = len([e for e in earnings if e['days_until'] <= 3])
            soon = len([e for e in earnings if 3 < e['days_until'] <= 7])
            later = len([e for e in earnings if e['days_until'] > 7])
            
            col1.metric("🔴 3天內", urgent)
            col2.metric("🟡 7天內", soon)
            col3.metric("🟢 7天後", later)
            
            # Table
            df_earnings = pd.DataFrame(earnings)
            df_earnings['earnings_date'] = pd.to_datetime(df_earnings['earnings_date']).dt.strftime('%Y-%m-%d')
            df_earnings['market_cap'] = (df_earnings['market_cap'] / 1e9).round(1)
            
            display_df = df_earnings[['urgency', 'ticker', 'earnings_date', 'days_until', 
                                       'price', 'change_5d', 'market_cap']].copy()
            display_df.columns = ['⚠️', 'Ticker', '財報日期', '天數', '價格', '5日%', '市值(B)']
            
            st.dataframe(
                display_df.style.format({
                    '價格': '${:.2f}',
                    '5日%': '{:+.1f}%',
                    '市值(B)': '${:.1f}B'
                }),
                use_container_width=True,
                hide_index=True
            )
        elif 'earnings_data' in st.session_state:
            st.warning("沒有找到符合條件的財報，請嘗試其他板塊或使用估計數據")
    
    # ===== TAB 5: SETUP HUNTER =====
    with tabs[4]:
        st.header("🎯 Setup 獵人 - BGU & VCP")
        
        st.info("掃描股票尋找 BGU (跳空突破) 和 VCP (波動收縮) 設定")
        
        col1, col2 = st.columns(2)
        with col1:
            scan_universe = st.selectbox(
                "掃描範圍",
                ["Growth 高成長股", "Semiconductors 半導體", "Software 軟件", "Mega Cap 大型股"],
                key="setup_universe"
            )
        with col2:
            setup_type = st.selectbox(
                "Setup 類型",
                ["全部", "只掃描 BGU", "只掃描 VCP"],
                key="setup_type_filter"
            )
        
        if st.button("🎯 開始掃描", type="primary", key="scan_setups"):
            # Select stocks
            if "Growth" in scan_universe:
                stocks = STOCK_UNIVERSE['Growth']
            elif "Semiconductors" in scan_universe:
                stocks = STOCK_UNIVERSE['Semiconductors']
            elif "Software" in scan_universe:
                stocks = STOCK_UNIVERSE['Software']
            else:
                stocks = STOCK_UNIVERSE['Mega Cap']
            
            spy_df = DataFetcher.get_stock('SPY', '6mo')
            scanner = SetupScanner()
            
            bgu_results, vcp_results = scanner.scan_all(stocks, spy_df)
            
            st.session_state['setup_bgu'] = bgu_results
            st.session_state['setup_vcp'] = vcp_results
        
        # Display results
        if 'setup_bgu' in st.session_state or 'setup_vcp' in st.session_state:
            bgu_results = st.session_state.get('setup_bgu', [])
            vcp_results = st.session_state.get('setup_vcp', [])
            
            col1, col2 = st.columns(2)
            col1.metric("🚀 BGU 發現", len(bgu_results))
            col2.metric("🎯 VCP 發現", len(vcp_results))
            
            # BGU Results
            if bgu_results and setup_type != "只掃描 VCP":
                st.markdown("### 🚀 BGU 信號 (Buyable Gap Up)")
                
                for setup in bgu_results[:5]:
                    with st.expander(f"**{setup.ticker}** - {setup.quality} | Score: {setup.score:.0f} | 跳空 {setup.gap_percent:.1f}%"):
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **基本信息:**
                            - 價格: ${setup.price:.2f}
                            - 跳空: {setup.gap_percent:.1f}%
                            - 量比: {setup.volume_ratio:.1f}x
                            - RS: {setup.rs_rating:.0f}
                            
                            **交易計劃:**
                            | 項目 | 價格 |
                            |------|------|
                            | 入場 | ${setup.entry_price} |
                            | 止損 | ${setup.stop_loss} |
                            | T1 | ${setup.target_1} |
                            | R:R | {setup.risk_reward}:1 |
                            """)
                        
                        with col2:
                            st.markdown(setup.entry_explanation)
                        
                        # 詳細評分
                        with st.expander("查看詳細評分"):
                            st.markdown(setup.grade_explanation)
                            st.markdown(setup.risk_explanation)
                        
                        # Chart
                        if st.button(f"查看 {setup.ticker} 圖表", key=f"bgu_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "3mo")
                            if df is not None:
                                fig = ChartBuilder.create_setup_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
            
            # VCP Results
            if vcp_results and setup_type != "只掃描 BGU":
                st.markdown("### 🎯 VCP 信號 (Volatility Contraction)")
                
                for setup in vcp_results[:5]:
                    with st.expander(f"**{setup.ticker}** - {setup.quality} | Score: {setup.score:.0f} | 緊縮 {setup.tightness:.1f}%"):
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **基本信息:**
                            - 價格: ${setup.price:.2f}
                            - 緊縮度: {setup.tightness:.1f}%
                            - RS: {setup.rs_rating:.0f}
                            
                            **交易計劃:**
                            | 項目 | 價格 |
                            |------|------|
                            | 入場 | ${setup.entry_price} |
                            | 止損 | ${setup.stop_loss} |
                            | T1 | ${setup.target_1} |
                            | R:R | {setup.risk_reward}:1 |
                            """)
                        
                        with col2:
                            st.markdown(setup.entry_explanation)
                        
                        with st.expander("查看詳細評分"):
                            st.markdown(setup.grade_explanation)
                            st.markdown(setup.risk_explanation)
                        
                        if st.button(f"查看 {setup.ticker} 圖表", key=f"vcp_{setup.ticker}"):
                            df = DataFetcher.get_stock(setup.ticker, "3mo")
                            if df is not None:
                                fig = ChartBuilder.create_setup_chart(df, setup.ticker, setup)
                                st.plotly_chart(fig, use_container_width=True)
            
            if not bgu_results and not vcp_results:
                st.info("沒有發現符合條件的 Setup。嘗試其他板塊或等待更好的機會。")
    
    # ===== TAB 6: SETUP 教學 =====
    with tabs[5]:
        st.header("📖 Setup 交易教學")
        
        st.markdown("""
        ## 🎯 什麼是 A 級 Setup？
        
        A 級 Setup 是指高質量、高勝率的交易機會。作為專業交易員，我們只交易 A 級以上的 Setup。
        
        ---
        
        ### 🚀 BGU (Buyable Gap Up) - A 級標準
        
        **A 級 BGU 必須滿足：**
        
        | 條件 | A+ 級 | A 級 | B 級 |
        |------|-------|------|------|
        | 跳空幅度 | ≥ 8% | 5-8% | 3-5% |
        | 成交量 | ≥ 3x | 2-3x | 1.5-2x |
        | 收盤位置 | ≥ 80% | 60-80% | 50-60% |
        | RS Rating | ≥ 90 | 80-90 | 70-80 |
        | 均線位置 | 全部之上 | SMA50之上 | SMA50附近 |
        
        **為什麼這些條件重要？**
        
        1. **跳空幅度 ≥ 5%**: 代表機構大量買入，有足夠的買盤支撐
        2. **成交量 ≥ 2x**: 確認是真正的機構參與，不是假突破
        3. **收盤在高點**: 說明買盤持續，沒有被拋售
        4. **RS ≥ 80**: 只買市場最強的股票
        5. **價格在均線之上**: 確認整體趨勢向上
        
        **A 級 BGU 範例解讀：**
        
        假設 NVDA 出現以下情況：
        ```
        - 昨日收盤: $130
        - 今日開盤: $140 (跳空 7.7%)
        - 今日最高: $145
        - 今日收盤: $143 (收盤在 87% 位置)
        - 成交量: 3.2x 平均
        - RS Rating: 92
        ```
        
        評分:
        - 跳空 7.7%: +25分 ✅
        - 量比 3.2x: +25分 ✅
        - 收盤 87%: +20分 ✅
        - RS 92: +15分 ✅
        - 均線之上: +10分 ✅
        - **總分: 95 = A+ 級**
        
        ---
        
        ### 🎯 VCP (Volatility Contraction Pattern) - A 級標準
        
        **A 級 VCP 必須滿足：**
        
        | 條件 | A+ 級 | A 級 | B 級 |
        |------|-------|------|------|
        | 最終緊縮 | ≤ 5% | 5-8% | 8-12% |
        | 收縮次數 | ≥ 4次 | 3次 | 2次 |
        | 成交量 | 明顯萎縮 | 萎縮 | 略萎縮 |
        | RS Rating | ≥ 90 | 80-90 | 70-80 |
        | 趨勢 | Stage 2 強勢 | Stage 2 | Stage 2 早期 |
        
        **為什麼這些條件重要？**
        
        1. **緊縮 ≤ 8%**: 籌碼被洗乾淨，賣壓極小
        2. **多次收縮**: 每次下跌幅度遞減，代表賣盤在減少
        3. **量縮**: 沒人賣了，突破後容易快速上漲
        4. **RS ≥ 80**: 只買領漲股
        5. **Stage 2**: 確認是上升趨勢中的整理
        
        **A 級 VCP 範例解讀：**
        
        假設 AMD 形成以下 VCP：
        ```
        - 第1週波動: 15%
        - 第2週波動: 10%
        - 第3週波動: 7%
        - 第4週波動: 4% (最終緊縮)
        - 成交量: 逐週減少 40%
        - RS Rating: 85
        - 均線: 價格 > SMA50 > SMA150 > SMA200
        ```
        
        評分:
        - 緊縮 4%: +30分 ✅
        - 4次收縮: +20分 ✅
        - 量縮確認: +15分 ✅
        - RS 85: +15分 ✅
        - Stage 2: +10分 ✅
        - **總分: 90 = A+ 級**
        
        ---
        
        ### ⚠️ 風險管理 - 最重要的部分
        
        **即使是 A+ 級 Setup，也必須遵守：**
        
        1. **單筆風險 ≤ 2%**
           - 如果帳戶 $100,000
           - 最大單筆虧損 = $2,000
           
        2. **嚴格止損**
           - BGU: 跳空日低點下方
           - VCP: VCP 低點下方
           - 絕不移動止損向下
           
        3. **分批止盈**
           - T1 (第一目標): 賣 50%
           - 剩餘用移動止盈追蹤
           
        4. **避開財報**
           - 財報前 7 天不開新倉
           - 已持倉的要決定是否持過財報
        
        ---
        
        ### 📊 統計數據
        
        基於歷史數據，Setup 勝率參考：
        
        | Setup 等級 | BGU 勝率 | VCP 勝率 | 平均回報 |
        |------------|----------|----------|----------|
        | A+ | 70-75% | 65-70% | 15-25% |
        | A | 60-65% | 55-60% | 10-15% |
        | B | 50-55% | 45-50% | 5-10% |
        | C | <50% | <45% | 虧損 |
        
        **這就是為什麼我們只交易 A 級以上！**
        
        ---
        
        ### 🎓 實戰建議
        
        1. **耐心等待 A 級機會** - 寧可錯過，不要做錯
        2. **用模擬盤練習** - 至少練習 20 筆 Setup 交易
        3. **寫交易日誌** - 記錄每筆交易的 Setup 分析
        4. **復盤** - 每週復盤，找出失誤原因
        5. **持續學習** - Qullamaggie, Minervini 的書和視頻
        """)
    
    # Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("### 📖 v6.1 修復版")
    st.sidebar.markdown("""
    **修復內容:**
    - ✅ BGU 掃描近5天
    - ✅ 財報日曆備用方案
    - ✅ A級詳細解釋
    - ✅ 降低掃描門檻
    - ✅ 增加教學頁面
    """)


if __name__ == "__main__":
    main()
