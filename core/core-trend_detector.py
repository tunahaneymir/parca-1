"""
Trend Detector - Parça 1
Based on: pa-strateji2 Parça 1

Features:
- EMA20/50 trend detection
- Sideways market filter
- 4H timeframe
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TrendResult:
    """Trend detection sonucu"""
    direction: Literal["UP", "DOWN", "SIDEWAYS"]
    ema_20: float
    ema_50: float
    ema_distance_pct: float
    atr_ratio: float
    price_range_pct: float
    confidence: float  # 0-1 arası trend güveni
    slope_up: bool
    slope_down: bool


class TrendDetector:
    """
    4H Timeframe Trend Detection
    
    Rules:
    - UPTREND: price > EMA20 > EMA50 + EMA20 slope up
    - DOWNTREND: price < EMA20 < EMA50 + EMA20 slope down
    - SIDEWAYS: 3 koşuldan 2'si:
        1. EMA distance < 0.5%
        2. ATR ratio < 0.6%
        3. Price range (8 candles) < 8%
    """
    
    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        sideways_ema_distance: float = 0.005,  # %0.5
        sideways_atr_ratio: float = 0.006,     # %0.6
        sideways_range: float = 0.08,          # %8
        ema_slope_lookback: int = 5
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.sideways_ema_distance = sideways_ema_distance
        self.sideways_atr_ratio = sideways_atr_ratio
        self.sideways_range = sideways_range
        self.ema_slope_lookback = ema_slope_lookback
    
    def detect(
        self, 
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: Optional[np.ndarray] = None
    ) -> TrendResult:
        """
        Trend tespiti yap
        
        Args:
            close: Close prices
            high: High prices
            low: Low prices
            atr: ATR values (optional, hesaplanır)
            
        Returns:
            TrendResult
        """
        # EMA hesapla
        ema_20 = self.calculate_ema(close, self.ema_fast)[-1]
        ema_50 = self.calculate_ema(close, self.ema_slow)[-1]
        
        current_price = close[-1]
        
        # EMA slope kontrolü
        ema_20_values = self.calculate_ema(close, self.ema_fast)
        slope_up = self._check_slope_up(ema_20_values)
        slope_down = self._check_slope_down(ema_20_values)
        
        # ATR hesapla
        if atr is None:
            atr = self.calculate_atr(high, low, close)
        current_atr = atr[-1]
        
        # Sideways kriterleri hesapla
        ema_distance_pct = abs(ema_20 - ema_50) / current_price
        atr_ratio = current_atr / current_price
        price_range_pct = self._calculate_price_range(high, low, current_price)
        
        # Sideways kontrolü
        sideways_signals = 0
        if ema_distance_pct < self.sideways_ema_distance:
            sideways_signals += 1
        if atr_ratio < self.sideways_atr_ratio:
            sideways_signals += 1
        if price_range_pct < self.sideways_range:
            sideways_signals += 1
        
        # Trend belirleme
        if sideways_signals >= 2:
            direction = "SIDEWAYS"
            confidence = sideways_signals / 3.0
        elif current_price > ema_20 > ema_50 and slope_up:
            direction = "UP"
            confidence = self._calculate_trend_confidence(
                current_price, ema_20, ema_50, slope_up
            )
        elif current_price < ema_20 < ema_50 and slope_down:
            direction = "DOWN"
            confidence = self._calculate_trend_confidence(
                current_price, ema_20, ema_50, slope_down
            )
        else:
            direction = "SIDEWAYS"
            confidence = 0.3
        
        return TrendResult(
            direction=direction,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_distance_pct=ema_distance_pct,
            atr_ratio=atr_ratio,
            price_range_pct=price_range_pct,
            confidence=confidence,
            slope_up=slope_up,
            slope_down=slope_down
        )
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Exponential Moving Average hesapla
        
        Args:
            prices: Fiyat dizisi
            period: EMA periyodu
            
        Returns:
            EMA değerleri
        """
        ema = np.zeros_like(prices, dtype=float)
        multiplier = 2 / (period + 1)
        
        # İlk değer SMA
        ema[period-1] = np.mean(prices[:period])
        
        # EMA hesaplama
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def calculate_atr(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        close: np.ndarray, 
        period: int = 14
    ) -> np.ndarray:
        """
        Average True Range hesapla
        
        Args:
            high: High fiyatları
            low: Low fiyatları
            close: Close fiyatları
            period: ATR periyodu
            
        Returns:
            ATR değerleri
        """
        # True Range hesapla
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]
        
        # ATR = TR'nin EMA'sı
        atr = self.calculate_ema(tr, period)
        
        return atr
    
    def _check_slope_up(self, ema_values: np.ndarray) -> bool:
        """EMA20 yükseliyor mu?"""
        if len(ema_values) < self.ema_slope_lookback:
            return False
        
        recent_emas = ema_values[-self.ema_slope_lookback:]
        
        # Son 5 değerin en az 4'ü artan trend
        increasing_count = 0
        for i in range(1, len(recent_emas)):
            if recent_emas[i] > recent_emas[i-1]:
                increasing_count += 1
        
        return increasing_count >= (self.ema_slope_lookback - 1) * 0.6
    
    def _check_slope_down(self, ema_values: np.ndarray) -> bool:
        """EMA20 düşüyor mu?"""
        if len(ema_values) < self.ema_slope_lookback:
            return False
        
        recent_emas = ema_values[-self.ema_slope_lookback:]
        
        # Son 5 değerin en az 4'ü azalan trend
        decreasing_count = 0
        for i in range(1, len(recent_emas)):
            if recent_emas[i] < recent_emas[i-1]:
                decreasing_count += 1
        
        return decreasing_count >= (self.ema_slope_lookback - 1) * 0.6
    
    def _calculate_price_range(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        current_price: float,
        lookback: int = 8
    ) -> float:
        """Son N mumun fiyat aralığı"""
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])
        
        price_range = (recent_high - recent_low) / current_price
        return price_range
    
    def _calculate_trend_confidence(
        self,
        price: float,
        ema_20: float,
        ema_50: float,
        slope_confirmed: bool
    ) -> float:
        """Trend güven skoru (0-1)"""
        confidence = 0.5
        
        # EMA sıralaması
        if price > ema_20 > ema_50:
            confidence += 0.2
        elif price < ema_20 < ema_50:
            confidence += 0.2
        
        # Slope onayı
        if slope_confirmed:
            confidence += 0.2
        
        # EMA mesafesi (daha geniş = daha güçlü trend)
        ema_distance = abs(ema_20 - ema_50) / price
        if ema_distance > 0.02:  # >%2
            confidence += 0.1
        
        return min(confidence, 1.0)


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test data oluştur
    np.random.seed(42)
    
    # Uptrend simulation
    close = np.cumsum(np.random.randn(100) * 10 + 5) + 50000
    high = close + np.random.rand(100) * 20
    low = close - np.random.rand(100) * 20
    
    # Detector oluştur
    detector = TrendDetector()
    
    # Trend tespit et
    result = detector.detect(close, high, low)
    
    print("\n" + "="*60)
    print("TREND DETECTION RESULT")
    print("="*60)
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nEMA 20: ${result.ema_20:,.2f}")
    print(f"EMA 50: ${result.ema_50:,.2f}")
    print(f"Current Price: ${close[-1]:,.2f}")
    print(f"\nEMA Distance: {result.ema_distance_pct*100:.2f}%")
    print(f"ATR Ratio: {result.atr_ratio*100:.2f}%")
    print(f"Price Range: {result.price_range_pct*100:.2f}%")
    print(f"\nSlope Up: {result.slope_up}")
    print(f"Slope Down: {result.slope_down}")
    print("="*60 + "\n")
