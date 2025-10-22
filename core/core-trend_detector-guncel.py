"""
Zone Detector - Parça 1 (UPDATED with Quality Scoring)
Based on: pa-strateji2 Parça 1

Features:
- ZigZag++ detection
- Swing High/Low detection
- Zone merging & filtering
- Zone quality scoring (0-10)
- Multi-timeframe support
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta


@dataclass
class Zone:
    """Zone bilgisi"""
    id: str
    price_low: float
    price_high: float
    price_mid: float
    touch_count: int
    thickness_pct: float
    last_touch_index: int
    creation_index: int
    timeframe: str
    method: str  # "zigzag", "swing", "both"
    quality: float = 0.0  # 0-10 (otomatik hesaplanacak)
    days_since_last_touch: float = 0.0  # NEW: tazelik için
    
    def calculate_quality(
        self, 
        current_index: int,
        candle_time_minutes: int = 240  # 4H = 240 dakika
    ) -> float:
        """
        Zone kalite skoru hesapla (0-10)
        
        Kriterlere göre:
        - Touch count: 0-4 puan
        - Thickness: 0-3 puan  
        - Recency: 0-3 puan
        
        Args:
            current_index: Mevcut mum index'i
            candle_time_minutes: Bir mum kaç dakika
            
        Returns:
            Quality score (0-10)
        """
        score = 0.0
        
        # === 1. TOUCH COUNT SCORING (0-4 puan) ===
        if self.touch_count == 2:
            score += 4  # Mükemmel! Taze zone
        elif self.touch_count == 3:
            score += 3  # İyi
        elif self.touch_count == 4:
            score += 2  # Orta
        else:  # 5+
            score += 1  # Zayıf (çok test edilmiş)
        
        # === 2. THICKNESS SCORING (0-3 puan) ===
        thickness_percent = self.thickness_pct * 100
        
        if thickness_percent < 0.5:
            score += 3  # Çok ince = mükemmel
        elif thickness_percent < 1.0:
            score += 2  # İnce = iyi
        elif thickness_percent < 1.5:
            score += 1  # Kalın ama kabul edilebilir
        # else: 0 puan (çok kalın)
        
        # === 3. RECENCY SCORING (0-3 puan) ===
        candles_since = current_index - self.last_touch_index
        self.days_since_last_touch = (candles_since * candle_time_minutes) / (60 * 24)
        
        if self.days_since_last_touch < 7:
            score += 3  # Son 1 hafta = taze!
        elif self.days_since_last_touch < 30:
            score += 2  # Son 1 ay = iyi
        else:  # 30+ gün
            score += 1  # Eski
        
        self.quality = score
        return score
    
    def get_ml_features(self) -> Dict[str, float]:
        """
        ML modeli için zone feature'ları
        """
        return {
            'zone_quality': self.quality,
            'zone_touch_count': float(self.touch_count),
            'zone_thickness_pct': self.thickness_pct,
            'zone_days_since_touch': self.days_since_last_touch,
            'zone_is_fresh': float(self.days_since_last_touch < 7),
            'zone_is_thin': float(self.thickness_pct < 0.01),  # <%1
            'zone_method_both': float(self.method == "both"),
            'zone_age_candles': float(self.last_touch_index - self.creation_index),
        }


@dataclass
class ZigZagPoint:
    """ZigZag pivot noktası"""
    index: int
    price: float
    type: str  # "high" or "low"


class ZoneDetector:
    """
    Zone Detection Engine
    
    Methods:
    - ZigZag++: Swing point detection
    - Swing HL: Confirmation
    - Dual confirmation: Both methods agree
    - Quality scoring: 0-10 rating system
    """
    
    def __init__(
        self,
        zigzag_depth: int = 12,
        zigzag_deviation: float = 5.0,
        zigzag_backstep: int = 2,
        swing_strength: int = 5,
        min_touches: int = 2,
        max_thickness_pct: float = 0.015,  # %1.5
        merge_tolerance_pct: float = 0.015,  # %1.5
        min_quality: float = 4.0  # Minimum zone quality
    ):
        self.zigzag_depth = zigzag_depth
        self.zigzag_deviation = zigzag_deviation
        self.zigzag_backstep = zigzag_backstep
        self.swing_strength = swing_strength
        self.min_touches = min_touches
        self.max_thickness_pct = max_thickness_pct
        self.merge_tolerance_pct = merge_tolerance_pct
        self.min_quality = min_quality
    
    # ═══════════════════════════════════════════════════════════
    # MAIN DETECTION METHOD
    # ═══════════════════════════════════════════════════════════
    
    def detect_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeframe: str = "4H",
        method: str = "both"
    ) -> List[Zone]:
        """
        Zone detection ana method
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeframe: Timeframe (4H, 1H, 15M)
            method: "zigzag", "swing", or "both"
            
        Returns:
            List of quality-scored zones
        """
        zones = []
        
        if method in ["zigzag", "both"]:
            zigzag_zones = self._detect_zigzag_zones(high, low, close, timeframe)
            zones.extend(zigzag_zones)
        
        if method in ["swing", "both"]:
            swing_zones = self._detect_swing_zones(high, low, timeframe)
            zones.extend(swing_zones)
        
        # Merge overlapping zones
        zones = self._merge_zones(zones, close[-1])
        
        # Filter by criteria
        zones = self._filter_zones(zones)
        
        # === QUALITY SCORING (YENİ!) ===
        current_index = len(close) - 1
        candle_minutes = self._get_candle_minutes(timeframe)
        
        for zone in zones:
            zone.calculate_quality(current_index, candle_minutes)
        
        # Filter by minimum quality
        zones = [z for z in zones if z.quality >= self.min_quality]
        
        # Sort by quality first, then proximity
        zones = sorted(zones, key=lambda z: (-z.quality, abs(z.price_mid - close[-1])))
        
        return zones
    
    def _get_candle_minutes(self, timeframe: str) -> int:
        """Timeframe'den dakika hesapla"""
        mapping = {
            "15M": 15,
            "1H": 60,
            "4H": 240,
            "1D": 1440
        }
        return mapping.get(timeframe, 240)
    
    # ═══════════════════════════════════════════════════════════
    # ZIGZAG++ DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_zigzag_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeframe: str
    ) -> List[Zone]:
        """ZigZag++ ile zone detection"""
        # ZigZag pivotları bul
        pivots = self._calculate_zigzag(high, low)
        
        # Pivotlardan zone oluştur
        zones = []
        for pivot in pivots:
            # Pivot etrafında zone oluştur
            zone_low = pivot.price * (1 - self.max_thickness_pct / 2)
            zone_high = pivot.price * (1 + self.max_thickness_pct / 2)
            
            # Touch count hesapla
            touches = self._count_touches(high, low, zone_low, zone_high)
            
            # Last touch index hesapla
            last_touch = self._find_last_touch(high, low, zone_low, zone_high, len(close) - 1)
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"zz_{timeframe}_{pivot.index}",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=pivot.price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / pivot.price,
                    last_touch_index=last_touch,
                    creation_index=pivot.index,
                    timeframe=timeframe,
                    method="zigzag"
                )
                zones.append(zone)
        
        return zones
    
    def _calculate_zigzag(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> List[ZigZagPoint]:
        """
        ZigZag++ pivot calculation
        
        Basitleştirilmiş versiyon
        """
        pivots = []
        last_pivot_type = None
        last_pivot_price = 0.0
        last_pivot_index = 0
        
        for i in range(self.zigzag_depth, len(high) - self.zigzag_depth):
            # High pivot kontrolü
            is_high_pivot = True
            for j in range(1, self.zigzag_depth + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_high_pivot = False
                    break
            
            # Low pivot kontrolü
            is_low_pivot = True
            for j in range(1, self.zigzag_depth + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_low_pivot = False
                    break
            
            # Pivot tespit edildi
            if is_high_pivot and (last_pivot_type != "high" or 
                                 (high[i] - last_pivot_price) / last_pivot_price > self.zigzag_deviation / 100):
                if i - last_pivot_index >= self.zigzag_backstep:
                    pivots.append(ZigZagPoint(index=i, price=high[i], type="high"))
                    last_pivot_type = "high"
                    last_pivot_price = high[i]
                    last_pivot_index = i
            
            elif is_low_pivot and (last_pivot_type != "low" or 
                                  (last_pivot_price - low[i]) / last_pivot_price > self.zigzag_deviation / 100):
                if i - last_pivot_index >= self.zigzag_backstep:
                    pivots.append(ZigZagPoint(index=i, price=low[i], type="low"))
                    last_pivot_type = "low"
                    last_pivot_price = low[i]
                    last_pivot_index = i
        
        return pivots
    
    # ═══════════════════════════════════════════════════════════
    # SWING HIGH/LOW DETECTION
    # ═══════════════════════════════════════════════════════════
    
    def _detect_swing_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        timeframe: str
    ) -> List[Zone]:
        """Swing High/Low ile zone detection"""
        zones = []
        
        # Swing Highs
        swing_highs = self._find_swing_highs(high)
        for idx, price in swing_highs:
            zone_low = price * (1 - self.max_thickness_pct / 2)
            zone_high = price * (1 + self.max_thickness_pct / 2)
            touches = self._count_touches(high, low, zone_low, zone_high)
            last_touch = self._find_last_touch(high, low, zone_low, zone_high, len(high) - 1)
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"sw_{timeframe}_{idx}_h",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / price,
                    last_touch_index=last_touch,
                    creation_index=idx,
                    timeframe=timeframe,
                    method="swing"
                )
                zones.append(zone)
        
        # Swing Lows
        swing_lows = self._find_swing_lows(low)
        for idx, price in swing_lows:
            zone_low = price * (1 - self.max_thickness_pct / 2)
            zone_high = price * (1 + self.max_thickness_pct / 2)
            touches = self._count_touches(high, low, zone_low, zone_high)
            last_touch = self._find_last_touch(high, low, zone_low, zone_high, len(low) - 1)
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"sw_{timeframe}_{idx}_l",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / price,
                    last_touch_index=last_touch,
                    creation_index=idx,
                    timeframe=timeframe,
                    method="swing"
                )
                zones.append(zone)
        
        return zones
    
    def _find_swing_highs(self, high: np.ndarray) -> List[Tuple[int, float]]:
        """Swing High noktaları bul"""
        swings = []
        strength = self.swing_strength
        
        for i in range(strength, len(high) - strength):
            is_swing = True
            
            # Sol ve sağ kontrol
            for j in range(1, strength + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                swings.append((i, high[i]))
        
        return swings
    
    def _find_swing_lows(self, low: np.ndarray) -> List[Tuple[int, float]]:
        """Swing Low noktaları bul"""
        swings = []
        strength = self.swing_strength
        
        for i in range(strength, len(low) - strength):
            is_swing = True
            
            # Sol ve sağ kontrol
            for j in range(1, strength + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_swing = False
                    break
            
            if is_swing:
                swings.append((i, low[i]))
        
        return swings
    
    # ═══════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════
    
    def _count_touches(
        self,
        high: np.ndarray,
        low: np.ndarray,
        zone_low: float,
        zone_high: float
    ) -> int:
        """Zone'a kaç kez dokunulmuş?"""
        touches = 0
        
        for i in range(len(high)):
            # Fiyat zone içine girdiyse touch
            if low[i] <= zone_high and high[i] >= zone_low:
                touches += 1
        
        return touches
    
    def _find_last_touch(
        self,
        high: np.ndarray,
        low: np.ndarray,
        zone_low: float,
        zone_high: float,
        current_index: int
    ) -> int:
        """Zone'un son dokunma index'i"""
        last_touch = 0
        
        for i in range(len(high)):
            if low[i] <= zone_high and high[i] >= zone_low:
                last_touch = i
        
        return last_touch
    
    def _merge_zones(self, zones: List[Zone], current_price: float) -> List[Zone]:
        """Çakışan zone'ları birleştir"""
        if not zones:
            return zones
        
        # Fiyata göre sırala
        zones = sorted(zones, key=lambda z: z.price_mid)
        
        merged = []
        current = zones[0]
        
        for next_zone in zones[1:]:
            # Çakışma kontrolü
            overlap = min(current.price_high, next_zone.price_high) - max(current.price_low, next_zone.price_low)
            overlap_pct = overlap / current_price
            
            if overlap > 0 and overlap_pct > self.merge_tolerance_pct:
                # Merge
                current = Zone(
                    id=f"merged_{current.id}_{next_zone.id}",
                    price_low=min(current.price_low, next_zone.price_low),
                    price_high=max(current.price_high, next_zone.price_high),
                    price_mid=(current.price_mid + next_zone.price_mid) / 2,
                    touch_count=max(current.touch_count, next_zone.touch_count),
                    thickness_pct=(max(current.price_high, next_zone.price_high) - 
                                 min(current.price_low, next_zone.price_low)) / current_price,
                    last_touch_index=max(current.last_touch_index, next_zone.last_touch_index),
                    creation_index=min(current.creation_index, next_zone.creation_index),
                    timeframe=current.timeframe,
                    method="both" if current.method != next_zone.method else current.method
                )
            else:
                merged.append(current)
                current = next_zone
        
        merged.append(current)
        return merged
    
    def _filter_zones(self, zones: List[Zone]) -> List[Zone]:
        """Kriterlere uymayan zone'ları filtrele"""
        filtered = []
        
        for zone in zones:
            # Min touches
            if zone.touch_count < self.min_touches:
                continue
            
            # Max thickness
            if zone.thickness_pct > self.max_thickness_pct:
                continue
            
            filtered.append(zone)
        
        return filtered


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    
    # Test data oluştur (daha realistic)
    np.random.seed(42)
    n = 500
    
    # Base price trend
    base = 50000 + np.cumsum(np.random.randn(n) * 50)
    
    # High/Low/Close
    high = base + np.random.rand(n) * 100 + 50
    low = base - np.random.rand(n) * 100 - 50
    close = (high + low) / 2 + np.random.randn(n) * 20
    
    # Detector oluştur
    detector = ZoneDetector(
        min_quality=4.0  # Minimum 4/10 quality
    )
    
    # Zone tespit et
    zones = detector.detect_zones(high, low, close, timeframe="4H", method="both")
    
    print("\n" + "="*80)
    print("ZONE DETECTION RESULTS")
    print("="*80)
    print(f"Total zones found: {len(zones)}")
    print(f"Current price: ${close[-1]:,.2f}")
    print("="*80)
    
    # En iyi 10 zone'u göster
    for i, zone in enumerate(zones[:10], 1):
        print(f"\n[{i}] Zone ID: {zone.id}")
        print(f"    Price Range: ${zone.price_low:,.2f} - ${zone.price_high:,.2f}")
        print(f"    Mid Price: ${zone.price_mid:,.2f}")
        print(f"    Quality: {zone.quality:.1f}/10 ⭐")
        print(f"    Touch Count: {zone.touch_count}")
        print(f"    Thickness: {zone.thickness_pct*100:.2f}%")
        print(f"    Days Since Touch: {zone.days_since_last_touch:.1f}")
        print(f"    Method: {zone.method}")
        print(f"    Distance from Price: {abs(zone.price_mid - close[-1])/close[-1]*100:.2f}%")
    
    print("\n" + "="*80)
    print("QUALITY DISTRIBUTION")
    print("="*80)
    
    # Quality histogram
    quality_buckets = {
        "9-10 (Excellent)": 0,
        "7-8 (Good)": 0,
        "5-6 (Fair)": 0,
        "4 (Minimum)": 0
    }
    
    for zone in zones:
        if zone.quality >= 9:
            quality_buckets["9-10 (Excellent)"] += 1
        elif zone.quality >= 7:
            quality_buckets["7-8 (Good)"] += 1
        elif zone.quality >= 5:
            quality_buckets["5-6 (Fair)"] += 1
        else:
            quality_buckets["4 (Minimum)"] += 1
    
    for bucket, count in quality_buckets.items():
        bar = "█" * count
        print(f"{bucket:20s}: {bar} ({count})")
    
    print("="*80)
    
    # ML Features örneği
    if zones:
        print("\n" + "="*80)
        print("ML FEATURES EXAMPLE (First Zone)")
        print("="*80)
        features = zones[0].get_ml_features()
        for key, value in features.items():
            print(f"{key:30s}: {value:8.4f}")
        print("="*80 + "\n")
