"""
Zone Detector - Parça 1
Based on: pa-strateji2 Parça 1

Features:
- ZigZag++ detection
- Swing High/Low detection
- Zone merging & filtering
- Multi-timeframe support
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime


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
    quality: float = 0.0  # 0-10 (sonra hesaplanacak)


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
    """
    
    def __init__(
        self,
        zigzag_depth: int = 12,
        zigzag_deviation: float = 5.0,
        zigzag_backstep: int = 2,
        swing_strength: int = 5,
        min_touches: int = 2,
        max_thickness_pct: float = 0.015,  # %1.5
        merge_tolerance_pct: float = 0.015  # %1.5
    ):
        self.zigzag_depth = zigzag_depth
        self.zigzag_deviation = zigzag_deviation
        self.zigzag_backstep = zigzag_backstep
        self.swing_strength = swing_strength
        self.min_touches = min_touches
        self.max_thickness_pct = max_thickness_pct
        self.merge_tolerance_pct = merge_tolerance_pct
    
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
            List of zones
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
        
        # Sort by proximity to current price
        zones = self._sort_zones(zones, close[-1])
        
        return zones
    
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
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"zz_{timeframe}_{pivot.index}",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=pivot.price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / pivot.price,
                    last_touch_index=len(close) - 1,
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
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"sw_{timeframe}_{idx}_h",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / price,
                    last_touch_index=len(high) - 1,
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
            
            if touches >= self.min_touches:
                zone = Zone(
                    id=f"sw_{timeframe}_{idx}_l",
                    price_low=zone_low,
                    price_high=zone_high,
                    price_mid=price,
                    touch_count=touches,
                    thickness_pct=(zone_high - zone_low) / price,
                    last_touch_index=len(low) - 1,
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
    
    def _sort_zones(self, zones: List[Zone], current_price: float) -> List[Zone]:
        """Zone'ları mevcut fiyata yakınlığa göre sırala"""
        return sorted(zones, key=lambda z: abs(z.price_mid - current_price))


# Test
if __name__ == "__main__":
    detector = ZoneDetector()
    
    # Dummy data
    high = np.random.rand(200) * 100 + 50000
    low = high - np.random.rand(200) * 50
    close = (high + low) / 2
    
    zones = detector.detect_zones(high, low, close, timeframe="4H", method="both")
    
    print(f"\n✅ Found {len(zones)} zones")
    for zone in zones[:5]:
        print(f"  Zone: ${zone.price_low:.0f} - ${zone.price_high:.0f} ({zone.method})")
