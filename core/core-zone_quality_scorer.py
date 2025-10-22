"""
Zone Quality Scorer - Parça 1
Based on: pa-strateji2 Parça 1

Features:
- Zone quality scoring (0-10)
- Touch count scoring (0-4 points)
- Thickness scoring (0-3 points)
- Recency scoring (0-3 points)
- Days since last touch calculation
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zone_detector import Zone


class ZoneQualityScorer:
    """
    Zone Quality Scoring System
    
    Scoring System (0-10):
    - Touch count: 0-4 points
    - Thickness: 0-3 points
    - Recency: 0-3 points
    
    Total: Maximum 10 points
    """
    
    def __init__(self):
        """Initialize scorer with default thresholds"""
        # Touch count thresholds
        self.touch_perfect = 2      # 4 points
        self.touch_good = 3         # 3 points
        self.touch_medium = 4       # 2 points
        # 5+ touches = 1 point (too tested)
        
        # Thickness thresholds (percentage)
        self.thickness_excellent = 0.5  # %0.5 -> 3 points
        self.thickness_good = 1.0       # %1.0 -> 2 points
        self.thickness_fair = 1.5       # %1.5 -> 1 point
        # > 1.5% = 0 points
        
        # Recency thresholds (days)
        self.recency_fresh = 7      # < 7 days -> 3 points
        self.recency_good = 30      # < 30 days -> 2 points
        # > 30 days = 1 point
    
    def calculate_quality(
        self,
        zone: 'Zone',
        current_index: int,
        candle_time_minutes: int = 240
    ) -> float:
        """
        Calculate zone quality score (0-10)
        
        Args:
            zone: Zone object to score
            current_index: Current candle index
            candle_time_minutes: Minutes per candle (4H=240, 1H=60, 15M=15)
            
        Returns:
            Quality score (0-10)
        """
        score = 0.0
        
        # ═══════════════════════════════════════════════════════════
        # 1. TOUCH COUNT SCORING (0-4 points)
        # ═══════════════════════════════════════════════════════════
        score += self._score_touch_count(zone.touch_count)
        
        # ═══════════════════════════════════════════════════════════
        # 2. THICKNESS SCORING (0-3 points)
        # ═══════════════════════════════════════════════════════════
        thickness_percent = zone.thickness_pct * 100
        score += self._score_thickness(thickness_percent)
        
        # ═══════════════════════════════════════════════════════════
        # 3. RECENCY SCORING (0-3 points)
        # ═══════════════════════════════════════════════════════════
        days_since = self._calculate_days_since_touch(
            current_index,
            zone.last_touch_index,
            candle_time_minutes
        )
        zone.days_since_last_touch = days_since  # Update zone object
        score += self._score_recency(days_since)
        
        return min(score, 10.0)  # Cap at 10
    
    def _score_touch_count(self, touch_count: int) -> float:
        """
        Score based on touch count
        
        Logic:
        - 2 touches = Perfect! (4 points) - Fresh zone
        - 3 touches = Good (3 points) - Tested few times
        - 4 touches = Medium (2 points) - Getting tested
        - 5+ touches = Weak (1 point) - Too tested, may break
        
        Args:
            touch_count: Number of touches
            
        Returns:
            Score (0-4 points)
        """
        if touch_count == self.touch_perfect:
            return 4.0  # Perfect! Fresh zone
        elif touch_count == self.touch_good:
            return 3.0  # Good
        elif touch_count == self.touch_medium:
            return 2.0  # Medium
        else:  # 5+ touches
            return 1.0  # Weak (too tested)
    
    def _score_thickness(self, thickness_percent: float) -> float:
        """
        Score based on zone thickness
        
        Logic:
        - < 0.5% = Excellent (3 points) - Very precise level
        - < 1.0% = Good (2 points) - Acceptable thickness
        - < 1.5% = Fair (1 point) - Thick but usable
        - > 1.5% = Poor (0 points) - Too thick
        
        Args:
            thickness_percent: Zone thickness as percentage
            
        Returns:
            Score (0-3 points)
        """
        if thickness_percent < self.thickness_excellent:
            return 3.0  # Excellent! Very precise
        elif thickness_percent < self.thickness_good:
            return 2.0  # Good
        elif thickness_percent < self.thickness_fair:
            return 1.0  # Fair
        else:
            return 0.0  # Too thick
    
    def _score_recency(self, days_since: float) -> float:
        """
        Score based on recency (how fresh is the zone?)
        
        Logic:
        - < 7 days = Fresh (3 points) - Market remembers
        - < 30 days = Good (2 points) - Still relevant
        - > 30 days = Old (1 point) - Questionable validity
        
        Args:
            days_since: Days since last touch
            
        Returns:
            Score (0-3 points)
        """
        if days_since < self.recency_fresh:
            return 3.0  # Fresh! Recent test
        elif days_since < self.recency_good:
            return 2.0  # Good, within month
        else:
            return 1.0  # Old, but not worthless
    
    def _calculate_days_since_touch(
        self,
        current_index: int,
        last_touch_index: int,
        candle_time_minutes: int
    ) -> float:
        """
        Calculate days since last touch
        
        Args:
            current_index: Current candle index
            last_touch_index: Last touch candle index
            candle_time_minutes: Minutes per candle
            
        Returns:
            Days since last touch
        """
        candles_since = current_index - last_touch_index
        minutes_since = candles_since * candle_time_minutes
        days_since = minutes_since / (60 * 24)
        
        return days_since
    
    def get_quality_description(self, quality: float) -> str:
        """
        Get textual description of quality score
        
        Args:
            quality: Quality score (0-10)
            
        Returns:
            Description string
        """
        if quality >= 9:
            return "EXCELLENT ⭐⭐⭐"
        elif quality >= 7:
            return "GOOD ⭐⭐"
        elif quality >= 5:
            return "FAIR ⭐"
        elif quality >= 4:
            return "MINIMUM (Acceptable)"
        else:
            return "POOR (Below threshold)"
    
    def get_quality_breakdown(
        self,
        zone: 'Zone',
        current_index: int,
        candle_time_minutes: int = 240
    ) -> dict:
        """
        Get detailed quality breakdown
        
        Args:
            zone: Zone object
            current_index: Current candle index
            candle_time_minutes: Minutes per candle
            
        Returns:
            Dictionary with breakdown
        """
        # Calculate components
        touch_score = self._score_touch_count(zone.touch_count)
        thickness_percent = zone.thickness_pct * 100
        thickness_score = self._score_thickness(thickness_percent)
        days_since = self._calculate_days_since_touch(
            current_index,
            zone.last_touch_index,
            candle_time_minutes
        )
        recency_score = self._score_recency(days_since)
        
        total_score = touch_score + thickness_score + recency_score
        
        return {
            'total_score': total_score,
            'description': self.get_quality_description(total_score),
            'breakdown': {
                'touch_count': {
                    'value': zone.touch_count,
                    'score': touch_score,
                    'max': 4.0
                },
                'thickness': {
                    'value': thickness_percent,
                    'score': thickness_score,
                    'max': 3.0
                },
                'recency': {
                    'value': days_since,
                    'score': recency_score,
                    'max': 3.0
                }
            }
        }


# ═══════════════════════════════════════════════════════════
# ÖRNEK KULLANIM
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataclasses import dataclass
    
    # Mock Zone for testing
    @dataclass
    class MockZone:
        touch_count: int
        thickness_pct: float
        last_touch_index: int
        days_since_last_touch: float = 0.0
    
    # Create scorer
    scorer = ZoneQualityScorer()
    
    print("\n" + "="*60)
    print("ZONE QUALITY SCORING EXAMPLES")
    print("="*60)
    
    # ═════════════════════════════════════════════════════════
    # EXAMPLE 1: EXCELLENT ZONE (10/10)
    # ═════════════════════════════════════════════════════════
    zone1 = MockZone(
        touch_count=2,           # Perfect! (4 points)
        thickness_pct=0.004,     # 0.4% - Excellent (3 points)
        last_touch_index=96      # 4 candles ago at 4H
    )
    
    current_index = 100
    quality1 = scorer.calculate_quality(zone1, current_index, candle_time_minutes=240)
    breakdown1 = scorer.get_quality_breakdown(zone1, current_index, 240)
    
    print("\n[EXAMPLE 1: EXCELLENT ZONE]")
    print(f"Touch Count: {zone1.touch_count}")
    print(f"Thickness: {zone1.thickness_pct*100:.1f}%")
    print(f"Days Since: {zone1.days_since_last_touch:.1f}")
    print(f"\nQuality Score: {quality1:.1f}/10")
    print(f"Description: {breakdown1['description']}")
    print(f"\nBreakdown:")
    print(f"  Touch Count: {breakdown1['breakdown']['touch_count']['score']:.1f}/4")
    print(f"  Thickness:   {breakdown1['breakdown']['thickness']['score']:.1f}/3")
    print(f"  Recency:     {breakdown1['breakdown']['recency']['score']:.1f}/3")
    
    # ═════════════════════════════════════════════════════════
    # EXAMPLE 2: GOOD ZONE (7/10)
    # ═════════════════════════════════════════════════════════
    zone2 = MockZone(
        touch_count=3,           # Good (3 points)
        thickness_pct=0.006,     # 0.6% - Good (2 points)
        last_touch_index=40      # 60 candles ago at 4H = 10 days
    )
    
    quality2 = scorer.calculate_quality(zone2, current_index, 240)
    breakdown2 = scorer.get_quality_breakdown(zone2, current_index, 240)
    
    print("\n[EXAMPLE 2: GOOD ZONE]")
    print(f"Touch Count: {zone2.touch_count}")
    print(f"Thickness: {zone2.thickness_pct*100:.1f}%")
    print(f"Days Since: {zone2.days_since_last_touch:.1f}")
    print(f"\nQuality Score: {quality2:.1f}/10")
    print(f"Description: {breakdown2['description']}")
    
    # ═════════════════════════════════════════════════════════
    # EXAMPLE 3: WEAK ZONE (4/10)
    # ═════════════════════════════════════════════════════════
    zone3 = MockZone(
        touch_count=6,           # Too many (1 point)
        thickness_pct=0.018,     # 1.8% - Too thick (0 points)
        last_touch_index=10      # 90 candles ago = 15 days
    )
    
    quality3 = scorer.calculate_quality(zone3, current_index, 240)
    breakdown3 = scorer.get_quality_breakdown(zone3, current_index, 240)
    
    print("\n[EXAMPLE 3: WEAK ZONE]")
    print(f"Touch Count: {zone3.touch_count}")
    print(f"Thickness: {zone3.thickness_pct*100:.1f}%")
    print(f"Days Since: {zone3.days_since_last_touch:.1f}")
    print(f"\nQuality Score: {quality3:.1f}/10")
    print(f"Description: {breakdown3['description']}")
    
    print("\n" + "="*60)
    print("✅ Zone Quality Scorer working correctly!")
    print("="*60 + "\n")
