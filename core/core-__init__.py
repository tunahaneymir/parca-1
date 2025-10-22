"""
Core PA Detection Modules - Parça 1

This module contains the foundational Price Action detection components:
- Trend Detection (EMA-based with sideways filtering)
- Zone Detection (ZigZag++ and Swing High/Low)
- Zone Quality Scoring (0-10 rating system)

Usage:
    from core import TrendDetector, ZoneDetector, ZoneQualityScorer
    
    # Initialize with config
    trend_detector = TrendDetector(config)
    zone_detector = ZoneDetector(config)
    
    # Detect trend
    trend_result = trend_detector.detect(close, high, low)
    
    # Detect zones
    zones = zone_detector.detect_zones(high, low, close, timeframe="4H")
"""

from .trend_detector import TrendDetector, TrendResult
from .zone_detector import ZoneDetector, Zone, ZigZagPoint
from .zone_quality_scorer import ZoneQualityScorer

__all__ = [
    # Trend Detection
    'TrendDetector',
    'TrendResult',
    
    # Zone Detection
    'ZoneDetector',
    'Zone',
    'ZigZagPoint',
    
    # Quality Scoring
    'ZoneQualityScorer',
]

__version__ = '1.0.0'
