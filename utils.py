"""
Utility functions for the lie detection application.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import csv
import os

@dataclass
class RiskLevel:
    """Risk level classification result."""
    level: str
    color: str
    description: str
    confidence: float

class MetricsCalculator:
    """
    Calculates composite metrics from facial features.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        if weights is None:
            self.weights = {
                'blink_rate': 0.25,
                'pupil_dilation': 0.35,  # Increased weight
                'mouth_activity': 0.15,
                'head_movement': 0.10,
                'gaze_shift': 0.15
            }
        else:
            self.weights = weights
        
        self.history = []
        self.confidence_threshold = 30
    
    def calculate_deception_score(self, metrics: Dict[str, float]) -> float:
        """Calculate deception score from facial metrics."""
        if not metrics or not metrics.get('face_present', False):
            return 0.0
        
        score = 0.0
        for feature, weight in self.weights.items():
            value = metrics.get(feature, 0.0)
            if value > 0.0:
                score += weight * value
        
        return min(score, 1.0)
    
    def calculate_confidence(self, score: float, history_length: int = 30) -> float:
        """Calculate confidence level based on score stability."""
        if history_length < 10:
            return 0.5
        
        stability = 1.0 - min(history_length / 100.0, 0.5)
        base_confidence = 0.6 + (score - 0.5) * 0.4
        confidence = base_confidence * (0.7 + 0.3 * stability)
        return np.clip(confidence, 0.1, 0.95)
    
    def classify_risk(self, score: float) -> RiskLevel:
        """Classify deception risk level."""
        if score < 0.25:
            return RiskLevel(
                level='LOW',
                color='GREEN',
                description='No significant deception indicators',
                confidence=self.calculate_confidence(score)
            )
        elif score < 0.45:
            return RiskLevel(
                level='MEDIUM',
                color='YELLOW',
                description='Some stress indicators detected',
                confidence=self.calculate_confidence(score)
            )
        elif score < 0.65:
            return RiskLevel(
                level='ELEVATED',
                color='ORANGE',
                description='Multiple deception indicators present',
                confidence=self.calculate_confidence(score)
            )
        else:
            return RiskLevel(
                level='HIGH',
                color='RED',
                description='Strong deception indicators detected',
                confidence=self.calculate_confidence(score)
            )
    
    def get_trend_analysis(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze trend in deception scores."""
        if len(scores) < 5:
            return {'trend': 'INSUFFICIENT_DATA'}
        
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        x = np.arange(len(scores_array))
        slope = np.polyfit(x, scores_array, 1)[0]
        
        if slope > 0.02:
            trend = 'INCREASING'
        elif slope < -0.02:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'
        
        return {
            'trend': trend,
            'slope': slope,
            'mean': mean_score,
            'std_dev': std_score,
            'stability': 1.0 - min(std_score / 0.5, 0.9)
        }

class DataExporter:
    """Handles export of metrics data."""
    
    def __init__(self, export_directory: str = 'exports'):
        self.export_directory = export_directory
        self._create_directory()
    
    def _create_directory(self):
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)
    
    def export_to_json(self, data: Dict[str, Any], filename: str = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_{timestamp}.json'
        filepath = os.path.join(self.export_directory, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        return filepath
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        if not data:
            return None
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_{timestamp}.csv'
        filepath = os.path.join(self.export_directory, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return filepath

class SignalProcessor:
    """Processes signals with filtering and smoothing."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = []
    
    def filter_value(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer)
    
    def filter_array(self, values: List[float]) -> np.ndarray:
        if len(values) < self.window_size:
            return np.array(values)
        result = []
        for i in range(len(values)):
            start = max(0, i - self.window_size + 1)
            window = values[start:i + 1]
            result.append(np.mean(window))
        return np.array(result)
    
    def reset(self):
        self.buffer.clear()