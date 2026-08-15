"""
Data export module for the lie detection application.
Manages export of metrics and analysis results.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class DataExportManager:
    """
    Manages export of analysis data to various formats.
    """
    
    def __init__(self, base_directory: str = 'exports'):
        """
        Initialize export manager.
        
        Args:
            base_directory: Root directory for exports.
        """
        self.base_directory = base_directory
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_directory = os.path.join(
            base_directory, 
            f'session_{self.session_id}'
        )
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for exports."""
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        
        if not os.path.exists(self.session_directory):
            os.makedirs(self.session_directory)
    
    def export_metrics(self, metrics: Dict[str, Any], filename: str = None) -> str:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics.
            filename: Custom filename.
            
        Returns:
            Path to exported file.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f'metrics_{timestamp}.json'
        
        filepath = os.path.join(self.session_directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
        
        return filepath
    
    def export_metrics_series(self, metrics_list: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Export series of metrics to CSV file.
        
        Args:
            metrics_list: List of metric dictionaries.
            filename: Custom filename.
            
        Returns:
            Path to exported file.
        """
        if not metrics_list:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f'metrics_series_{timestamp}.csv'
        
        filepath = os.path.join(self.session_directory, filename)
        
        # Flatten nested dictionaries for CSV
        flattened_data = []
        for metrics in metrics_list:
            flat_dict = {}
            self._flatten_dict(metrics, flat_dict)
            flattened_data.append(flat_dict)
        
        # Get all unique keys
        all_keys = set()
        for item in flattened_data:
            all_keys.update(item.keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(flattened_data)
        
        return filepath
    
    def _flatten_dict(self, data: Dict[str, Any], result: Dict[str, Any], prefix: str = ''):
        """
        Flatten nested dictionary for CSV export.
        
        Args:
            data: Dictionary to flatten.
            result: Result dictionary.
            prefix: Key prefix for nested keys.
        """
        for key, value in data.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_dict(value, result, f"{new_key}_")
            else:
                result[new_key] = value
    
    def export_analysis_report(self, summary: Dict[str, Any]) -> str:
        """
        Generate and export analysis report.
        
        Args:
            summary: Summary statistics and analysis results.
            
        Returns:
            Path to exported report.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'analysis_report_{timestamp}.txt'
        filepath = os.path.join(self.session_directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write("LIE DETECTION ANALYSIS REPORT\n")
            file.write("=" * 50 + "\n\n")
            file.write(f"Session ID: {self.session_id}\n")
            file.write(f"Report Generated: {datetime.now().isoformat()}\n\n")
            
            file.write("SUMMARY STATISTICS\n")
            file.write("-" * 30 + "\n")
            
            for key, value in summary.items():
                if isinstance(value, float):
                    file.write(f"{key}: {value:.4f}\n")
                else:
                    file.write(f"{key}: {value}\n")
        
        return filepath
    
    def generate_session_metadata(self, config: Dict[str, Any]) -> str:
        """
        Generate metadata file for the session.
        
        Args:
            config: Configuration used for the session.
            
        Returns:
            Path to metadata file.
        """
        metadata = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': config
        }
        
        filepath = os.path.join(self.session_directory, 'session_metadata.json')
        
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        
        return filepath