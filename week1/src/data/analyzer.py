import pandas as pd
import numpy as np
from typing import Dict, Any, List

from config.settings import Settings
from src.utils.logger import get_logger
from src.utils.helpers import calculate_text_statistics, find_text_columns, find_label_columns

logger = get_logger(__name__)


class DatasetAnalyzer:
    """Perform statistical analysis on dataset"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.analysis_results = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        logger.info("Starting statistical analysis...")
        
        for split_name in self.dataset.keys():
            logger.info(f"Analyzing {split_name} split...")
            
            df = self.dataset[split_name].to_pandas()
            self.analysis_results[split_name] = self._analyze_split(df, split_name)
        
        self.analysis_results["cross_split"] = self._cross_split_analysis()
        
        return self.analysis_results
    
    def _analyze_split(self, df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """Analyze a single split"""
        results = {
            "split_name": split_name,
            "num_samples": len(df)
        }
        
        text_columns = find_text_columns(df)
        for col in text_columns:
            results[f"{col}_stats"] = calculate_text_statistics(df[col])
        
        label_columns = find_label_columns(df)
        for col in label_columns:
            results[f"{col}_analysis"] = self._analyze_labels(df[col])
        
        return results
    
    def _analyze_labels(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze label distribution"""
        value_counts = series.value_counts()
        
        max_count = value_counts.max()
        min_count = value_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        probabilities = value_counts / len(series)
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
        
        return {
            "num_classes": len(value_counts),
            "distribution": value_counts.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "entropy": entropy,
            "majority_class": str(value_counts.index[0]),
            "minority_class": str(value_counts.index[-1])
        }
    
    def _cross_split_analysis(self) -> Dict[str, Any]:
        """Analyze across splits"""
        return {
            "total_samples": sum(
                len(self.dataset[split]) 
                for split in self.dataset.keys()
            ),
            "split_ratios": {
                split: len(self.dataset[split]) / sum(
                    len(self.dataset[s]) for s in self.dataset.keys()
                )
                for split in self.dataset.keys()
            }
        }
