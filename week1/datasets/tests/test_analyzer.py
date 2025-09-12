import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from src.data.analyzer import DatasetAnalyzer


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer class"""
    
    def test_analyze_labels(self):
        """Test label analysis"""
        mock_dataset = {
            'train': Mock()
        }
        
        analyzer = DatasetAnalyzer(mock_dataset)
        
        labels = pd.Series(['A', 'A', 'B', 'B', 'B', 'C'])
        result = analyzer._analyze_labels(labels)
        
        assert result['num_classes'] == 3
        assert result['majority_class'] == 'B'
        assert result['minority_class'] == 'C'
