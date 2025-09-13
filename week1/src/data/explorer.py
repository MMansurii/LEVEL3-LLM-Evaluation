import pandas as pd
from typing import Dict, List, Any, Optional

from config.settings import Settings
from src.utils.logger import get_logger
from src.utils.helpers import find_text_columns, find_label_columns

logger = get_logger(__name__)


class DatasetExplorer:
    """Explore dataset structure and content"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.exploration_results = {}
        
    def explore(self) -> Dict[str, Any]:
        """Perform comprehensive dataset exploration"""
        logger.info("Starting dataset exploration...")
        
        for split_name in self.dataset.keys():
            logger.info(f"Exploring {split_name} split...")
            
            df = self.dataset[split_name].to_pandas()
            self.exploration_results[split_name] = self._explore_split(df, split_name)
        
        return self.exploration_results
    
    def _explore_split(self, df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """Explore a single dataset split"""
        results = {
            "name": split_name,
            "num_samples": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        text_columns = find_text_columns(df)
        results["text_columns"] = text_columns
        
        for col in text_columns:
            results[f"{col}_preview"] = self._get_text_preview(df[col])
        
        label_columns = find_label_columns(df)
        results["label_columns"] = label_columns
        
        for col in label_columns:
            results[f"{col}_unique"] = df[col].nunique()
            results[f"{col}_distribution"] = df[col].value_counts().head(10).to_dict()
        
        return results
    
    def _get_text_preview(self, series: pd.Series, n: int = 3) -> List[str]:
        """Get preview of text samples"""
        samples = []
        for text in series.dropna().head(n):
            if len(text) > 200:
                text = text[:200] + "..."
            samples.append(text)
        return samples
