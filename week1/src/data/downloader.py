import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, DatasetDict, load_dataset_builder
from datasets.exceptions import ExpectedMoreSplitsError

from config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetDownloader:
    """Handle dataset downloading and caching"""
    
    def __init__(self, dataset_name: str = None):
        self.dataset_name = dataset_name or Settings.DEFAULT_DATASET
        self.cache_dir = Settings.CACHE_DIR
        self.dataset: Optional[DatasetDict] = None
        
    def download(self) -> DatasetDict:
        """Download dataset from HuggingFace"""
        from datasets.exceptions import ExpectedMoreSplitsError
        
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            
            try:
                # Try normal download first
                self.dataset = load_dataset(
                    self.dataset_name,
                    cache_dir=str(self.cache_dir)
                )
            except ExpectedMoreSplitsError as e:
                # If we get the 'dev' split error, ignore verification
                logger.warning(f"Split verification issue: {e}")
                logger.info("Attempting to load with verification disabled...")
                
                # Load with verification disabled
                from datasets import load_dataset_builder
                
                # Get the builder
                builder = load_dataset_builder(
                    self.dataset_name,
                    cache_dir=str(self.cache_dir)
                )
                
                # Download and prepare without verification
                builder.download_and_prepare(
                    verification_mode="no_checks"  # This skips split verification
                )
                
                # Load the dataset
                self.dataset = builder.as_dataset()
                
                # If there's validation but no dev, create alias
                if 'validation' in self.dataset and 'dev' not in self.dataset:
                    self.dataset['dev'] = self.dataset['validation']
                    logger.info("Created 'dev' as alias for 'validation'")
            
            logger.info(f"Successfully downloaded {self.dataset_name}")
            logger.info(f"Available splits: {list(self.dataset.keys())}")
            
            for split in self.dataset.keys():
                logger.info(f"  {split}: {len(self.dataset[split])} samples")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic dataset information"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call download() first.")
        
        return {
            "name": self.dataset_name,
            "splits": list(self.dataset.keys()),
            "split_sizes": {
                split: len(self.dataset[split]) 
                for split in self.dataset.keys()
            }
        }
