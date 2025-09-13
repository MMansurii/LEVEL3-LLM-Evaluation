from pathlib import Path
import os
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Global configuration settings"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    BASE_DIR_output = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR_output / "result" / "outputs"  # This ensures outputs are in the project folder
    DATA_DIR = BASE_DIR_output / "result"/ "data"
    REPORTS_DIR = BASE_DIR_output / "result"/ "reports"
    VIZ_DIR = BASE_DIR_output / "result"/ "visualizations"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Dataset defaults
    DEFAULT_DATASET = "QCRI/HumAID-all"
    CACHE_DIR = DATA_DIR / "cache"
    
    # Analysis settings
    SAMPLE_SIZE = 1000
    RANDOM_SEED = 42
    
    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 100
    STYLE = "whitegrid"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.OUTPUT_DIR, cls.DATA_DIR, cls.REPORTS_DIR, cls.VIZ_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_dataset_config(cls) -> Dict[str, Any]:
        """Load dataset configuration from YAML"""
        config_file = cls.BASE_DIR / "config" / "datasets.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
