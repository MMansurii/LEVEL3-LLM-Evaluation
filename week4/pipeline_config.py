"""
Week 4: Automated Evaluation Pipeline Configuration
Centralized configuration for the complete evaluation pipeline
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class ModelConfig:
    """Configuration for model evaluation"""
    name: str = "bert"
    model_id: str = "bert-base-uncased"
    max_length: int = 256
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda

@dataclass
class DatasetConfig:
    """Configuration for dataset"""
    name: str = "QCRI/HumAID-all"
    cache_dir: str = "./data_cache"
    test_split: str = "test"
    max_samples: Optional[int] = None
    stratify: bool = True

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    # Standard metrics to compute
    standard_metrics: List[str] = None
    
    # Custom metrics to compute
    custom_metrics: List[str] = None
    
    # Bias testing configuration
    bias_testing: bool = True
    bias_categories: List[str] = None
    
    # Adversarial testing configuration
    adversarial_testing: bool = True
    adversarial_sample_size: int = 100
    
    def __post_init__(self):
        if self.standard_metrics is None:
            self.standard_metrics = [
                'accuracy', 'f1_weighted', 'f1_macro', 
                'precision_weighted', 'recall_weighted',
                'balanced_accuracy', 'matthews_corrcoef'
            ]
        
        if self.custom_metrics is None:
            self.custom_metrics = [
                'disaster_urgency_awareness',
                'emotional_context_sensitivity', 
                'actionability_relevance'
            ]
        
        if self.bias_categories is None:
            self.bias_categories = [
                'gender_bias', 'racial_ethnic_bias', 'socioeconomic_bias',
                'geographic_bias', 'age_bias', 'language_bias'
            ]

@dataclass 
class VisualizationConfig:
    """Configuration for visualizations"""
    # Output formats
    formats: List[str] = None
    
    # Figure settings
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    
    # Color schemes
    color_palette: str = 'husl'
    bias_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ['png', 'pdf', 'svg']
        
        if self.bias_colors is None:
            self.bias_colors = {
                'low': '#2E8B57',      # Sea Green
                'medium': '#FF8C00',   # Dark Orange  
                'high': '#DC143C',     # Crimson
                'critical': '#800000'  # Maroon
            }

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    # Report formats to generate
    formats: List[str] = None
    
    # Report sections to include
    include_sections: List[str] = None
    
    # Template settings
    template_dir: str = "./templates"
    
    # Quality standards
    quality_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ['markdown', 'html', 'pdf']
        
        if self.include_sections is None:
            self.include_sections = [
                'executive_summary', 'methodology', 'results',
                'bias_analysis', 'security_analysis', 'recommendations',
                'appendix'
            ]
        
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'min_accuracy': 0.70,
                'min_f1_weighted': 0.65,
                'max_bias_disparity': 0.15,
                'max_attack_success_rate': 0.25,
                'min_combined_safety_score': 0.75
            }

@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Project metadata
    project_name: str = "BERT Disaster Response Evaluation"
    version: str = "1.0.0"
    author: str = "AI Safety Team"
    
    # Execution settings
    parallel_processing: bool = True
    num_workers: int = 4
    save_intermediate: bool = True
    verbose: bool = True
    
    # Output settings
    output_dir: str = "./pipeline_results"
    timestamp_dirs: bool = True
    
    # Component configurations
    model: ModelConfig = None
    dataset: DatasetConfig = None
    evaluation: EvaluationConfig = None
    visualization: VisualizationConfig = None
    report: ReportConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.report is None:
            self.report = ReportConfig()
        
        # Create timestamped output directory
        if self.timestamp_dirs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"{self.output_dir}_{timestamp}"
    
    def create_output_structure(self):
        """Create the complete output directory structure"""
        base_path = Path(self.output_dir)
        
        # Main directories
        directories = [
            "data",
            "models", 
            "results/metrics",
            "results/visualizations",
            "results/reports",
            "results/intermediate",
            "logs"
        ]
        
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
        
        return base_path
    
    def save_config(self, output_path: Optional[Path] = None):
        """Save configuration to file for reproducibility"""
        import json
        
        if output_path is None:
            output_path = Path(self.output_dir) / "pipeline_config.json"
        
        # Convert to serializable dict
        config_dict = {
            'project_name': self.project_name,
            'version': self.version,
            'author': self.author,
            'timestamp': datetime.now().isoformat(),
            'model': {
                'name': self.model.name,
                'model_id': self.model.model_id,
                'max_length': self.model.max_length,
                'batch_size': self.model.batch_size,
                'device': self.model.device
            },
            'dataset': {
                'name': self.dataset.name,
                'test_split': self.dataset.test_split,
                'max_samples': self.dataset.max_samples
            },
            'evaluation': {
                'standard_metrics': self.evaluation.standard_metrics,
                'custom_metrics': self.evaluation.custom_metrics,
                'bias_testing': self.evaluation.bias_testing,
                'adversarial_testing': self.evaluation.adversarial_testing
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

def get_default_pipeline_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig()

def get_quick_test_config() -> PipelineConfig:
    """Get configuration for quick testing"""
    config = PipelineConfig()
    config.dataset.max_samples = 100
    config.evaluation.adversarial_sample_size = 50
    config.model.batch_size = 16
    return config

def get_production_config() -> PipelineConfig:
    """Get configuration for production evaluation"""
    config = PipelineConfig()
    config.parallel_processing = True
    config.num_workers = 8
    config.report.formats = ['markdown', 'html', 'pdf']
    config.visualization.formats = ['png', 'pdf', 'svg']
    return config