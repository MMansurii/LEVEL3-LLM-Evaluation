"""
Week 4: Data Manager
Handles all data loading, preprocessing, and management
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from collections import Counter

from pipeline_config import DatasetConfig

logger = logging.getLogger(__name__)

class DataManager:
    """Manages dataset loading, preprocessing, and access"""
    
    def __init__(self, config: DatasetConfig, output_path: Path):
        self.config = config
        self.output_path = output_path
        self.dataset = None
        self.label_names = None
        self.dataset_info = {}
        
        # Data splits
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_and_prepare_data(self) -> Dict[str, Any]:
        """Load and prepare the complete dataset"""
        
        logger.info(f"Loading dataset: {self.config.name}")
        
        try:
            # Load dataset from HuggingFace
            self.dataset = load_dataset(
                self.config.name,
                cache_dir=self.config.cache_dir
            )
            
            # Extract label information
            if 'train' in self.dataset:
                self.label_names = self.dataset['train'].features['class_label'].names
            
            # Analyze dataset structure
            self.dataset_info = self._analyze_dataset()
            
            # Prepare data splits
            self._prepare_data_splits()
            
            # Save dataset information
            self._save_dataset_info()
            
            logger.info(f"✅ Dataset loaded successfully")
            logger.info(f"   Total samples: {self.dataset_info['total_samples']:,}")
            logger.info(f"   Classes: {len(self.label_names)}")
            
            return self.dataset_info
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset structure and statistics"""
        
        info = {
            'name': self.config.name,
            'splits': list(self.dataset.keys()),
            'num_classes': len(self.label_names),
            'class_names': self.label_names,
            'total_samples': sum(len(split) for split in self.dataset.values())
        }
        
        # Split-specific information
        info['split_info'] = {}
        for split_name, split_data in self.dataset.items():
            split_info = {
                'size': len(split_data),
                'class_distribution': self._get_class_distribution(split_data['class_label'])
            }
            info['split_info'][split_name] = split_info
        
        # Text statistics (using training set)
        if 'train' in self.dataset:
            train_texts = self.dataset['train']['tweet_text']
            info['text_stats'] = self._analyze_text_statistics(train_texts)
        
        return info
    
    def _get_class_distribution(self, labels: List[int]) -> Dict[str, Any]:
        """Get class distribution statistics"""
        
        label_counts = Counter(labels)
        total = len(labels)
        
        distribution = {
            'counts': dict(label_counts),
            'percentages': {k: (v/total)*100 for k, v in label_counts.items()},
            'most_common_class': label_counts.most_common(1)[0][0] if label_counts else None,
            'least_common_class': label_counts.most_common()[-1][0] if label_counts else None,
            'imbalance_ratio': label_counts.most_common(1)[0][1] / label_counts.most_common()[-1][1] if label_counts else 1
        }
        
        return distribution
    
    def _analyze_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text characteristics"""
        
        # Length statistics
        word_lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        text_stats = {
            'word_length': {
                'mean': np.mean(word_lengths),
                'median': np.median(word_lengths),
                'min': np.min(word_lengths),
                'max': np.max(word_lengths),
                'std': np.std(word_lengths),
                'p95': np.percentile(word_lengths, 95)
            },
            'char_length': {
                'mean': np.mean(char_lengths),
                'median': np.median(char_lengths),
                'min': np.min(char_lengths),
                'max': np.max(char_lengths),
                'std': np.std(char_lengths),
                'p95': np.percentile(char_lengths, 95)
            },
            'total_texts': len(texts)
        }
        
        return text_stats
    
    def _prepare_data_splits(self):
        """Prepare train/validation/test splits"""
        
        # Use existing splits if available
        if self.config.test_split in self.dataset:
            self.test_data = self._prepare_split_data(self.config.test_split)
            logger.info(f"Using existing test split: {len(self.test_data['texts'])} samples")
        
        if 'train' in self.dataset:
            # Split training data into train/val if no validation split exists
            if 'validation' in self.dataset:
                self.train_data = self._prepare_split_data('train')
                self.val_data = self._prepare_split_data('validation')
            else:
                # Create train/val split from training data
                full_train_data = self._prepare_split_data('train')
                
                # Stratified split to maintain class distribution
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    full_train_data['texts'],
                    full_train_data['labels'],
                    test_size=0.2,
                    stratify=full_train_data['labels'] if self.config.stratify else None,
                    random_state=42
                )
                
                self.train_data = {'texts': train_texts, 'labels': train_labels}
                self.val_data = {'texts': val_texts, 'labels': val_labels}
        
        # Apply sample limits if specified
        if self.config.max_samples:
            for data_split in [self.train_data, self.val_data, self.test_data]:
                if data_split and len(data_split['texts']) > self.config.max_samples:
                    # Sample while maintaining class distribution
                    indices = self._stratified_sample_indices(
                        data_split['labels'], self.config.max_samples
                    )
                    data_split['texts'] = [data_split['texts'][i] for i in indices]
                    data_split['labels'] = [data_split['labels'][i] for i in indices]
    
    def _prepare_split_data(self, split_name: str) -> Dict[str, List]:
        """Prepare data for a specific split"""
        
        split_data = self.dataset[split_name]
        
        return {
            'texts': split_data['tweet_text'],
            'labels': split_data['class_label']
        }
    
    def _stratified_sample_indices(self, labels: List[int], sample_size: int) -> List[int]:
        """Get stratified sample indices"""
        
        # Calculate samples per class
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        # Proportional sampling
        samples_per_class = {}
        remaining_samples = sample_size
        
        for label, count in label_counts.items():
            proportion = count / total_samples
            class_samples = min(int(proportion * sample_size), count)
            samples_per_class[label] = class_samples
            remaining_samples -= class_samples
        
        # Distribute remaining samples
        while remaining_samples > 0:
            for label in samples_per_class:
                if remaining_samples > 0 and samples_per_class[label] < label_counts[label]:
                    samples_per_class[label] += 1
                    remaining_samples -= 1
        
        # Sample indices for each class
        selected_indices = []
        label_indices = {label: [i for i, l in enumerate(labels) if l == label] 
                        for label in label_counts}
        
        for label, num_samples in samples_per_class.items():
            class_indices = label_indices[label]
            selected = np.random.choice(class_indices, size=num_samples, replace=False)
            selected_indices.extend(selected)
        
        return sorted(selected_indices)
    
    def _save_dataset_info(self):
        """Save dataset information to file"""
        
        import json
        
        info_file = self.output_path / "data" / "dataset_info.json"
        
        # Make serializable
        serializable_info = {}
        for key, value in self.dataset_info.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                serializable_info[key] = value
            elif hasattr(value, 'tolist'):  # numpy arrays
                serializable_info[key] = value.tolist()
            else:
                serializable_info[key] = str(value)
        
        with open(info_file, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        logger.info(f"Dataset info saved to: {info_file}")
    
    def get_test_data(self) -> Dict[str, List]:
        """Get test data for evaluation"""
        
        if self.test_data is None:
            raise ValueError("Test data not available. Run load_and_prepare_data() first.")
        
        return self.test_data
    
    def get_train_data(self) -> Dict[str, List]:
        """Get training data"""
        
        if self.train_data is None:
            raise ValueError("Training data not available. Run load_and_prepare_data() first.")
        
        return self.train_data
    
    def get_validation_data(self) -> Dict[str, List]:
        """Get validation data"""
        
        if self.val_data is None:
            raise ValueError("Validation data not available. Run load_and_prepare_data() first.")
        
        return self.val_data
    
    def get_label_names(self) -> List[str]:
        """Get class label names"""
        
        if self.label_names is None:
            raise ValueError("Label names not available. Run load_and_prepare_data() first.")
        
        return self.label_names
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get complete dataset information"""
        
        return self.dataset_info
    
    def get_sample_texts(self, split: str = 'test', num_samples: int = 10) -> List[str]:
        """Get sample texts for testing purposes"""
        
        if split == 'test' and self.test_data:
            texts = self.test_data['texts']
        elif split == 'train' and self.train_data:
            texts = self.train_data['texts']
        elif split == 'val' and self.val_data:
            texts = self.val_data['texts']
        else:
            raise ValueError(f"Split '{split}' not available")
        
        # Return random sample
        indices = np.random.choice(len(texts), size=min(num_samples, len(texts)), replace=False)
        return [texts[i] for i in indices]
    
    def get_class_examples(self, class_name: str, split: str = 'test', num_examples: int = 5) -> List[Dict]:
        """Get example texts for a specific class"""
        
        if class_name not in self.label_names:
            raise ValueError(f"Class '{class_name}' not found in label names")
        
        class_idx = self.label_names.index(class_name)
        
        # Get data for the specified split
        if split == 'test' and self.test_data:
            texts, labels = self.test_data['texts'], self.test_data['labels']
        elif split == 'train' and self.train_data:
            texts, labels = self.train_data['texts'], self.train_data['labels']
        else:
            raise ValueError(f"Split '{split}' not available")
        
        # Find examples of the specified class
        class_examples = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            if label == class_idx:
                class_examples.append({
                    'text': text,
                    'label': class_name,
                    'index': i
                })
                
                if len(class_examples) >= num_examples:
                    break
        
        return class_examples
    
    def create_data_summary(self) -> Dict[str, Any]:
        """Create comprehensive data summary for reporting"""
        
        summary = {
            'dataset_name': self.config.name,
            'total_samples': self.dataset_info['total_samples'],
            'num_classes': len(self.label_names),
            'class_names': self.label_names,
            'splits': {}
        }
        
        # Add split summaries
        for split_name in ['train', 'test', 'val']:
            split_data = getattr(self, f"{split_name}_data")
            if split_data:
                summary['splits'][split_name] = {
                    'size': len(split_data['texts']),
                    'class_distribution': self._get_class_distribution(split_data['labels'])
                }
        
        # Add text statistics
        if 'text_stats' in self.dataset_info:
            summary['text_statistics'] = self.dataset_info['text_stats']
        
        return summary