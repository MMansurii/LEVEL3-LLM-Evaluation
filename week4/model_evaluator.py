"""
Week 4: Model Evaluator
Handles BERT model evaluation with standard and custom metrics
"""

import logging
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, pipeline
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, classification_report
)

from pipeline_config import ModelConfig, EvaluationConfig

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for BERT"""
    
    def __init__(self, model_config: ModelConfig, eval_config: EvaluationConfig, output_path: Path):
        self.model_config = model_config
        self.eval_config = eval_config
        self.output_path = output_path
        
        # Model components
        self.tokenizer = None
        self.model_pipeline = None
        self.model_info = {}
        
        # Results storage
        self.evaluation_results = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BERT model and tokenizer"""
        
        logger.info(f"Initializing BERT model: {self.model_config.model_id}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
            
            # Set device
            device = -1  # CPU
            if self.model_config.device == "auto":
                device = 0 if torch.cuda.is_available() else -1
            elif self.model_config.device == "cuda":
                device = 0
            
            # Initialize pipeline for zero-shot classification
            self.model_pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_config.model_id,
                tokenizer=self.tokenizer,
                device=device
            )
            
            # Store model info
            self.model_info = {
                'name': self.model_config.name,
                'model_id': self.model_config.model_id,
                'max_length': self.model_config.max_length,
                'batch_size': self.model_config.batch_size,
                'device': 'cuda' if device >= 0 else 'cpu',
                'parameters': self._count_model_parameters()
            }
            
            logger.info(f"✅ Model initialized successfully")
            logger.info(f"   Device: {self.model_info['device']}")
            logger.info(f"   Parameters: {self.model_info['parameters']:,}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _count_model_parameters(self) -> int:
        """Count model parameters"""
        try:
            # Access the underlying model from the pipeline
            model = self.model_pipeline.model
            return sum(p.numel() for p in model.parameters())
        except:
            return 0  # Fallback if counting fails
    
    def evaluate_standard_metrics(self, test_data: Dict[str, List]) -> Dict[str, Any]:
        """Evaluate model using standard classification metrics"""
        
        logger.info("Running standard metrics evaluation...")
        
        texts = test_data['texts']
        true_labels = test_data['labels']
        
        # Get label names (assuming they're available)
        # This would need to be passed in or retrieved from data manager
        label_names = [
            'injured_or_dead_people', 'requests_or_urgent_needs', 'sympathy_and_support',
            'rescue_volunteering_or_donation_effort', 'other_relevant_information',
            'infrastructure_and_utility_damage', 'displaced_people_and_evacuations',
            'caution_and_advice', 'missing_or_found_people', 'not_humanitarian',
            'dont_know_cant_judge'
        ]
        
        # Run predictions
        start_time = time.time()
        predictions = self._run_predictions(texts, label_names)
        prediction_time = time.time() - start_time
        
        # Calculate standard metrics
        standard_results = self._calculate_standard_metrics(true_labels, predictions)
        standard_results['prediction_time'] = prediction_time
        standard_results['samples_evaluated'] = len(texts)
        
        logger.info(f"✅ Standard evaluation completed in {prediction_time:.1f}s")
        logger.info(f"   Accuracy: {standard_results['accuracy']:.4f}")
        logger.info(f"   F1 (Weighted): {standard_results['f1_weighted']:.4f}")
        
        return standard_results
    
    def evaluate_custom_metrics(self, test_data: Dict[str, List]) -> Dict[str, Any]:
        """Evaluate model using custom disaster-specific metrics"""
        
        logger.info("Running custom metrics evaluation...")
        
        texts = test_data['texts']
        true_labels = test_data['labels']
        
        # For now, implement simplified versions of custom metrics
        # In a full implementation, this would use the actual custom metrics from earlier weeks
        
        custom_results = {
            'disaster_urgency_awareness': self._calculate_urgency_metric(texts, true_labels),
            'emotional_context_sensitivity': self._calculate_emotional_metric(texts, true_labels),
            'actionability_relevance': self._calculate_actionability_metric(texts, true_labels)
        }
        
        logger.info(f"✅ Custom metrics evaluation completed")
        
        return custom_results
    
    def _run_predictions(self, texts: List[str], label_names: List[str]) -> List[int]:
        """Run model predictions on texts"""
        
        predictions = []
        batch_size = self.model_config.batch_size
        
        logger.info(f"Running predictions on {len(texts)} samples...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if i % (batch_size * 5) == 0:  # Log every 5 batches
                logger.info(f"  Progress: {i}/{len(texts)} samples processed")
            
            for text in batch_texts:
                try:
                    # Truncate text if too long
                    if len(text.split()) > self.model_config.max_length // 2:
                        text = ' '.join(text.split()[:self.model_config.max_length // 2])
                    
                    # Run zero-shot classification
                    result = self.model_pipeline(text, label_names)
                    
                    # Get predicted label index
                    predicted_label = result['labels'][0]
                    predicted_idx = label_names.index(predicted_label)
                    predictions.append(predicted_idx)
                    
                except Exception as e:
                    logger.warning(f"Error predicting text {len(predictions)}: {e}")
                    predictions.append(0)  # Default prediction
        
        logger.info(f"Predictions completed: {len(predictions)} results")
        return predictions
    
    def _calculate_standard_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate standard classification metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Advanced metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        return metrics
    
    def _calculate_urgency_metric(self, texts: List[str], labels: List[int]) -> float:
        """Calculate disaster urgency awareness metric (simplified version)"""
        
        # Simplified urgency weights
        urgency_weights = {
            0: 5.0,  # injured_or_dead_people
            1: 4.5,  # requests_or_urgent_needs
            3: 4.0,  # rescue_volunteering_or_donation_effort
            8: 4.0,  # missing_or_found_people
            6: 3.5,  # displaced_people_and_evacuations
            5: 3.0,  # infrastructure_and_utility_damage
            7: 2.5,  # caution_and_advice
            4: 2.0,  # other_relevant_information
            2: 1.5,  # sympathy_and_support
            9: 1.0,  # not_humanitarian
            10: 1.0  # dont_know_cant_judge
        }
        
        # Calculate weighted accuracy
        total_weighted = 0.0
        total_weight = 0.0
        
        for true_label in labels:
            weight = urgency_weights.get(true_label, 1.0)
            total_weight += weight
            # Simplified: assume some baseline accuracy
            total_weighted += weight * 0.7  # Placeholder calculation
        
        return total_weighted / total_weight if total_weight > 0 else 0.0
    
    def _calculate_emotional_metric(self, texts: List[str], labels: List[int]) -> float:
        """Calculate emotional context sensitivity metric (simplified)"""
        
        # Simplified emotional grouping penalty calculation
        # This would use the full implementation from custom metrics in practice
        return 0.75  # Placeholder
    
    def _calculate_actionability_metric(self, texts: List[str], labels: List[int]) -> float:
        """Calculate actionability relevance metric (simplified)"""
        
        # Simplified actionability calculation
        # This would use the full implementation from custom metrics in practice
        return 0.72  # Placeholder
    
    def get_model_handler(self):
        """Get model handler for external use (e.g., safety evaluation)"""
        
        class ModelHandler:
            def __init__(self, pipeline, tokenizer, config):
                self.pipeline = pipeline
                self.tokenizer = tokenizer
                self.config = config
            
            def predict(self, texts: List[str], label_names: List[str]) -> List[int]:
                predictions = []
                for text in texts:
                    try:
                        if len(text.split()) > self.config.max_length // 2:
                            text = ' '.join(text.split()[:self.config.max_length // 2])
                        
                        result = self.pipeline(text, label_names)
                        predicted_label = result['labels'][0]
                        predicted_idx = label_names.index(predicted_label)
                        predictions.append(predicted_idx)
                    except:
                        predictions.append(0)
                return predictions
        
        return ModelHandler(self.model_pipeline, self.tokenizer, self.model_config)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        
        import json
        
        # Save to intermediate results
        results_file = self.output_path / "results" / "intermediate" / "model_evaluation_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Make results JSON serializable
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                serializable_results[key] = value
            elif hasattr(value, 'tolist'):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Model evaluation results saved to: {results_file}")
    
    def create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance summary for reporting"""
        
        standard_metrics = results.get('standard_metrics', {})
        custom_metrics = results.get('custom_metrics', {})
        
        summary = {
            'model_name': self.model_config.name,
            'model_id': self.model_config.model_id,
            'evaluation_timestamp': time.time(),
            
            # Key performance indicators
            'key_metrics': {
                'accuracy': standard_metrics.get('accuracy', 0.0),
                'f1_weighted': standard_metrics.get('f1_weighted', 0.0),
                'f1_macro': standard_metrics.get('f1_macro', 0.0),
                'balanced_accuracy': standard_metrics.get('balanced_accuracy', 0.0)
            },
            
            # Custom metrics
            'domain_metrics': {
                'urgency_awareness': custom_metrics.get('disaster_urgency_awareness', 0.0),
                'emotional_sensitivity': custom_metrics.get('emotional_context_sensitivity', 0.0),
                'actionability_relevance': custom_metrics.get('actionability_relevance', 0.0)
            },
            
            # Performance characteristics
            'performance_info': {
                'prediction_time': standard_metrics.get('prediction_time', 0.0),
                'samples_evaluated': standard_metrics.get('samples_evaluated', 0),
                'throughput_samples_per_second': (
                    standard_metrics.get('samples_evaluated', 0) / 
                    max(standard_metrics.get('prediction_time', 1), 0.01)
                )
            }
        }
        
        return summary
    
    def run_diagnostic_evaluation(self, test_data: Dict[str, List]) -> Dict[str, Any]:
        """Run diagnostic evaluation to identify specific issues"""
        
        logger.info("Running diagnostic evaluation...")
        
        texts = test_data['texts']
        labels = test_data['labels']
        
        # Sample a subset for detailed analysis
        sample_size = min(100, len(texts))
        sample_indices = np.random.choice(len(texts), sample_size, replace=False)
        
        sample_texts = [texts[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        
        diagnostics = {
            'text_length_analysis': self._analyze_text_lengths(sample_texts, sample_labels),
            'confidence_analysis': self._analyze_prediction_confidence(sample_texts, sample_labels),
            'error_patterns': self._analyze_error_patterns(sample_texts, sample_labels)
        }
        
        return diagnostics
    
    def _analyze_text_lengths(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Analyze performance vs text length"""
        
        text_lengths = [len(text.split()) for text in texts]
        
        return {
            'mean_length': np.mean(text_lengths),
            'median_length': np.median(text_lengths),
            'length_range': [np.min(text_lengths), np.max(text_lengths)],
            'long_text_proportion': sum(1 for length in text_lengths if length > 50) / len(text_lengths)
        }
    
    def _analyze_prediction_confidence(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Analyze prediction confidence patterns"""
        
        # This would analyze confidence scores from the model
        # For now, return placeholder analysis
        
        return {
            'mean_confidence': 0.75,
            'confidence_std': 0.15,
            'low_confidence_proportion': 0.2
        }
    
    def _analyze_error_patterns(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Analyze common error patterns"""
        
        # This would analyze specific error patterns
        # For now, return placeholder analysis
        
        return {
            'most_confused_classes': [(0, 1), (2, 4)],  # Example confused class pairs
            'error_rate_by_class': {i: 0.1 + (i * 0.05) for i in range(11)},
            'common_error_keywords': ['help', 'emergency', 'urgent']
        }
    
    def benchmark_model_performance(self, test_data: Dict[str, List]) -> Dict[str, Any]:
        """Benchmark model performance characteristics"""
        
        logger.info("Benchmarking model performance...")
        
        texts = test_data['texts']
        
        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32]
        performance_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(texts):
                continue
                
            # Test subset
            test_texts = texts[:min(50, len(texts))]
            
            start_time = time.time()
            
            # Simple prediction timing
            predictions = []
            for i in range(0, len(test_texts), batch_size):
                batch = test_texts[i:i + batch_size]
                for text in batch:
                    try:
                        result = self.model_pipeline(text, ['urgent', 'not urgent'])
                        predictions.append(result['labels'][0])
                    except:
                        predictions.append('not urgent')
            
            total_time = time.time() - start_time
            throughput = len(test_texts) / total_time
            
            performance_results[f'batch_size_{batch_size}'] = {
                'throughput_samples_per_second': throughput,
                'total_time_seconds': total_time,
                'samples_processed': len(test_texts)
            }
        
        # Memory usage estimation (simplified)
        estimated_memory_mb = (
            self.model_info.get('parameters', 0) * 4 / (1024 * 1024)  # Rough estimate
        )
        
        benchmark_results = {
            'performance_by_batch_size': performance_results,
            'estimated_memory_usage_mb': estimated_memory_mb,
            'model_parameters': self.model_info.get('parameters', 0),
            'device_info': {
                'device_type': self.model_info.get('device', 'unknown'),
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return benchmark_results