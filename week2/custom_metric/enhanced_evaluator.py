"""
Enhanced Evaluation engine with integrated custom metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from pathlib import Path
import time

from config import EvaluationConfig
from data_loader import DatasetLoader
from model_handler import ModelManager
from custom_metrics import AdvancedMetricsEvaluator

logger = logging.getLogger(__name__)

class EnhancedModelEvaluator:
    """Enhanced evaluation engine with custom metrics support"""
    
    def __init__(self, config: EvaluationConfig, dataset_loader: DatasetLoader, 
                 model_manager: ModelManager):
        self.config = config
        self.dataset_loader = dataset_loader
        self.model_manager = model_manager
        self.results = {}
        self.advanced_evaluator = AdvancedMetricsEvaluator()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_models_with_custom_metrics(self, test_texts: List[str], test_labels: List[int], 
                                          model_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Evaluate all models with both standard and custom metrics"""
        
        if model_names is None:
            model_names = self.model_manager.get_loaded_models()
        
        if not model_names:
            logger.error("No models available for evaluation")
            return {}
        
        logger.info(f"Starting enhanced evaluation of {len(model_names)} models on {len(test_texts)} samples")
        logger.info(f"Models to evaluate: {model_names}")
        
        label_names = self.dataset_loader.label_names
        
        for model_name in model_names:
            logger.info(f"\n{'='*25} Evaluating {model_name.upper()} {'='*25}")
            
            handler = self.model_manager.get_handler(model_name)
            if not handler:
                logger.error(f"Handler for {model_name} not found")
                continue
            
            # Make predictions
            start_time = time.time()
            try:
                predictions = handler.predict(test_texts, label_names)
                prediction_time = time.time() - start_time
                
                # Calculate comprehensive metrics using advanced evaluator
                logger.info("Computing comprehensive metrics...")
                advanced_results = self.advanced_evaluator.evaluate_all_metrics(
                    test_labels, predictions, test_texts, label_names
                )
                
                # Combine with basic model info
                model_results = {
                    'model_name': model_name,
                    'prediction_time': prediction_time,
                    'predictions': predictions,
                    'true_labels': test_labels,
                    'texts': test_texts,  # Store for custom metric analysis
                    **advanced_results
                }
                
                self.results[model_name] = model_results
                
                # Display results
                self._display_enhanced_model_results(model_name, model_results)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Save results and generate visualizations
        if self.results:
            self._save_enhanced_results()
            self._generate_enhanced_visualizations()
        
        return self.results
    
    def _display_enhanced_model_results(self, model_name: str, results: Dict[str, Any]):
        """Display comprehensive results for a single model"""
        
        print(f"\n{'='*15} {model_name.upper()} RESULTS {'='*15}")
        
        # Basic info
        print(f"Prediction Time: {results['prediction_time']:.2f} seconds")
        
        # Standard metrics
        standard = results['standard_metrics']
        print(f"\n📊 STANDARD METRICS:")
        print(f"  Accuracy: {standard.get('accuracy', 0):.4f}")
        print(f"  F1 (Weighted): {standard.get('f1_weighted', 0):.4f}")
        print(f"  F1 (Macro): {standard.get('f1_macro', 0):.4f}")
        print(f"  Precision (Weighted): {standard.get('precision_weighted', 0):.4f}")
        print(f"  Recall (Weighted): {standard.get('recall_weighted', 0):.4f}")
        print(f"  Balanced Accuracy: {standard.get('balanced_accuracy', 0):.4f}")
        print(f"  Matthews Correlation: {standard.get('matthews_corrcoef', 0):.4f}")
        
        # Custom metrics
        custom = results['custom_metrics']
        interpretations = results['custom_interpretations']
        print(f"\n🎯 CUSTOM DISASTER-SPECIFIC METRICS:")
        
        print(f"  Disaster Urgency Awareness (DUA): {custom.get('Disaster Urgency Awareness (DUA)', 0):.4f}")
        print(f"    → {interpretations.get('Disaster Urgency Awareness (DUA)', 'N/A')}")
        
        print(f"  Emotional Context Sensitivity (ECS): {custom.get('Emotional Context Sensitivity (ECS)', 0):.4f}")
        print(f"    → {interpretations.get('Emotional Context Sensitivity (ECS)', 'N/A')}")
        
        print(f"  Actionability Relevance (AR): {custom.get('Actionability Relevance (AR)', 0):.4f}")
        print(f"    → {interpretations.get('Actionability Relevance (AR)', 'N/A')}")
        
        # Actionability breakdown
        if 'actionability_breakdown' in results:
            print(f"\n⚡ ACTIONABILITY LEVEL BREAKDOWN:")
            breakdown = results['actionability_breakdown']
            for level, data in breakdown.items():
                print(f"  {level.replace('_', ' ').title()}: {data['accuracy']:.4f} "
                      f"(weight: {data['weight']}, samples: {data['samples']})")
        
        # Top confusion pairs
        confusion_analysis = results.get('confusion_analysis', {})
        if 'top_confusions' in confusion_analysis:
            print(f"\n❌ TOP CONFUSION PAIRS:")
            for i, confusion in enumerate(confusion_analysis['top_confusions'][:3]):
                print(f"  {i+1}. {confusion['true_class']} → {confusion['predicted_class']}: "
                      f"{confusion['count']} errors ({confusion['percentage']:.1f}%)")
    
    def compare_models_comprehensive(self) -> Optional[pd.DataFrame]:
        """Create comprehensive comparison between models"""
        
        if len(self.results) < 2:
            logger.warning("Need at least 2 models for comprehensive comparison")
            return None
        
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE MODEL COMPARISON")
        logger.info(f"{'='*60}")
        
        # Standard metrics comparison
        print(f"\n📊 STANDARD METRICS COMPARISON:")
        standard_data = []
        for model_name, results in self.results.items():
            standard = results['standard_metrics']
            standard_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{standard.get('accuracy', 0):.4f}",
                'F1 (Weighted)': f"{standard.get('f1_weighted', 0):.4f}",
                'F1 (Macro)': f"{standard.get('f1_macro', 0):.4f}",
                'Precision': f"{standard.get('precision_weighted', 0):.4f}",
                'Recall': f"{standard.get('recall_weighted', 0):.4f}",
                'Balanced Acc': f"{standard.get('balanced_accuracy', 0):.4f}",
                'Time (s)': f"{results['prediction_time']:.2f}"
            })
        
        standard_df = pd.DataFrame(standard_data)
        print(standard_df.to_string(index=False))
        
        # Custom metrics comparison  
        print(f"\n🎯 CUSTOM METRICS COMPARISON:")
        custom_data = []
        for model_name, results in self.results.items():
            custom = results['custom_metrics']
            custom_data.append({
                'Model': model_name.upper(),
                'DUA (Urgency)': f"{custom.get('Disaster Urgency Awareness (DUA)', 0):.4f}",
                'ECS (Emotional)': f"{custom.get('Emotional Context Sensitivity (ECS)', 0):.4f}",
                'AR (Actionability)': f"{custom.get('Actionability Relevance (AR)', 0):.4f}"
            })
        
        custom_df = pd.DataFrame(custom_data)
        print(custom_df.to_string(index=False))
        
        # Save comparisons
        standard_df.to_csv(Path(self.config.output_dir) / "standard_metrics_comparison.csv", index=False)
        custom_df.to_csv(Path(self.config.output_dir) / "custom_metrics_comparison.csv", index=False)
        
        # Combined comparison for return
        combined_data = []
        for model_name, results in self.results.items():
            standard = results['standard_metrics']
            custom = results['custom_metrics']
            combined_data.append({
                'Model': model_name.upper(),
                'Accuracy': standard.get('accuracy', 0),
                'F1_Weighted': standard.get('f1_weighted', 0),
                'F1_Macro': standard.get('f1_macro', 0),
                'DUA': custom.get('Disaster Urgency Awareness (DUA)', 0),
                'ECS': custom.get('Emotional Context Sensitivity (ECS)', 0),
                'AR': custom.get('Actionability Relevance (AR)', 0),
                'Time': results['prediction_time']
            })
        
        return pd.DataFrame(combined_data)
    
    def _generate_enhanced_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("Generating enhanced visualizations...")
        
        # 1. Metrics comparison plot
        self.advanced_evaluator.create_metrics_comparison_plot(
            self.results, 
            save_path=Path(self.config.output_dir) / "comprehensive_metrics_comparison.png"
        )
        
        # 2. Actionability breakdown plot
        self.advanced_evaluator.create_actionability_breakdown_plot(
            self.results,
            save_path=Path(self.config.output_dir) / "actionability_breakdown.png"
        )
        
        # 3. Custom vs Standard metrics correlation
        self._create_metrics_correlation_plot()
        
        # 4. Confusion matrices with custom annotations
        self._create_enhanced_confusion_matrices()
    
    def _create_metrics_correlation_plot(self):
        """Create correlation plot between standard and custom metrics"""
        if len(self.results) < 2:
            return
        
        models = list(self.results.keys())
        
        # Extract metrics for correlation analysis
        metrics_data = {}
        for model in models:
            standard = self.results[model]['standard_metrics']
            custom = self.results[model]['custom_metrics']
            
            metrics_data[model] = {
                'F1_Weighted': standard.get('f1_weighted', 0),
                'Accuracy': standard.get('accuracy', 0),
                'Balanced_Acc': standard.get('balanced_accuracy', 0),
                'DUA': custom.get('Disaster Urgency Awareness (DUA)', 0),
                'ECS': custom.get('Emotional Context Sensitivity (ECS)', 0),
                'AR': custom.get('Actionability Relevance (AR)', 0)
            }
        
        # Create correlation heatmap
        df = pd.DataFrame(metrics_data).T
        correlation_matrix = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.3f')
        plt.title('Correlation Between Standard and Custom Metrics\n(Across Models)')
        plt.tight_layout()
        
        save_path = Path(self.config.output_dir) / "metrics_correlation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_enhanced_confusion_matrices(self):
        """Create confusion matrices with actionability level annotations"""
        
        n_models = len(self.results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(10*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        label_names = self.dataset_loader.label_names
        
        # Get actionability mapping from custom metrics
        ar_metric = self.advanced_evaluator.custom_metrics[2]  # ActionabilityRelevanceMetric
        
        # Create color-coded labels based on actionability
        actionability_colors = {
            'high_action': 'red',
            'medium_action': 'orange', 
            'low_action': 'yellow',
            'informational': 'lightblue',
            'no_action': 'lightgray'
        }
        
        colored_labels = []
        for label in label_names:
            action_level = ar_metric._get_actionability_level(label)
            colored_labels.append(f"{label.replace('_', ' ')[:15]}...")
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = np.array(results['confusion_analysis']['confusion_matrix'])
            
            # Normalize confusion matrix for better visualization
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            im = axes[idx].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                                 ha="center", va="center",
                                 color="white" if cm_normalized[i, j] > thresh else "black",
                                 fontsize=8)
            
            axes[idx].set_title(f'{model_name.upper()}\nConfusion Matrix (Count & Normalized)')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            
            # Set ticks and labels
            tick_marks = np.arange(len(label_names))
            axes[idx].set_xticks(tick_marks)
            axes[idx].set_yticks(tick_marks)
            axes[idx].set_xticklabels(colored_labels, rotation=45, ha='right')
            axes[idx].set_yticklabels(colored_labels)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        save_path = Path(self.config.output_dir) / "enhanced_confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_enhanced_results(self):
        """Save comprehensive results to files"""
        
        # Save detailed results as JSON
        results_path = Path(self.config.output_dir) / "comprehensive_evaluation_results.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {
                'model_name': results['model_name'],
                'prediction_time': results['prediction_time'],
                'standard_metrics': results['standard_metrics'],
                'custom_metrics': results['custom_metrics'],
                'custom_interpretations': results['custom_interpretations'],
                'actionability_breakdown': results['actionability_breakdown'],
                'per_class_metrics': results['per_class_metrics'],
                'confusion_analysis': {
                    'confusion_matrix': results['confusion_analysis']['confusion_matrix'],
                    'top_confusions': results['confusion_analysis']['top_confusions'],
                    'class_accuracies': results['confusion_analysis']['class_accuracies']
                }
                # Skip predictions, texts, and true_labels for JSON
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Comprehensive results saved to: {results_path}")
        
        # Save detailed predictions with custom analysis
        if self.config.save_predictions:
            for model_name, results in self.results.items():
                self._save_detailed_predictions(model_name, results)
    
    def _save_detailed_predictions(self, model_name: str, results: Dict[str, Any]):
        """Save predictions with custom metric analysis"""
        
        pred_path = Path(self.config.output_dir) / f"{model_name}_detailed_predictions.csv"
        
        # Get actionability and urgency info for each prediction
        ar_metric = self.advanced_evaluator.custom_metrics[2]  # ActionabilityRelevanceMetric  
        dua_metric = self.advanced_evaluator.custom_metrics[0]  # DisasterUrgencyAwarenessMetric
        
        detailed_data = []
        for i, (true_label, pred_label, text) in enumerate(zip(
            results['true_labels'], results['predictions'], results['texts']
        )):
            true_class = self.dataset_loader.label_names[true_label]
            pred_class = self.dataset_loader.label_names[pred_label]
            correct = true_label == pred_label
            
            # Get actionability and urgency levels
            true_actionability = ar_metric._get_actionability_level(true_class)
            pred_actionability = ar_metric._get_actionability_level(pred_class)
            
            true_urgency = dua_metric.urgency_weights.get(true_class, 1.0)
            pred_urgency = dua_metric.urgency_weights.get(pred_class, 1.0)
            
            detailed_data.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'true_class': true_class,
                'predicted_class': pred_class,
                'correct': correct,
                'true_actionability': true_actionability,
                'predicted_actionability': pred_actionability,
                'true_urgency_weight': true_urgency,
                'predicted_urgency_weight': pred_urgency,
                'actionability_correct': true_actionability == pred_actionability,
                'urgency_penalty': abs(true_urgency - pred_urgency) if not correct else 0.0
            })
        
        pred_df = pd.DataFrame(detailed_data)
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"{model_name} detailed predictions saved to: {pred_path}")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report with custom metrics"""
        
        if not self.results:
            logger.warning("No results available for report generation")
            return ""
        
        # Generate advanced metrics report
        report_content = self.advanced_evaluator.generate_advanced_metrics_report(
            self.results,
            output_path=Path(self.config.output_dir) / "comprehensive_evaluation_report.md"
        )
        
        return report_content
    
    def get_model_rankings(self) -> Dict[str, str]:
        """Get model rankings across different criteria"""
        
        if len(self.results) < 2:
            return {}
        
        models = list(self.results.keys())
        
        rankings = {
            'overall_standard': max(models, key=lambda x: self.results[x]['standard_metrics'].get('f1_weighted', 0)),
            'accuracy': max(models, key=lambda x: self.results[x]['standard_metrics'].get('accuracy', 0)),
            'urgency_awareness': max(models, key=lambda x: self.results[x]['custom_metrics'].get('Disaster Urgency Awareness (DUA)', 0)),
            'emotional_sensitivity': max(models, key=lambda x: self.results[x]['custom_metrics'].get('Emotional Context Sensitivity (ECS)', 0)),
            'actionability_detection': max(models, key=lambda x: self.results[x]['custom_metrics'].get('Actionability Relevance (AR)', 0)),
            'speed': min(models, key=lambda x: self.results[x]['prediction_time']),
            'balanced_performance': max(models, key=lambda x: (
                self.results[x]['standard_metrics'].get('f1_weighted', 0) + 
                self.results[x]['custom_metrics'].get('Disaster Urgency Awareness (DUA)', 0) +
                self.results[x]['custom_metrics'].get('Emotional Context Sensitivity (ECS)', 0) +
                self.results[x]['custom_metrics'].get('Actionability Relevance (AR)', 0)
            ) / 4)
        }
        
        return rankings
