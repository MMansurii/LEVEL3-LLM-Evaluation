"""
Custom Metrics Module for Disaster Response Tweet Classification
Implements both standard and domain-specific custom metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from scipy.spatial.distance import cosine
from collections import Counter, defaultdict
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseCustomMetric(ABC):
    """Abstract base class for custom metrics"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def compute(self, y_true: List[int], y_pred: List[int], 
                texts: List[str], label_names: List[str]) -> float:
        """Compute the custom metric"""
        pass
    
    @abstractmethod
    def get_interpretation(self, score: float) -> str:
        """Provide interpretation of the score"""
        pass

class DisasterUrgencyAwarenessMetric(BaseCustomMetric):
    """
    Custom Metric 1: Disaster Urgency Awareness (DUA)
    
    Theory:
    In disaster response, correctly identifying urgent situations (injured/dead people, 
    urgent needs, rescue operations) is more critical than correctly classifying 
    less urgent content (sympathy, general information). This metric weights 
    classification performance based on the urgency level of different disaster 
    response categories.
    
    Why it's good:
    - Reflects real-world priorities in disaster response
    - Penalizes more heavily when urgent situations are misclassified
    - Provides actionable insights for emergency response systems
    
    Implementation:
    DUA = Σ(urgency_weight_i × correct_predictions_i) / Σ(urgency_weight_i × total_predictions_i)
    """
    
    def __init__(self):
        super().__init__(
            "Disaster Urgency Awareness (DUA)",
            "Measures how well the model identifies urgent vs non-urgent disaster situations"
        )
        
        # Define urgency weights based on disaster response priorities
        self.urgency_weights = {
            'injured_or_dead_people': 5.0,           # Highest priority
            'requests_or_urgent_needs': 4.5,         # Very high priority  
            'rescue_volunteering_or_donation_effort': 4.0,  # High priority
            'missing_or_found_people': 4.0,          # High priority
            'displaced_people_and_evacuations': 3.5, # High-medium priority
            'infrastructure_and_utility_damage': 3.0, # Medium priority
            'caution_and_advice': 2.5,              # Medium-low priority
            'other_relevant_information': 2.0,       # Low priority
            'sympathy_and_support': 1.5,            # Lower priority
            'not_humanitarian': 1.0,                # Lowest priority
            'dont_know_cant_judge': 1.0             # Lowest priority
        }
    
    def compute(self, y_true: List[int], y_pred: List[int], 
                texts: List[str], label_names: List[str]) -> float:
        """Compute Disaster Urgency Awareness score"""
        
        total_weighted_correct = 0.0
        total_weighted_samples = 0.0
        
        for true_label, pred_label in zip(y_true, y_pred):
            true_class = label_names[true_label]
            weight = self.urgency_weights.get(true_class, 1.0)
            
            total_weighted_samples += weight
            if true_label == pred_label:
                total_weighted_correct += weight
        
        if total_weighted_samples == 0:
            return 0.0
        
        return total_weighted_correct / total_weighted_samples
    
    def get_interpretation(self, score: float) -> str:
        """Interpret DUA score"""
        if score >= 0.85:
            return "Excellent urgency awareness - highly suitable for emergency systems"
        elif score >= 0.75:
            return "Good urgency awareness - suitable for disaster response applications"
        elif score >= 0.65:
            return "Moderate urgency awareness - needs improvement for critical applications"
        elif score >= 0.55:
            return "Poor urgency awareness - significant risk in emergency scenarios"
        else:
            return "Very poor urgency awareness - unsuitable for disaster response"

class EmotionalContextSensitivityMetric(BaseCustomMetric):
    """
    Custom Metric 2: Emotional Context Sensitivity (ECS)
    
    Theory:
    Disaster response tweets contain different emotional contexts (distress, hope, 
    gratitude, urgency). A good model should be sensitive to these emotional 
    nuances and avoid confusing emotionally similar but functionally different 
    categories (e.g., sympathy vs requests for help).
    
    Why it's good:
    - Captures the model's understanding of human emotional states during disasters
    - Important for mental health support and appropriate response routing
    - Helps distinguish between passive support and active help requests
    
    Implementation:
    ECS measures confusion between emotionally similar categories and rewards
    models that maintain clear distinctions between them.
    """
    
    def __init__(self):
        super().__init__(
            "Emotional Context Sensitivity (ECS)",
            "Measures model's ability to distinguish between emotionally similar disaster categories"
        )
        
        # Define emotional similarity groups (higher penalty for confusion within groups)
        self.emotional_groups = {
            'active_help': {
                'rescue_volunteering_or_donation_effort',
                'requests_or_urgent_needs'
            },
            'distress_reporting': {
                'injured_or_dead_people',
                'missing_or_found_people',
                'displaced_people_and_evacuations'
            },
            'passive_support': {
                'sympathy_and_support',
                'other_relevant_information'
            },
            'informational': {
                'infrastructure_and_utility_damage',
                'caution_and_advice',
                'other_relevant_information'
            }
        }
        
        # Create confusion penalty matrix
        self.confusion_penalties = self._create_confusion_matrix(len(self.emotional_groups))
    
    def _create_confusion_matrix(self, n_groups: int) -> Dict[Tuple[str, str], float]:
        """Create confusion penalty matrix based on emotional similarity"""
        penalties = {}
        
        # Default penalty for any confusion
        base_penalty = 1.0
        
        # Higher penalty for confusion within the same emotional group
        within_group_penalty = 2.0
        
        # Lower penalty for confusion between different emotional groups
        between_group_penalty = 0.5
        
        return {
            'within_group': within_group_penalty,
            'between_group': between_group_penalty,
            'base': base_penalty
        }
    
    def _get_emotional_group(self, class_name: str) -> Optional[str]:
        """Get the emotional group for a class"""
        for group, classes in self.emotional_groups.items():
            if class_name in classes:
                return group
        return None
    
    def compute(self, y_true: List[int], y_pred: List[int], 
                texts: List[str], label_names: List[str]) -> float:
        """Compute Emotional Context Sensitivity score"""
        
        total_penalty = 0.0
        total_samples = len(y_true)
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                true_class = label_names[true_label]
                pred_class = label_names[pred_label]
                
                true_group = self._get_emotional_group(true_class)
                pred_group = self._get_emotional_group(pred_class)
                
                if true_group and pred_group:
                    if true_group == pred_group:
                        # Confusion within same emotional group - higher penalty
                        penalty = self.confusion_penalties['within_group']
                    else:
                        # Confusion between different groups - lower penalty
                        penalty = self.confusion_penalties['between_group']
                else:
                    # Default penalty
                    penalty = self.confusion_penalties['base']
                
                total_penalty += penalty
        
        # Convert to a score between 0 and 1 (higher is better)
        if total_samples == 0:
            return 0.0
        
        max_possible_penalty = total_samples * self.confusion_penalties['within_group']
        sensitivity_score = 1.0 - (total_penalty / max_possible_penalty)
        
        return max(0.0, sensitivity_score)
    
    def get_interpretation(self, score: float) -> str:
        """Interpret ECS score"""
        if score >= 0.80:
            return "Excellent emotional sensitivity - distinguishes well between similar contexts"
        elif score >= 0.70:
            return "Good emotional sensitivity - mostly appropriate contextual understanding"
        elif score >= 0.60:
            return "Moderate emotional sensitivity - some confusion in similar contexts"
        elif score >= 0.50:
            return "Poor emotional sensitivity - frequent confusion between emotional contexts"
        else:
            return "Very poor emotional sensitivity - lacks contextual understanding"

class ActionabilityRelevanceMetric(BaseCustomMetric):
    """
    Custom Metric 3: Actionability Relevance (AR)
    
    Theory:
    In disaster response, information can be classified as actionable (requires 
    immediate response) or informational (provides context). This metric evaluates 
    how well the model distinguishes between content that requires action versus 
    content that is purely informational. Critical for routing messages to 
    appropriate response teams.
    
    Why it's good:
    - Directly maps to operational needs in disaster management
    - Helps optimize resource allocation during emergencies
    - Reduces information overload for emergency responders
    
    Implementation:
    AR considers the actionability level of each category and measures precision
    and recall for actionable vs non-actionable categories separately.
    """
    
    def __init__(self):
        super().__init__(
            "Actionability Relevance (AR)",
            "Measures model's ability to distinguish actionable from informational content"
        )
        
        # Define actionability levels
        self.actionability_mapping = {
            'injured_or_dead_people': 'high_action',
            'requests_or_urgent_needs': 'high_action',
            'missing_or_found_people': 'high_action',
            'rescue_volunteering_or_donation_effort': 'medium_action',
            'displaced_people_and_evacuations': 'medium_action',
            'infrastructure_and_utility_damage': 'medium_action',
            'caution_and_advice': 'low_action',
            'other_relevant_information': 'informational',
            'sympathy_and_support': 'informational',
            'not_humanitarian': 'no_action',
            'dont_know_cant_judge': 'no_action'
        }
        
        # Define action level weights
        self.action_weights = {
            'high_action': 4.0,
            'medium_action': 2.5,
            'low_action': 1.5,
            'informational': 1.0,
            'no_action': 0.5
        }
    
    def _get_actionability_level(self, class_name: str) -> str:
        """Get actionability level for a class"""
        return self.actionability_mapping.get(class_name, 'informational')
    
    def compute(self, y_true: List[int], y_pred: List[int], 
                texts: List[str], label_names: List[str]) -> float:
        """Compute Actionability Relevance score"""
        
        # Group predictions by actionability level
        actionability_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for true_label, pred_label in zip(y_true, y_pred):
            true_class = label_names[true_label]
            action_level = self._get_actionability_level(true_class)
            
            actionability_performance[action_level]['total'] += 1
            if true_label == pred_label:
                actionability_performance[action_level]['correct'] += 1
        
        # Calculate weighted actionability score
        total_weighted_correct = 0.0
        total_weighted_samples = 0.0
        
        for action_level, performance in actionability_performance.items():
            weight = self.action_weights[action_level]
            correct = performance['correct']
            total = performance['total']
            
            total_weighted_samples += weight * total
            total_weighted_correct += weight * correct
        
        if total_weighted_samples == 0:
            return 0.0
        
        return total_weighted_correct / total_weighted_samples
    
    def get_detailed_breakdown(self, y_true: List[int], y_pred: List[int], 
                             label_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown by actionability level"""
        actionability_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for true_label, pred_label in zip(y_true, y_pred):
            true_class = label_names[true_label]
            action_level = self._get_actionability_level(true_class)
            
            actionability_performance[action_level]['total'] += 1
            if true_label == pred_label:
                actionability_performance[action_level]['correct'] += 1
        
        # Calculate accuracy for each level
        breakdown = {}
        for action_level, performance in actionability_performance.items():
            if performance['total'] > 0:
                accuracy = performance['correct'] / performance['total']
                breakdown[action_level] = {
                    'accuracy': accuracy,
                    'samples': performance['total'],
                    'weight': self.action_weights[action_level]
                }
        
        return breakdown
    
    def get_interpretation(self, score: float) -> str:
        """Interpret AR score"""
        if score >= 0.80:
            return "Excellent actionability detection - reliable for emergency routing"
        elif score >= 0.70:
            return "Good actionability detection - suitable for disaster response systems"
        elif score >= 0.60:
            return "Moderate actionability detection - may miss some urgent situations"
        elif score >= 0.50:
            return "Poor actionability detection - risk of delayed emergency response"
        else:
            return "Very poor actionability detection - unsuitable for operational use"

class StandardMetricsCalculator:
    """Calculator for all standard classification metrics"""
    
    def __init__(self):
        self.standard_metrics = {
            'accuracy': accuracy_score,
            'precision_weighted': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score,
            'matthews_corrcoef': matthews_corrcoef,
        }
    
    def compute_all_standard_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Compute all standard metrics"""
        results = {}
        
        for metric_name, metric_func in self.standard_metrics.items():
            try:
                score = metric_func(y_true, y_pred)
                results[metric_name] = score
            except Exception as e:
                logger.warning(f"Could not compute {metric_name}: {e}")
                results[metric_name] = 0.0
        
        return results
    
    def get_per_class_metrics(self, y_true: List[int], y_pred: List[int], 
                            label_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Get detailed per-class metrics"""
        
        # Per-class precision, recall, f1
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_results = {}
        for i, label in enumerate(label_names):
            if i < len(precision_per_class):
                per_class_results[label] = {
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1': f1_per_class[i]
                }
        
        return per_class_results
    
    def get_confusion_matrix_analysis(self, y_true: List[int], y_pred: List[int], 
                                    label_names: List[str]) -> Dict[str, Any]:
        """Get detailed confusion matrix analysis"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Most confused classes
        confusion_pairs = []
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                if i != j and cm[i][j] > 0:
                    confusion_pairs.append({
                        'true_class': label_names[i],
                        'predicted_class': label_names[j],
                        'count': int(cm[i][j]),
                        'percentage': cm[i][j] / np.sum(cm[i]) * 100
                    })
        
        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'top_confusions': confusion_pairs[:10],  # Top 10 most confused pairs
            'class_accuracies': [cm[i][i] / np.sum(cm[i]) if np.sum(cm[i]) > 0 else 0 
                               for i in range(len(label_names))]
        }

class AdvancedMetricsEvaluator:
    """Main class for computing both standard and custom metrics"""
    
    def __init__(self):
        self.standard_calculator = StandardMetricsCalculator()
        self.custom_metrics = [
            DisasterUrgencyAwarenessMetric(),
            EmotionalContextSensitivityMetric(),
            ActionabilityRelevanceMetric()
        ]
    
    def evaluate_all_metrics(self, y_true: List[int], y_pred: List[int], 
                           texts: List[str], label_names: List[str]) -> Dict[str, Any]:
        """Compute all metrics (standard + custom)"""
        
        logger.info("Computing standard metrics...")
        standard_metrics = self.standard_calculator.compute_all_standard_metrics(y_true, y_pred)
        
        logger.info("Computing per-class metrics...")
        per_class_metrics = self.standard_calculator.get_per_class_metrics(y_true, y_pred, label_names)
        
        logger.info("Computing confusion matrix analysis...")
        confusion_analysis = self.standard_calculator.get_confusion_matrix_analysis(y_true, y_pred, label_names)
        
        logger.info("Computing custom metrics...")
        custom_metrics = {}
        custom_interpretations = {}
        
        for metric in self.custom_metrics:
            try:
                score = metric.compute(y_true, y_pred, texts, label_names)
                custom_metrics[metric.name] = score
                custom_interpretations[metric.name] = metric.get_interpretation(score)
                logger.info(f"✓ {metric.name}: {score:.4f}")
            except Exception as e:
                logger.error(f"Error computing {metric.name}: {e}")
                custom_metrics[metric.name] = 0.0
                custom_interpretations[metric.name] = "Error in computation"
        
        # Get detailed breakdown for Actionability Relevance
        ar_metric = self.custom_metrics[2]  # ActionabilityRelevanceMetric is third
        actionability_breakdown = ar_metric.get_detailed_breakdown(y_true, y_pred, label_names)
        
        return {
            'standard_metrics': standard_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_analysis': confusion_analysis,
            'custom_metrics': custom_metrics,
            'custom_interpretations': custom_interpretations,
            'actionability_breakdown': actionability_breakdown
        }
    
    def create_metrics_comparison_plot(self, model_results: Dict[str, Dict], 
                                     save_path: Optional[str] = None):
        """Create visualization comparing models across all metrics"""
        
        models = list(model_results.keys())
        
        # Extract key metrics for comparison
        metrics_to_plot = [
            'accuracy', 'f1_weighted', 'f1_macro', 
            'Disaster Urgency Awareness (DUA)',
            'Emotional Context Sensitivity (ECS)', 
            'Actionability Relevance (AR)'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            values = []
            for model in models:
                if metric in model_results[model]['standard_metrics']:
                    values.append(model_results[model]['standard_metrics'][metric])
                elif metric in model_results[model]['custom_metrics']:
                    values.append(model_results[model]['custom_metrics'][metric])
                else:
                    values.append(0.0)
            
            bars = ax.bar(models, values, alpha=0.7, 
                         color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_actionability_breakdown_plot(self, model_results: Dict[str, Dict],
                                          save_path: Optional[str] = None):
        """Create plot showing actionability level performance"""
        
        models = list(model_results.keys())
        action_levels = ['high_action', 'medium_action', 'low_action', 'informational', 'no_action']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(action_levels))
        width = 0.35 if len(models) == 2 else 0.25
        
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i, model in enumerate(models):
            accuracies = []
            for level in action_levels:
                breakdown = model_results[model].get('actionability_breakdown', {})
                if level in breakdown:
                    accuracies.append(breakdown[level]['accuracy'])
                else:
                    accuracies.append(0.0)
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, accuracies, width, 
                         label=model.upper(), color=colors[i], alpha=0.7)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                if acc > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Actionability Level')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance by Actionability Level')
        ax.set_xticks(x)
        ax.set_xticklabels([level.replace('_', ' ').title() for level in action_levels])
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_advanced_metrics_report(self, model_results: Dict[str, Dict],
                                       output_path: str = "./advanced_metrics_report.md") -> str:
        """Generate comprehensive metrics report"""
        
        report_lines = [
            "# Advanced Metrics Analysis Report",
            "",
            "## Overview",
            f"This report provides detailed analysis using both standard classification metrics and custom domain-specific metrics designed for disaster response applications.",
            "",
            "## Standard Metrics Summary",
            "",
            "### Key Standard Metrics",
            "- **Accuracy**: Overall classification accuracy",
            "- **Precision (Weighted)**: Precision weighted by class frequency", 
            "- **Recall (Weighted)**: Recall weighted by class frequency",
            "- **F1 (Weighted)**: Harmonic mean of precision and recall, weighted",
            "- **F1 (Macro)**: Unweighted average F1 across all classes",
            "- **Balanced Accuracy**: Accuracy adjusted for class imbalance",
            "- **Matthews Correlation Coefficient**: Correlation between predictions and truth",
            ""
        ]
        
        # Standard metrics comparison
        for model_name, results in model_results.items():
            standard = results['standard_metrics']
            report_lines.extend([
                f"### {model_name.upper()} - Standard Metrics",
                f"- Accuracy: {standard.get('accuracy', 0):.4f}",
                f"- Precision (Weighted): {standard.get('precision_weighted', 0):.4f}",
                f"- Recall (Weighted): {standard.get('recall_weighted', 0):.4f}",
                f"- F1 (Weighted): {standard.get('f1_weighted', 0):.4f}",
                f"- F1 (Macro): {standard.get('f1_macro', 0):.4f}",
                f"- Balanced Accuracy: {standard.get('balanced_accuracy', 0):.4f}",
                f"- Matthews Correlation: {standard.get('matthews_corrcoef', 0):.4f}",
                ""
            ])
        
        # Custom metrics section
        report_lines.extend([
            "## Custom Domain-Specific Metrics",
            "",
            "### 1. Disaster Urgency Awareness (DUA)",
            "**Purpose**: Measures how well the model identifies urgent vs non-urgent disaster situations",
            "**Why Important**: In emergency response, correctly identifying urgent situations (injuries, urgent needs) is more critical than general information",
            "**Calculation**: Weighted accuracy where urgent categories receive higher weights",
            "",
            "### 2. Emotional Context Sensitivity (ECS)", 
            "**Purpose**: Evaluates the model's ability to distinguish between emotionally similar but functionally different categories",
            "**Why Important**: Prevents confusion between passive support (sympathy) and active requests (help needed)",
            "**Calculation**: Penalizes confusions within similar emotional contexts more heavily",
            "",
            "### 3. Actionability Relevance (AR)",
            "**Purpose**: Measures distinction between actionable content (requires response) vs informational content",
            "**Why Important**: Critical for routing messages to appropriate response teams and resource allocation",
            "**Calculation**: Weighted performance across actionability levels (high/medium/low action vs informational)",
            ""
        ])
        
        # Custom metrics results
        for model_name, results in model_results.items():
            custom = results['custom_metrics']
            interpretations = results['custom_interpretations']
            
            report_lines.extend([
                f"### {model_name.upper()} - Custom Metrics Results",
                f"**Disaster Urgency Awareness**: {custom.get('Disaster Urgency Awareness (DUA)', 0):.4f}",
                f"- {interpretations.get('Disaster Urgency Awareness (DUA)', 'N/A')}",
                "",
                f"**Emotional Context Sensitivity**: {custom.get('Emotional Context Sensitivity (ECS)', 0):.4f}",
                f"- {interpretations.get('Emotional Context Sensitivity (ECS)', 'N/A')}",
                "",
                f"**Actionability Relevance**: {custom.get('Actionability Relevance (AR)', 0):.4f}",
                f"- {interpretations.get('Actionability Relevance (AR)', 'N/A')}",
                ""
            ])
        
        # Recommendations
        if len(model_results) > 1:
            # Find best performing model for each metric type
            models = list(model_results.keys())
            
            best_standard = max(models, key=lambda x: model_results[x]['standard_metrics'].get('f1_weighted', 0))
            best_urgency = max(models, key=lambda x: model_results[x]['custom_metrics'].get('Disaster Urgency Awareness (DUA)', 0))
            best_emotional = max(models, key=lambda x: model_results[x]['custom_metrics'].get('Emotional Context Sensitivity (ECS)', 0))
            best_actionability = max(models, key=lambda x: model_results[x]['custom_metrics'].get('Actionability Relevance (AR)', 0))
            
            report_lines.extend([
                "## Performance Analysis & Recommendations",
                "",
                f"### Best Performing Models by Metric Category",
                f"- **Overall Standard Performance**: {best_standard.upper()} (F1-Weighted: {model_results[best_standard]['standard_metrics'].get('f1_weighted', 0):.4f})",
                f"- **Urgency Detection**: {best_urgency.upper()} (DUA: {model_results[best_urgency]['custom_metrics'].get('Disaster Urgency Awareness (DUA)', 0):.4f})",
                f"- **Emotional Sensitivity**: {best_emotional.upper()} (ECS: {model_results[best_emotional]['custom_metrics'].get('Emotional Context Sensitivity (ECS)', 0):.4f})",
                f"- **Actionability Detection**: {best_actionability.upper()} (AR: {model_results[best_actionability]['custom_metrics'].get('Actionability Relevance (AR)', 0):.4f})",
                "",
                "### Operational Recommendations",
                "1. **For Emergency Response Systems**: Prioritize models with high DUA and AR scores",
                "2. **For Social Media Monitoring**: Focus on models with high ECS scores", 
                "3. **For General Classification**: Consider balanced performance across all metrics",
                "",
                "### Key Insights",
                "- Custom metrics reveal domain-specific model capabilities not captured by standard metrics",
                "- Models may excel in different aspects of disaster response classification",
                "- Consider ensemble approaches combining model strengths",
                ""
            ])
        
        report_content = '\n'.join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Advanced metrics report saved to: {output_path}")
        return report_content
