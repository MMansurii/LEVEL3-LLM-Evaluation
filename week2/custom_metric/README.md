# Custom Metrics for Disaster Response Classification

## 📚 Overview

This document explains the implementation of custom evaluation metrics specifically designed for disaster response tweet classification, going beyond standard metrics to capture domain-specific performance aspects.

## 🎯 Why Custom Metrics?

### Limitations of Standard Metrics
Standard classification metrics (accuracy, F1, precision, recall) treat all classification errors equally. However, in disaster response:

- **Not all errors are equal**: Misclassifying an urgent rescue request is more critical than misclassifying a sympathy message
- **Context matters**: Emotional nuances and actionability levels are crucial for proper response
- **Operational needs**: Emergency responders need different types of information routed appropriately

### Our Solution: Domain-Specific Custom Metrics
We developed three custom metrics that capture aspects critical to disaster response applications:

1. **Disaster Urgency Awareness (DUA)** - Prioritizes urgent situations
2. **Emotional Context Sensitivity (ECS)** - Handles emotional nuances  
3. **Actionability Relevance (AR)** - Distinguishes actionable from informational content

---

## 📊 Standard Metrics Reference

Before diving into custom metrics, here are the standard metrics we also compute:

| Metric | Formula | Purpose | Best Used When |
|--------|---------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | Balanced datasets |
| **Precision (Weighted)** | Σ(precision_i × support_i) / Total | Weighted precision | Imbalanced classes |
| **Precision (Macro)** | Σ(precision_i) / n_classes | Unweighted precision | Equal class importance |
| **Recall (Weighted)** | Σ(recall_i × support_i) / Total | Weighted recall | Imbalanced classes |
| **F1 (Weighted)** | Weighted harmonic mean | Balanced precision/recall | General performance |
| **F1 (Macro)** | Unweighted F1 average | Class-agnostic performance | Equal class treatment |
| **Balanced Accuracy** | Average per-class accuracy | Handle imbalance | Very imbalanced data |
| **Matthews Correlation** | Correlation coefficient | Overall quality | Binary/multiclass |

---

## 🎯 Custom Metric 1: Disaster Urgency Awareness (DUA)

### Theory
In disaster response, correctly identifying urgent situations (injuries, urgent needs, rescue operations) is more critical than correctly classifying less urgent content (sympathy, general information). Standard metrics treat all errors equally, but a model that misclassifies urgent situations poses greater risk.

### Mathematical Definition
```
DUA = Σ(urgency_weight_i × correct_predictions_i) / Σ(urgency_weight_i × total_predictions_i)
```

Where `urgency_weight_i` is the predefined weight for class `i`.

### Urgency Weight Assignment

| Category | Weight | Rationale |
|----------|--------|-----------|
| **injured_or_dead_people** | 5.0 | Life-threatening situations - highest priority |
| **requests_or_urgent_needs** | 4.5 | Direct calls for help - immediate action needed |
| **rescue_volunteering_or_donation_effort** | 4.0 | Active assistance - high priority for coordination |
| **missing_or_found_people** | 4.0 | Search and rescue priority - time-critical |
| **displaced_people_and_evacuations** | 3.5 | Shelter and safety needs - high-medium priority |
| **infrastructure_and_utility_damage** | 3.0 | Infrastructure assessment - medium priority |
| **caution_and_advice** | 2.5 | Prevention information - medium-low priority |
| **other_relevant_information** | 2.0 | General information - low priority |
| **sympathy_and_support** | 1.5 | Emotional support - lower priority |
| **not_humanitarian** | 1.0 | Non-disaster content - lowest priority |
| **dont_know_cant_judge** | 1.0 | Unclear content - lowest priority |

### Implementation Details
```python
def compute_dua(y_true, y_pred, label_names, urgency_weights):
    total_weighted_correct = 0.0
    total_weighted_samples = 0.0
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_class = label_names[true_label]
        weight = urgency_weights.get(true_class, 1.0)
        
        total_weighted_samples += weight
        if true_label == pred_label:
            total_weighted_correct += weight
    
    return total_weighted_correct / total_weighted_samples
```

### Interpretation Scale
- **0.85-1.00**: Excellent urgency awareness - highly suitable for emergency systems
- **0.75-0.84**: Good urgency awareness - suitable for disaster response applications  
- **0.65-0.74**: Moderate urgency awareness - needs improvement for critical applications
- **0.55-0.64**: Poor urgency awareness - significant risk in emergency scenarios
- **0.00-0.54**: Very poor urgency awareness - unsuitable for disaster response

### Why DUA is Superior to Standard Metrics
1. **Real-world alignment**: Reflects actual disaster response priorities
2. **Risk-aware**: Penalizes more dangerous errors more heavily  
3. **Actionable**: Directly guides model improvement for emergency applications
4. **Resource optimization**: Helps allocate limited emergency resources effectively

---

## 💭 Custom Metric 2: Emotional Context Sensitivity (ECS)

### Theory
Disaster response tweets contain different emotional contexts (distress, hope, gratitude, urgency). A good model should be sensitive to these emotional nuances and avoid confusing emotionally similar but functionally different categories (e.g., sympathy vs urgent requests).

### Mathematical Definition
```
ECS = 1.0 - (Σ(confusion_penalty_ij) / max_possible_penalty)
```

Where confusion penalties are higher for confusions within the same emotional group.

### Emotional Grouping Strategy

| Emotional Group | Categories | Rationale |
|----------------|------------|-----------|
| **active_help** | rescue_volunteering_or_donation_effort, requests_or_urgent_needs | Active assistance seeking/offering |
| **distress_reporting** | injured_or_dead_people, missing_or_found_people, displaced_people_and_evacuations | Reporting distressing situations |
| **passive_support** | sympathy_and_support, other_relevant_information | Providing moral/informational support |
| **informational** | infrastructure_and_utility_damage, caution_and_advice, other_relevant_information | Factual information sharing |

### Confusion Penalty Matrix
- **Within same emotional group**: 2.0x penalty (more problematic)
- **Between different groups**: 0.5x penalty (less problematic)  
- **Base confusion**: 1.0x penalty (default)

### Implementation Logic
```python
def compute_ecs(y_true, y_pred, label_names, emotional_groups):
    total_penalty = 0.0
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            true_class = label_names[true_label]
            pred_class = label_names[pred_label]
            
            true_group = get_emotional_group(true_class)
            pred_group = get_emotional_group(pred_class)
            
            if true_group == pred_group:
                penalty = 2.0  # Higher penalty for same emotional context
            else:
                penalty = 0.5  # Lower penalty for different contexts
            
            total_penalty += penalty
    
    # Convert to score (higher = better)
    max_penalty = len(y_true) * 2.0
    return max(0.0, 1.0 - (total_penalty / max_penalty))
```

### Interpretation Scale
- **0.80-1.00**: Excellent emotional sensitivity - distinguishes well between contexts
- **0.70-0.79**: Good emotional sensitivity - mostly appropriate understanding
- **0.60-0.69**: Moderate emotional sensitivity - some confusion in similar contexts
- **0.50-0.59**: Poor emotional sensitivity - frequent contextual confusion
- **0.00-0.49**: Very poor emotional sensitivity - lacks contextual understanding

### Why ECS Matters
1. **Mental health**: Proper routing of emotional content to counseling vs emergency services
2. **Response appropriateness**: Different emotional contexts require different responses
3. **Human-centered**: Recognizes the psychological aspects of disaster response
4. **Operational efficiency**: Reduces misrouting of emotionally charged messages

---

## ⚡ Custom Metric 3: Actionability Relevance (AR)

### Theory
In disaster response, information can be classified as actionable (requires immediate response) or informational (provides context). This metric evaluates how well the model distinguishes between content that requires action versus content that is purely informational, which is critical for routing messages to appropriate response teams.

### Mathematical Definition
```
AR = Σ(actionability_weight_i × correct_predictions_i) / Σ(actionability_weight_i × total_predictions_i)
```

### Actionability Level Classification

| Actionability Level | Categories | Weight | Response Required |
|-------------------|------------|--------|------------------|
| **high_action** | injured_or_dead_people, requests_or_urgent_needs, missing_or_found_people | 4.0 | Immediate emergency response |
| **medium_action** | rescue_volunteering_or_donation_effort, displaced_people_and_evacuations, infrastructure_and_utility_damage | 2.5 | Coordinated response needed |
| **low_action** | caution_and_advice | 1.5 | Public information dissemination |
| **informational** | other_relevant_information, sympathy_and_support | 1.0 | Monitoring and archival |
| **no_action** | not_humanitarian, dont_know_cant_judge | 0.5 | No response needed |

### Implementation with Detailed Breakdown
```python
def compute_ar_with_breakdown(y_true, y_pred, label_names, actionability_mapping, action_weights):
    # Group by actionability level
    level_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_class = label_names[true_label]
        action_level = actionability_mapping[true_class]
        
        level_performance[action_level]['total'] += 1
        if true_label == pred_label:
            level_performance[action_level]['correct'] += 1
    
    # Calculate weighted score
    total_weighted_correct = 0.0
    total_weighted_samples = 0.0
    
    for level, performance in level_performance.items():
        weight = action_weights[level]
        correct = performance['correct']
        total = performance['total']
        
        total_weighted_samples += weight * total
        total_weighted_correct += weight * correct
    
    return total_weighted_correct / total_weighted_samples, level_performance
```

### Interpretation Scale
- **0.80-1.00**: Excellent actionability detection - reliable for emergency routing
- **0.70-0.79**: Good actionability detection - suitable for disaster response systems
- **0.60-0.69**: Moderate actionability detection - may miss some urgent situations
- **0.50-0.59**: Poor actionability detection - risk of delayed emergency response
- **0.00-0.49**: Very poor actionability detection - unsuitable for operational use

### Operational Value of AR
1. **Resource allocation**: Directs limited resources to actionable situations
2. **Response routing**: Automatically routes messages to appropriate teams
3. **Information filtering**: Reduces noise for emergency responders
4. **Workflow optimization**: Streamlines disaster response operations

---

## 🔬 Technical Implementation Guide

### Adding the Custom Metrics to Your Project

1. **Install the custom metrics module**:
   ```bash
   # Save custom_metrics.py in your project directory
   # Save enhanced_evaluator.py in your project directory
   # Save main_enhanced.py in your project directory
   ```

2. **Run with custom metrics**:
   ```bash
   # Learn about the metrics theory
   python main_enhanced.py --mode metrics-theory
   
   # Quick demonstration
   python main_enhanced.py --mode quick-custom
   
   # Full evaluation with custom metrics
   python main_enhanced.py --mode evaluation
   ```

3. **Integration with existing code**:
   ```python
   from custom_metrics import AdvancedMetricsEvaluator
   from enhanced_evaluator import EnhancedModelEvaluator
   
   # Initialize
   evaluator = AdvancedMetricsEvaluator()
   
   # Compute all metrics
   results = evaluator.evaluate_all_metrics(
       y_true, y_pred, texts, label_names
   )
   ```

### Output Files Generated

The enhanced system generates several additional files:

- `comprehensive_evaluation_results.json` - All metrics in structured format
- `comprehensive_evaluation_report.md` - Human-readable analysis
- `comprehensive_metrics_comparison.png` - Visual comparison including custom metrics
- `actionability_breakdown.png` - Performance by actionability level
- `enhanced_confusion_matrices.png` - Confusion matrices with actionability annotations
- `metrics_correlation.png` - Correlation between standard and custom metrics
- `standard_metrics_comparison.csv` - Standard metrics table
- `custom_metrics_comparison.csv` - Custom metrics table
- `{model}_detailed_predictions.csv` - Predictions with actionability/urgency analysis

---

## 📈 Advantages of Our Custom Metrics Approach

### 1. **Domain Relevance**
- Specifically designed for disaster response scenarios
- Captures nuances that matter in emergency situations
- Based on real operational needs of disaster response teams

### 2. **Complementary to Standard Metrics**
- Work alongside standard metrics, don't replace them
- Provide additional dimensions of model performance
- Help identify specific strengths and weaknesses

### 3. **Actionable Insights**
- Provide clear guidance on model improvement
- Help prioritize training data collection
- Guide feature engineering efforts

### 4. **Real-World Impact**
- Better model selection for operational deployment
- Improved resource allocation in disasters
- Enhanced emergency response effectiveness

### 5. **Extensible Framework**
- Easy to add new custom metrics
- Modular design allows metric customization
- Adaptable to other disaster types or domains

---

## 🎓 Educational Value

### For Students and Researchers
- Demonstrates how to design domain-specific evaluation metrics
- Shows the limitations of standard metrics in specialized applications
- Provides a framework for developing custom metrics in other domains

### For Practitioners
- Practical example of moving from research to operational deployment
- Consideration of real-world constraints and priorities
- Framework for evaluating AI systems in high-stakes environments

---

## 🚀 Future Extensions

### Potential Additional Custom Metrics
1. **Temporal Sensitivity**: Consider time-criticality of different disaster phases
2. **Geographic Relevance**: Weight predictions by geographic clustering
3. **Resource Availability**: Consider local resource constraints in evaluation
4. **Multi-language Sensitivity**: Evaluate performance across different languages
5. **Reliability Under Stress**: Evaluate performance during high-volume periods

### Advanced Features
- **Dynamic weighting**: Adjust weights based on disaster type and phase
- **Uncertainty quantification**: Include confidence scores in evaluation
- **Feedback integration**: Learn from operational feedback to improve metrics
- **Cross-disaster validation**: Evaluate generalization across disaster types

---

## 📚 References and Theoretical Foundation

### Academic Background
- **Information Retrieval**: Precision and recall concepts
- **Multi-criteria Decision Analysis**: Weighted scoring methods  
- **Emergency Management**: Disaster response priority frameworks
- **Natural Language Processing**: Emotion detection and sentiment analysis
- **Human-Computer Interaction**: User-centered evaluation approaches

### Industry Applications
- **Emergency Response Systems**: FEMA, Red Cross operational priorities
- **Social Media Monitoring**: Twitter crisis management best practices
- **Public Health**: WHO emergency communication guidelines
- **Disaster Risk Reduction**: UN Sendai Framework priorities

This comprehensive custom metrics framework represents a significant advancement in domain-specific AI evaluation, moving beyond generic metrics to capture what truly matters in disaster response applications.