#!/usr/bin/env python3
"""
Enhanced main entry point for LLM evaluation project with custom metrics
Usage: python main_enhanced.py [options]
"""

import argparse
import logging
import sys
from pathlib import Path

# Import our modules
from config import get_default_config, ProjectConfig
from data_loader import DatasetLoader, DatasetAnalyzer
from model_handler import ModelManager
from enhanced_evaluator import EnhancedModelEvaluator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llm_evaluation_enhanced.log')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def run_enhanced_evaluation(config: ProjectConfig, models_to_eval: list = None):
    """Run comprehensive evaluation with custom metrics"""
    print("="*70)
    print("ENHANCED LLM EVALUATION WITH CUSTOM DISASTER METRICS")
    print("="*70)
    
    # Initialize components
    dataset_loader = DatasetLoader(config.dataset)
    model_manager = ModelManager(config.models)
    enhanced_evaluator = EnhancedModelEvaluator(config.evaluation, dataset_loader, model_manager)
    
    # Load dataset
    print("\n1️⃣ Loading and analyzing dataset...")
    dataset_loader.load_dataset()
    
    # Quick dataset overview
    info = dataset_loader.get_dataset_info()
    print(f"   ✓ Dataset: {info['name']}")
    print(f"   ✓ Total samples: {info['total_samples']:,}")
    print(f"   ✓ Classes: {info['num_classes']}")
    print(f"   ✓ Test samples: {info['split_sizes'].get('test', 'N/A')}")
    
    # Load models
    print("\n2️⃣ Loading models...")
    if models_to_eval:
        available_models = [m for m in models_to_eval if m in config.models]
        if not available_models:
            print(f"❌ Error: None of the specified models {models_to_eval} are configured")
            print(f"Available models: {list(config.models.keys())}")
            return
    else:
        available_models = None
    
    load_results = model_manager.load_models(available_models)
    
    if not any(load_results.values()):
        print("❌ Error: No models loaded successfully")
        return
    
    loaded_models = [name for name, success in load_results.items() if success]
    print(f"   ✅ Successfully loaded models: {loaded_models}")
    
    # Get test data
    print("\n3️⃣ Preparing test data...")
    test_texts, test_labels = dataset_loader.get_split_data(
        split_name='test',
        max_samples=config.dataset.max_samples
    )
    print(f"   ✓ Test samples: {len(test_texts):,}")
    if config.dataset.max_samples:
        print(f"   ℹ️ Limited to {config.dataset.max_samples} samples for testing")
    
    # Run enhanced evaluation
    print("\n4️⃣ Running comprehensive evaluation...")
    print("   📊 Computing standard classification metrics...")
    print("   🎯 Computing custom disaster-specific metrics...")
    print("   ⚡ Analyzing actionability levels...")
    print("   💭 Evaluating emotional context sensitivity...")
    print("   🚨 Measuring urgency awareness...")
    
    results = enhanced_evaluator.evaluate_models_with_custom_metrics(
        test_texts, test_labels, loaded_models
    )
    
    if not results:
        print("❌ Error: No evaluation results generated")
        return
    
    # Generate comprehensive comparison
    print("\n5️⃣ Generating comprehensive analysis...")
    comparison_df = enhanced_evaluator.compare_models_comprehensive()
    
    # Get model rankings
    rankings = enhanced_evaluator.get_model_rankings()
    if rankings:
        print(f"\n🏆 MODEL RANKINGS:")
        print(f"   🥇 Best Overall (Standard): {rankings['overall_standard'].upper()}")
        print(f"   🎯 Best Urgency Detection: {rankings['urgency_awareness'].upper()}")  
        print(f"   💭 Best Emotional Sensitivity: {rankings['emotional_sensitivity'].upper()}")
        print(f"   ⚡ Best Actionability Detection: {rankings['actionability_detection'].upper()}")
        print(f"   ⚡ Fastest Model: {rankings['speed'].upper()}")
        print(f"   🏅 Most Balanced: {rankings['balanced_performance'].upper()}")
    
    # Generate final report
    print("\n6️⃣ Generating comprehensive report...")
    enhanced_evaluator.generate_comprehensive_report()
    
    print(f"\n✅ Enhanced evaluation complete!")
    print(f"📁 Results saved to: {config.evaluation.output_dir}")
    
    # Show output files
    output_dir = Path(config.evaluation.output_dir)
    output_files = [
        "comprehensive_evaluation_results.json",
        "comprehensive_evaluation_report.md", 
        "comprehensive_metrics_comparison.png",
        "actionability_breakdown.png",
        "enhanced_confusion_matrices.png",
        "metrics_correlation.png",
        "standard_metrics_comparison.csv",
        "custom_metrics_comparison.csv"
    ]
    
    print(f"\n📋 Generated Files:")
    for filename in output_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ⚠️ {filename} (not generated)")

def run_metrics_analysis_only(config: ProjectConfig):
    """Run analysis focusing on metrics explanation and theory"""
    print("="*60)
    print("CUSTOM METRICS ANALYSIS & THEORY")
    print("="*60)
    
    from custom_metrics import (
        DisasterUrgencyAwarenessMetric, 
        EmotionalContextSensitivityMetric,
        ActionabilityRelevanceMetric,
        StandardMetricsCalculator
    )
    
    print("\n📚 STANDARD CLASSIFICATION METRICS:")
    print("="*50)
    
    standard_metrics_info = {
        'Accuracy': 'Overall proportion of correct predictions',
        'Precision (Weighted)': 'Precision averaged by class frequency - good for imbalanced data',
        'Precision (Macro)': 'Unweighted average precision across classes',
        'Precision (Micro)': 'Global precision computed from total true/false positives',
        'Recall (Weighted)': 'Recall averaged by class frequency',
        'Recall (Macro)': 'Unweighted average recall across classes', 
        'Recall (Micro)': 'Global recall computed from total true/false positives',
        'F1 (Weighted)': 'Harmonic mean of precision/recall, weighted by class frequency',
        'F1 (Macro)': 'Unweighted average F1 across classes - treats all classes equally',
        'F1 (Micro)': 'Global F1 computed from global precision/recall',
        'Balanced Accuracy': 'Average recall per class - handles class imbalance well',
        'Matthews Correlation Coefficient': 'Correlation between predictions and truth (-1 to +1)'
    }
    
    for metric, description in standard_metrics_info.items():
        print(f"📊 {metric:25} : {description}")
    
    print(f"\n🎯 CUSTOM DISASTER-SPECIFIC METRICS:")
    print("="*50)
    
    # Initialize custom metrics to show their properties
    dua_metric = DisasterUrgencyAwarenessMetric()
    ecs_metric = EmotionalContextSensitivityMetric() 
    ar_metric = ActionabilityRelevanceMetric()
    
    print(f"\n1️⃣ DISASTER URGENCY AWARENESS (DUA)")
    print(f"   📋 Purpose: {dua_metric.description}")
    print(f"   🔢 Formula: Σ(urgency_weight_i × correct_i) / Σ(urgency_weight_i × total_i)")
    print(f"   💡 Why Important:")
    print(f"      • In disasters, urgent situations (injuries, urgent needs) are more critical")
    print(f"      • Standard metrics treat all errors equally")
    print(f"      • DUA penalizes urgent misclassifications more heavily")
    print(f"      • Directly applicable to emergency response prioritization")
    
    print(f"\n   ⚖️ Urgency Weights:")
    for category, weight in sorted(dua_metric.urgency_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"      • {category:35} : {weight}")
    
    print(f"\n2️⃣ EMOTIONAL CONTEXT SENSITIVITY (ECS)")
    print(f"   📋 Purpose: {ecs_metric.description}")
    print(f"   💡 Why Important:")
    print(f"      • Disaster tweets contain complex emotional contexts")
    print(f"      • Critical to distinguish passive support from active requests")
    print(f"      • Helps route messages appropriately (counseling vs emergency response)")
    print(f"      • Standard metrics miss emotional nuances")
    
    print(f"\n   🧠 Emotional Groups:")
    for group, categories in ecs_metric.emotional_groups.items():
        print(f"      • {group:20} : {', '.join(categories)}")
    
    print(f"\n   ⚠️ Confusion Penalties:")
    print(f"      • Within same emotional group: {ecs_metric.confusion_penalties['within_group']}x penalty")
    print(f"      • Between different groups: {ecs_metric.confusion_penalties['between_group']}x penalty")
    print(f"      • Base confusion penalty: {ecs_metric.confusion_penalties['base']}x penalty")
    
    print(f"\n3️⃣ ACTIONABILITY RELEVANCE (AR)")
    print(f"   📋 Purpose: {ar_metric.description}")
    print(f"   💡 Why Important:")
    print(f"      • Distinguishes content requiring action from informational content")
    print(f"      • Critical for resource allocation during emergencies")
    print(f"      • Reduces information overload for emergency responders")
    print(f"      • Maps directly to operational needs")
    
    print(f"\n   ⚡ Actionability Levels & Weights:")
    for category, level in ar_metric.actionability_mapping.items():
        weight = ar_metric.action_weights[level]
        print(f"      • {category:35} : {level:15} (weight: {weight})")
    
    print(f"\n📈 THEORETICAL ADVANTAGES OF CUSTOM METRICS:")
    print("="*50)
    advantages = [
        "Domain Relevance: Tailored specifically for disaster response scenarios",
        "Operational Value: Directly applicable to emergency management systems", 
        "Resource Optimization: Help prioritize limited emergency response resources",
        "Human-Centered: Consider emotional and psychological aspects of disasters",
        "Actionable Insights: Provide specific guidance for model improvement",
        "Real-World Impact: Better alignment with actual disaster response needs",
        "Complementary: Work alongside standard metrics for comprehensive evaluation"
    ]
    
    for i, advantage in enumerate(advantages, 1):
        print(f"{i}. {advantage}")
    
    print(f"\n🔬 IMPLEMENTATION DETAILS:")
    print("="*50)
    print("✅ All metrics are normalized to [0, 1] range for easy comparison")
    print("✅ Higher scores always indicate better performance")  
    print("✅ Metrics include interpretation functions for actionable insights")
    print("✅ Detailed breakdown available for error analysis")
    print("✅ Extensible framework for adding new domain-specific metrics")

def run_quick_custom_test(config: ProjectConfig):
    """Run quick test focusing on custom metrics demonstration"""
    print("="*60)
    print("QUICK CUSTOM METRICS DEMONSTRATION")
    print("="*60)
    
    # Limit samples for quick demo
    config.dataset.max_samples = 50
    print(f"ℹ️ Using {config.dataset.max_samples} samples for quick demonstration")
    
    # Run enhanced evaluation
    run_enhanced_evaluation(config, ['bert'])  # Only BERT for speed

def main():
    """Main function with enhanced options"""
    parser = argparse.ArgumentParser(
        description="Enhanced LLM Evaluation: BERT vs LLaMA with Custom Disaster Response Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_enhanced.py --mode evaluation                    # Full evaluation with custom metrics
  python main_enhanced.py --mode quick-custom                 # Quick demo of custom metrics
  python main_enhanced.py --mode metrics-theory               # Learn about custom metrics theory
  python main_enhanced.py --mode evaluation --models bert     # Evaluate only BERT
  python main_enhanced.py --mode evaluation --max-samples 500 # Limit test samples
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['evaluation', 'quick-custom', 'metrics-theory', 'analysis'],
        default='evaluation',
        help='Run mode: evaluation (full with custom metrics), quick-custom (demo), metrics-theory (learn), analysis (dataset only)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['bert', 'llama'],
        help='Models to evaluate (default: all configured models)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of test samples (for testing purposes)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results_enhanced',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Enhanced LLM Evaluation Project with Custom Metrics")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load configuration
    config = get_default_config()
    
    # Apply command line overrides
    if args.max_samples:
        config.dataset.max_samples = args.max_samples
    
    config.evaluation.output_dir = args.output_dir
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'metrics-theory':
            run_metrics_analysis_only(config)
        elif args.mode == 'quick-custom':
            run_quick_custom_test(config)
        elif args.mode == 'analysis':
            from data_loader import DatasetLoader, DatasetAnalyzer
            # Dataset analysis only
            dataset_loader = DatasetLoader(config.dataset)
            dataset_loader.load_dataset()
            analyzer = DatasetAnalyzer(dataset_loader)
            analyzer.analyze_dataset()
            analyzer.plot_class_distribution(save_path=f"{config.evaluation.output_dir}/class_distribution.png")
            analyzer.plot_text_length_distribution(save_path=f"{config.evaluation.output_dir}/text_length_distribution.png")
            analyzer.generate_analysis_report(f"{config.evaluation.output_dir}/dataset_analysis.md")
        else:  # evaluation
            run_enhanced_evaluation(config, args.models)
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
    
    logger.info("Enhanced evaluation completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    