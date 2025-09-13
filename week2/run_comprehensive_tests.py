#!/usr/bin/env python3
"""
Main script for running comprehensive bias and adversarial testing
Usage: python run_comprehensive_tests.py [options]
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Import our modules
from config import get_default_config, ProjectConfig
from data_loader import DatasetLoader
from model_handler import ModelManager
from comprehensive_testing import ComprehensiveModelTester

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('comprehensive_testing.log')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def run_comprehensive_safety_testing(config: ProjectConfig, models_to_test: list = None):
    """Run comprehensive safety testing including bias and adversarial attacks"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL SAFETY TESTING")
    print("Bias Testing + Adversarial Attacks + Security Analysis")
    print("="*80)
    
    start_time = time.time()
    
    # Initialize components
    dataset_loader = DatasetLoader(config.dataset)
    model_manager = ModelManager(config.models)
    
    # Load dataset
    print("\n1️⃣ Loading dataset...")
    dataset_loader.load_dataset()
    info = dataset_loader.get_dataset_info()
    print(f"   ✓ Dataset: {info['name']}")
    print(f"   ✓ Classes: {info['num_classes']}")
    
    # Load models
    print("\n2️⃣ Loading models for testing...")
    if models_to_test:
        available_models = [m for m in models_to_test if m in config.models]
        if not available_models:
            print(f"❌ Error: None of the specified models {models_to_test} are configured")
            return
    else:
        available_models = None
    
    load_results = model_manager.load_models(available_models)
    
    if not any(load_results.values()):
        print("❌ Error: No models loaded successfully")
        return
    
    loaded_models = [name for name, success in load_results.items() if success]
    print(f"   ✅ Models loaded: {loaded_models}")
    
    # Get sample texts for adversarial testing
    print("\n3️⃣ Preparing test data...")
    test_texts, test_labels = dataset_loader.get_split_data('test', max_samples=100)
    sample_texts = test_texts[:20]  # Use sample for adversarial generation
    print(f"   ✓ Sample texts prepared: {len(sample_texts)}")
    
    # Run comprehensive testing for each model
    all_results = {}
    
    for model_name in loaded_models:
        print(f"\n{'='*20} TESTING {model_name.upper()} {'='*20}")
        
        handler = model_manager.get_handler(model_name)
        if not handler:
            print(f"❌ Handler for {model_name} not found")
            continue
        
        # Initialize comprehensive tester
        tester = ComprehensiveModelTester(dataset_loader.label_names)
        
        # Set up output directory for this model
        model_output_dir = f"./comprehensive_test_results/{model_name}"
        
        try:
            # Run comprehensive testing
            print(f"🔬 Running comprehensive safety analysis for {model_name}...")
            results = tester.run_comprehensive_testing(
                model_handler=handler,
                base_texts=sample_texts,
                output_dir=model_output_dir
            )
            
            all_results[model_name] = results
            
            # Display summary results
            analysis = results.get('comprehensive_analysis', {})
            readiness = analysis.get('deployment_readiness', {})
            
            print(f"\n📊 {model_name.upper()} SAFETY SUMMARY:")
            print(f"   🎯 Deployment Readiness: {readiness.get('readiness_level', 'Unknown')}")
            print(f"   📈 Combined Safety Score: {readiness.get('combined_score', 0):.3f}/1.000")
            print(f"   ⚠️ Critical Issues: {readiness.get('critical_issue_count', 0)}")
            print(f"   💡 Recommendation: {readiness.get('recommendation', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            logging.error(f"Error in comprehensive testing for {model_name}", exc_info=True)
            continue
    
    # Generate comparative analysis if multiple models tested
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARATIVE SAFETY ANALYSIS")
        print(f"{'='*60}")
        
        comparison_data = []
        for model_name, results in all_results.items():
            analysis = results.get('comprehensive_analysis', {})
            readiness = analysis.get('deployment_readiness', {})
            bias_score = analysis.get('bias_risk_score', {}).get('score', 0)
            security_score = analysis.get('security_analysis', {}).get('overall_score', 0)
            
            comparison_data.append({
                'Model': model_name.upper(),
                'Readiness Level': readiness.get('readiness_level', 'Unknown'),
                'Combined Score': f"{readiness.get('combined_score', 0):.3f}",
                'Bias Score': f"{bias_score:.3f}",
                'Security Score': f"{security_score:.3f}",
                'Critical Issues': readiness.get('critical_issue_count', 0)
            })
        
        # Display comparison table
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv("./comprehensive_test_results/model_safety_comparison.csv", index=False)
        
        # Determine overall recommendations
        print(f"\n🏆 SAFETY RANKINGS:")
        
        # Best overall safety
        best_combined = max(all_results.keys(), key=lambda x: 
            all_results[x].get('comprehensive_analysis', {}).get('deployment_readiness', {}).get('combined_score', 0))
        
        # Best bias performance
        best_bias = max(all_results.keys(), key=lambda x: 
            all_results[x].get('comprehensive_analysis', {}).get('bias_risk_score', {}).get('score', 0))
        
        # Best security
        best_security = max(all_results.keys(), key=lambda x: 
            all_results[x].get('comprehensive_analysis', {}).get('security_analysis', {}).get('overall_score', 0))
        
        print(f"   🥇 Best Overall Safety: {best_combined.upper()}")
        print(f"   ⚖️ Best Bias Performance: {best_bias.upper()}")
        print(f"   🛡️ Best Security: {best_security.upper()}")
        
        # Deployment recommendations
        deployable_models = [
            name for name, results in all_results.items()
            if results.get('comprehensive_analysis', {}).get('deployment_readiness', {}).get('readiness_level') 
            in ['Ready for Deployment', 'Deploy with Caution']
        ]
        
        print(f"\n✅ DEPLOYMENT RECOMMENDATIONS:")
        if deployable_models:
            print(f"   Models ready for deployment: {', '.join([m.upper() for m in deployable_models])}")
        else:
            print(f"   ⚠️ No models currently meet deployment safety standards")
            print(f"   🔧 All models require safety improvements before deployment")
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"⏱️ Total Testing Time: {duration/60:.1f} minutes")
    print(f"🔬 Models Tested: {len(all_results)}")
    print(f"📁 Results Location: ./comprehensive_test_results/")
    print(f"📊 Detailed Reports: Available in each model's subdirectory")
    
    # List generated files
    print(f"\n📋 KEY OUTPUT FILES:")
    print(f"   • model_safety_comparison.csv - Comparative safety analysis")
    print(f"   • {model_name}/comprehensive_testing_report.md - Main report for each model")
    print(f"   • {model_name}/bias_analysis_report.md - Detailed bias analysis")
    print(f"   • {model_name}/security_analysis_report.md - Security vulnerability analysis")
    print(f"   • {model_name}/deployment_dashboard.png - Visual safety dashboard")
    print(f"   • {model_name}/combined_risk_overview.png - Risk overview")

def run_bias_testing_only(config: ProjectConfig, models_to_test: list = None):
    """Run only bias testing"""
    
    print("="*60)
    print("BIAS TESTING ONLY")
    print("="*60)
    
    # Load components
    dataset_loader = DatasetLoader(config.dataset)
    model_manager = ModelManager(config.models)
    
    dataset_loader.load_dataset()
    load_results = model_manager.load_models(models_to_test)
    loaded_models = [name for name, success in load_results.items() if success]
    
    from bias_testing import BiasTestSuite
    
    for model_name in loaded_models:
        print(f"\n🔍 Running bias testing for {model_name.upper()}...")
        
        handler = model_manager.get_handler(model_name)
        bias_tester = BiasTestSuite(dataset_loader.label_names)
        
        # Generate and run bias tests
        bias_prompts = bias_tester.generate_bias_test_prompts()
        bias_results = bias_tester.evaluate_model_bias(handler, bias_prompts)
        
        # Generate visualizations and reports
        output_dir = Path(f"./bias_test_results/{model_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        bias_tester.visualize_bias_results(str(output_dir / "bias_analysis.png"))
        bias_tester.create_bias_heatmap(str(output_dir / "bias_heatmap.png"))
        bias_tester.generate_bias_report(str(output_dir / "bias_report.md"))
        
        print(f"   ✅ Bias testing complete for {model_name}")
        print(f"   📁 Results saved to: {output_dir}")

def run_adversarial_testing_only(config: ProjectConfig, models_to_test: list = None):
    """Run only adversarial testing"""
    
    print("="*60)
    print("ADVERSARIAL TESTING ONLY")
    print("="*60)
    
    # Load components
    dataset_loader = DatasetLoader(config.dataset)
    model_manager = ModelManager(config.models)
    
    dataset_loader.load_dataset()
    load_results = model_manager.load_models(models_to_test)
    loaded_models = [name for name, success in load_results.items() if success]
    
    # Get sample texts
    test_texts, _ = dataset_loader.get_split_data('test', max_samples=50)
    
    from adversarial_attacks import AdversarialAttackSuite
    
    for model_name in loaded_models:
        print(f"\n🛡️ Running adversarial testing for {model_name.upper()}...")
        
        handler = model_manager.get_handler(model_name)
        attack_tester = AdversarialAttackSuite(dataset_loader.label_names)
        
        # Generate and run attacks
        attack_prompts = attack_tester.generate_adversarial_prompts(test_texts)
        attack_results = attack_tester.evaluate_model_robustness(handler, attack_prompts)
        
        # Generate visualizations and reports
        output_dir = Path(f"./adversarial_test_results/{model_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        attack_tester.visualize_attack_results(str(output_dir / "attack_analysis.png"))
        attack_tester.create_vulnerability_heatmap(str(output_dir / "vulnerability_heatmap.png"))
        attack_tester.generate_robustness_report(str(output_dir / "security_report.md"))
        
        # Show security score
        security_score = attack_tester.get_security_score()
        print(f"   📊 Security Score: {security_score['overall_score']:.3f}/1.000")
        print(f"   🎯 Security Rating: {security_score['security_rating']}")
        print(f"   💡 Deployment Rec: {security_score['deployment_recommendation']}")
        
        print(f"   ✅ Adversarial testing complete for {model_name}")
        print(f"   📁 Results saved to: {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Safety Testing: Bias + Adversarial Attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_tests.py                           # Full safety testing (both models)
  python run_comprehensive_tests.py --models bert             # Test only BERT
  python run_comprehensive_tests.py --mode bias-only          # Only bias testing
  python run_comprehensive_tests.py --mode adversarial-only   # Only adversarial testing
  python run_comprehensive_tests.py --quick                   # Quick test with fewer samples
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['comprehensive', 'bias-only', 'adversarial-only'],
        default='comprehensive',
        help='Testing mode: comprehensive (both), bias-only, or adversarial-only'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['bert', 'llama'],
        help='Models to test (default: all configured models)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with reduced sample sizes'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./comprehensive_test_results',
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
    
    logger.info("Starting Comprehensive Model Safety Testing")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Models: {args.models or 'all'}")
    
    # Load configuration
    config = get_default_config()
    
    # Adjust for quick testing
    if args.quick:
        config.dataset.max_samples = 50
        logger.info("Quick testing mode enabled - using reduced sample sizes")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'comprehensive':
            run_comprehensive_safety_testing(config, args.models)
        elif args.mode == 'bias-only':
            run_bias_testing_only(config, args.models)
        elif args.mode == 'adversarial-only':
            run_adversarial_testing_only(config, args.models)
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1
    
    logger.info("Safety testing completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)