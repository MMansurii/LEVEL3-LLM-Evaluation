#!/usr/bin/env python3
"""
Week 4: Automated Evaluation Script
Main entry point for automated BERT evaluation pipeline

Usage:
    python run_automated_evaluation.py                    # Default full evaluation
    python run_automated_evaluation.py --quick           # Quick test
    python run_automated_evaluation.py --config custom   # Custom configuration
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

from pipeline_config import (
    get_default_pipeline_config, 
    get_quick_test_config, 
    get_production_config
)
from pipeline_orchestrator import PipelineOrchestrator

def setup_logging(verbose: bool = False):
    """Setup logging for the automation script"""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Initial logging setup (will be enhanced by orchestrator)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def print_banner():
    """Print welcome banner"""
    
    print("="*80)
    print(" BERT DISASTER RESPONSE MODEL - AUTOMATED EVALUATION PIPELINE")
    print("="*80)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Evaluation Framework Version: 1.0.0")
    print("="*80)

def print_pipeline_summary(pipeline_state):
    """Print pipeline execution summary"""
    
    print("\n" + "="*80)
    print(" PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    print(f" ⏱️  Total Execution Time: {pipeline_state['duration']:.1f} seconds")
    print(f" ✅ Stages Completed: {len(pipeline_state['stages_completed'])}/7")
    print(f" ❌ Errors Encountered: {len(pipeline_state['errors'])}")
    
    if pipeline_state['stages_completed']:
        print(f"\n 📊 Stage Performance:")
        for stage in pipeline_state['stages_completed']:
            status_icon = "✅" if stage['status'] == 'completed' else "❌"
            print(f"    {status_icon} {stage['name']}: {stage['duration']:.1f}s")
    
    if pipeline_state['errors']:
        print(f"\n ⚠️  Errors:")
        for error in pipeline_state['errors']:
            print(f"    ❌ {error['stage']}: {error['error']}")
    
    print("="*80)

def run_evaluation_pipeline(config_type: str = 'default', verbose: bool = False):
    """Run the complete evaluation pipeline"""
    
    print_banner()
    
    # Load appropriate configuration
    if config_type == 'quick':
        config = get_quick_test_config()
        print(" 🚀 Running QUICK TEST configuration")
    elif config_type == 'production':
        config = get_production_config()
        print(" 🏭 Running PRODUCTION configuration")
    else:
        config = get_default_pipeline_config()
        print(" 📋 Running DEFAULT configuration")
    
    print(f" 📁 Results will be saved to: {config.output_dir}")
    print()
    
    # Initialize and run pipeline
    try:
        orchestrator = PipelineOrchestrator(config)
        
        print("🔄 Starting automated evaluation pipeline...")
        pipeline_state = orchestrator.run_complete_pipeline()
        
        # Print summary
        print_pipeline_summary(pipeline_state)
        
        # Print results location
        results_summary = orchestrator.get_results_summary()
        if results_summary['status'] == 'completed':
            print(f"\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"📁 All results available in: {results_summary['output_directory']}")
            print(f"\n📋 Key Output Files:")
            print(f"   • Main Report: results/reports/main_evaluation_report.md")
            print(f"   • Executive Summary: results/reports/executive_summary.md") 
            print(f"   • Dashboard: results/visualizations/integrated_dashboard.png")
            print(f"   • Technical Details: results/reports/technical_report.md")
            
        else:
            print(f"\n⚠️ EVALUATION COMPLETED WITH ISSUES")
            print(f"📁 Partial results available in: {results_summary['output_directory']}")
            print(f"📋 Check logs for detailed error information")
        
        return 0 if results_summary['status'] == 'completed' else 1
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        logging.error(f"Pipeline execution failed", exc_info=True)
        return 1

def validate_environment():
    """Validate that the environment is ready for evaluation"""
    
    print("🔍 Validating environment...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ is required")
    
    # Check required packages
    required_packages = [
        'torch', 'transformers', 'datasets', 'sklearn', 
        'matplotlib', 'seaborn', 'pandas', 'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Required package '{package}' not installed")
    
    # Check disk space (basic check)
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        if free_space_gb < 2:
            issues.append(f"Low disk space: {free_space_gb:.1f}GB available")
    except:
        pass
    
    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Please resolve these issues before running the evaluation")
        return False
    
    print("✅ Environment validation passed")
    return True

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Automated BERT Evaluation Pipeline for Disaster Response Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_automated_evaluation.py                    # Full evaluation
  python run_automated_evaluation.py --quick           # Quick test (reduced samples)
  python run_automated_evaluation.py --production      # Production-grade evaluation
  python run_automated_evaluation.py --verbose         # Detailed logging
  python run_automated_evaluation.py --validate-only   # Just validate environment
        """
    )
    
    parser.add_argument(
        '--config',
        choices=['default', 'quick', 'production'],
        default='default',
        help='Pipeline configuration to use'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick evaluation with reduced sample sizes'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Run production-grade evaluation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment, do not run evaluation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate environment first
    if not validate_environment():
        return 1
    
    if args.validate_only:
        print("✅ Environment validation complete. Ready for evaluation.")
        return 0
    
    # Determine configuration type
    config_type = 'default'
    if args.quick:
        config_type = 'quick'
    elif args.production:
        config_type = 'production'
    elif args.config != 'default':
        config_type = args.config
    
    # Run the evaluation pipeline
    return run_evaluation_pipeline(config_type, args.verbose)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)