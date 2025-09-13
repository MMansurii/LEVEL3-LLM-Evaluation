"""
Week 4: Pipeline Orchestrator
Coordinates the complete evaluation pipeline execution
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from datetime import datetime

from pipeline_config import PipelineConfig
from data_manager import DataManager
from model_evaluator import ModelEvaluator
from safety_evaluator import SafetyEvaluator
from visualization_engine import VisualizationEngine
from report_generator import ReportGenerator

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Main orchestrator for the complete evaluation pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_path = None
        self.pipeline_state = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'stages_completed': [],
            'results': {},
            'errors': []
        }
        
        # Initialize components
        self.data_manager = None
        self.model_evaluator = None
        self.safety_evaluator = None
        self.visualization_engine = None
        self.report_generator = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        # Create output directory first
        self.output_path = self.config.create_output_structure()
        
        # Configure logging
        log_file = self.output_path / "logs" / "pipeline.log"
        
        logging.basicConfig(
            level=logging.DEBUG if self.config.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Pipeline orchestrator initialized")
        logger.info(f"Output directory: {self.output_path}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete evaluation pipeline"""
        
        logger.info("="*80)
        logger.info("STARTING COMPLETE EVALUATION PIPELINE")
        logger.info("="*80)
        
        self.pipeline_state['start_time'] = datetime.now()
        
        try:
            # Stage 1: Setup and Data Loading
            self._execute_stage("setup_and_data", self._setup_and_load_data)
            
            # Stage 2: Model Evaluation (Standard + Custom Metrics)
            self._execute_stage("model_evaluation", self._run_model_evaluation)
            
            # Stage 3: Safety Evaluation (Bias + Adversarial)
            self._execute_stage("safety_evaluation", self._run_safety_evaluation)
            
            # Stage 4: Results Analysis and Integration
            self._execute_stage("results_analysis", self._analyze_and_integrate_results)
            
            # Stage 5: Visualization Generation
            self._execute_stage("visualization", self._generate_visualizations)
            
            # Stage 6: Report Generation
            self._execute_stage("report_generation", self._generate_reports)
            
            # Stage 7: Pipeline Finalization
            self._execute_stage("finalization", self._finalize_pipeline)
            
        except Exception as e:
            self._handle_pipeline_error(e)
            raise
        
        finally:
            self.pipeline_state['end_time'] = datetime.now()
            self.pipeline_state['duration'] = (
                self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            ).total_seconds()
        
        logger.info("="*80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info(f"Total Duration: {self.pipeline_state['duration']:.1f} seconds")
        logger.info("="*80)
        
        return self.pipeline_state
    
    def _execute_stage(self, stage_name: str, stage_function):
        """Execute a pipeline stage with error handling and timing"""
        
        logger.info(f"\n{'='*20} STAGE: {stage_name.upper()} {'='*20}")
        stage_start = time.time()
        
        try:
            result = stage_function()
            stage_duration = time.time() - stage_start
            
            self.pipeline_state['stages_completed'].append({
                'name': stage_name,
                'duration': stage_duration,
                'status': 'completed',
                'result': result
            })
            
            logger.info(f"✅ Stage '{stage_name}' completed in {stage_duration:.1f}s")
            return result
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            error_info = {
                'stage': stage_name,
                'error': str(e),
                'duration': stage_duration,
                'traceback': traceback.format_exc()
            }
            
            self.pipeline_state['errors'].append(error_info)
            logger.error(f"❌ Stage '{stage_name}' failed after {stage_duration:.1f}s: {e}")
            raise
    
    def _setup_and_load_data(self) -> Dict[str, Any]:
        """Stage 1: Setup components and load data"""
        
        logger.info("Initializing pipeline components...")
        
        # Initialize data manager
        self.data_manager = DataManager(self.config.dataset, self.output_path)
        
        # Load and prepare dataset
        dataset_info = self.data_manager.load_and_prepare_data()
        
        # Initialize other components
        self.model_evaluator = ModelEvaluator(
            self.config.model, 
            self.config.evaluation,
            self.output_path
        )
        
        self.safety_evaluator = SafetyEvaluator(
            self.config.evaluation,
            self.output_path
        )
        
        self.visualization_engine = VisualizationEngine(
            self.config.visualization,
            self.output_path
        )
        
        self.report_generator = ReportGenerator(
            self.config.report,
            self.output_path
        )
        
        # Save configuration for reproducibility
        self.config.save_config(self.output_path / "pipeline_config.json")
        
        return {
            'dataset_info': dataset_info,
            'components_initialized': True
        }
    
    def _run_model_evaluation(self) -> Dict[str, Any]:
        """Stage 2: Run standard and custom model evaluation"""
        
        logger.info("Running comprehensive model evaluation...")
        
        # Get data
        test_data = self.data_manager.get_test_data()
        
        # Run standard metrics evaluation
        standard_results = self.model_evaluator.evaluate_standard_metrics(test_data)
        
        # Run custom metrics evaluation  
        custom_results = self.model_evaluator.evaluate_custom_metrics(test_data)
        
        # Combine results
        evaluation_results = {
            'standard_metrics': standard_results,
            'custom_metrics': custom_results,
            'model_info': self.model_evaluator.get_model_info()
        }
        
        # Save intermediate results
        self.model_evaluator.save_results(evaluation_results)
        
        self.pipeline_state['results']['model_evaluation'] = evaluation_results
        
        return evaluation_results
    
    def _run_safety_evaluation(self) -> Dict[str, Any]:
        """Stage 3: Run bias and adversarial testing"""
        
        logger.info("Running comprehensive safety evaluation...")
        
        # Get model handler and data
        model_handler = self.model_evaluator.get_model_handler()
        test_data = self.data_manager.get_test_data()
        label_names = self.data_manager.get_label_names()
        
        # Run bias testing
        bias_results = None
        if self.config.evaluation.bias_testing:
            bias_results = self.safety_evaluator.run_bias_testing(
                model_handler, label_names
            )
        
        # Run adversarial testing
        adversarial_results = None
        if self.config.evaluation.adversarial_testing:
            adversarial_results = self.safety_evaluator.run_adversarial_testing(
                model_handler, label_names, test_data
            )
        
        # Integrate safety results
        safety_results = self.safety_evaluator.integrate_safety_results(
            bias_results, adversarial_results
        )
        
        # Save intermediate results
        self.safety_evaluator.save_results(safety_results)
        
        self.pipeline_state['results']['safety_evaluation'] = safety_results
        
        return safety_results
    
    def _analyze_and_integrate_results(self) -> Dict[str, Any]:
        """Stage 4: Analyze and integrate all results"""
        
        logger.info("Analyzing and integrating all evaluation results...")
        
        model_results = self.pipeline_state['results']['model_evaluation']
        safety_results = self.pipeline_state['results']['safety_evaluation']
        
        # Integrate all results
        integrated_results = self._integrate_all_results(model_results, safety_results)
        
        # Perform quality assessment
        quality_assessment = self._assess_quality_standards(integrated_results)
        
        # Generate deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(
            integrated_results, quality_assessment
        )
        
        analysis_results = {
            'integrated_results': integrated_results,
            'quality_assessment': quality_assessment,
            'deployment_recommendations': deployment_recommendations
        }
        
        self.pipeline_state['results']['analysis'] = analysis_results
        
        return analysis_results
    
    def _generate_visualizations(self) -> Dict[str, Any]:
        """Stage 5: Generate comprehensive visualizations"""
        
        logger.info("Generating comprehensive visualizations...")
        
        all_results = self.pipeline_state['results']
        
        # Generate standard metrics visualizations
        standard_viz = self.visualization_engine.create_standard_metrics_plots(
            all_results['model_evaluation']['standard_metrics']
        )
        
        # Generate custom metrics visualizations
        custom_viz = self.visualization_engine.create_custom_metrics_plots(
            all_results['model_evaluation']['custom_metrics']
        )
        
        # Generate safety visualizations
        safety_viz = self.visualization_engine.create_safety_plots(
            all_results['safety_evaluation']
        )
        
        # Generate integrated dashboard
        dashboard = self.visualization_engine.create_integrated_dashboard(
            all_results
        )
        
        visualization_results = {
            'standard_visualizations': standard_viz,
            'custom_visualizations': custom_viz,
            'safety_visualizations': safety_viz,
            'integrated_dashboard': dashboard
        }
        
        self.pipeline_state['results']['visualizations'] = visualization_results
        
        return visualization_results
    
    def _generate_reports(self) -> Dict[str, Any]:
        """Stage 6: Generate comprehensive reports"""
        
        logger.info("Generating comprehensive evaluation reports...")
        
        all_results = self.pipeline_state['results']
        
        # Generate main evaluation report
        main_report = self.report_generator.generate_main_report(
            all_results, self.pipeline_state
        )
        
        # Generate technical appendix
        technical_report = self.report_generator.generate_technical_report(
            all_results
        )
        
        # Generate executive summary
        executive_summary = self.report_generator.generate_executive_summary(
            all_results
        )
        
        report_results = {
            'main_report': main_report,
            'technical_report': technical_report,
            'executive_summary': executive_summary
        }
        
        self.pipeline_state['results']['reports'] = report_results
        
        return report_results
    
    def _finalize_pipeline(self) -> Dict[str, Any]:
        """Stage 7: Finalize pipeline and cleanup"""
        
        logger.info("Finalizing pipeline execution...")
        
        # Generate pipeline summary
        pipeline_summary = self._generate_pipeline_summary()
        
        # Save final state
        self._save_pipeline_state()
        
        # Create results index
        results_index = self._create_results_index()
        
        # Cleanup temporary files if needed
        self._cleanup_temporary_files()
        
        return {
            'pipeline_summary': pipeline_summary,
            'results_index': results_index
        }
    
    def _integrate_all_results(self, model_results: Dict, safety_results: Dict) -> Dict[str, Any]:
        """Integrate all evaluation results"""
        
        integrated = {
            'timestamp': datetime.now().isoformat(),
            'model_info': model_results.get('model_info', {}),
            'performance_metrics': {
                'standard': model_results.get('standard_metrics', {}),
                'custom': model_results.get('custom_metrics', {})
            },
            'safety_metrics': safety_results,
            'combined_scores': self._calculate_combined_scores(model_results, safety_results)
        }
        
        return integrated
    
    def _assess_quality_standards(self, results: Dict) -> Dict[str, Any]:
        """Assess results against quality standards"""
        
        thresholds = self.config.report.quality_thresholds
        
        # Extract key metrics
        standard_metrics = results['performance_metrics']['standard']
        safety_metrics = results['safety_metrics']
        
        # Check against thresholds
        quality_checks = {
            'accuracy_check': {
                'value': standard_metrics.get('accuracy', 0),
                'threshold': thresholds['min_accuracy'],
                'passed': standard_metrics.get('accuracy', 0) >= thresholds['min_accuracy']
            },
            'f1_check': {
                'value': standard_metrics.get('f1_weighted', 0),
                'threshold': thresholds['min_f1_weighted'],
                'passed': standard_metrics.get('f1_weighted', 0) >= thresholds['min_f1_weighted']
            }
        }
        
        # Overall quality assessment
        overall_passed = all(check['passed'] for check in quality_checks.values())
        
        return {
            'individual_checks': quality_checks,
            'overall_passed': overall_passed,
            'quality_score': sum(1 for check in quality_checks.values() if check['passed']) / len(quality_checks)
        }
    
    def _generate_deployment_recommendations(self, results: Dict, quality: Dict) -> Dict[str, Any]:
        """Generate deployment recommendations based on all results"""
        
        recommendations = {
            'deployment_ready': quality['overall_passed'],
            'confidence_level': 'high' if quality['quality_score'] > 0.8 else 'medium' if quality['quality_score'] > 0.6 else 'low',
            'primary_concerns': [],
            'action_items': [],
            'monitoring_requirements': []
        }
        
        # Add specific recommendations based on results
        if not quality['overall_passed']:
            recommendations['primary_concerns'].append("Model does not meet minimum quality standards")
            recommendations['action_items'].append("Address performance issues before deployment")
        
        return recommendations
    
    def _calculate_combined_scores(self, model_results: Dict, safety_results: Dict) -> Dict[str, float]:
        """Calculate combined performance and safety scores"""
        
        # Extract key metrics
        standard_metrics = model_results.get('standard_metrics', {})
        custom_metrics = model_results.get('custom_metrics', {})
        
        # Calculate combined scores
        performance_score = (
            standard_metrics.get('f1_weighted', 0) * 0.4 +
            standard_metrics.get('accuracy', 0) * 0.3 +
            standard_metrics.get('balanced_accuracy', 0) * 0.3
        )
        
        # Placeholder for safety score calculation
        safety_score = 0.8  # Would calculate from actual safety results
        
        combined_score = (performance_score * 0.6 + safety_score * 0.4)
        
        return {
            'performance_score': performance_score,
            'safety_score': safety_score,
            'combined_score': combined_score
        }
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution summary"""
        
        return {
            'execution_time': self.pipeline_state['duration'],
            'stages_completed': len(self.pipeline_state['stages_completed']),
            'total_stages': 7,
            'success_rate': len(self.pipeline_state['stages_completed']) / 7,
            'errors_encountered': len(self.pipeline_state['errors']),
            'output_location': str(self.output_path)
        }
    
    def _save_pipeline_state(self):
        """Save complete pipeline state to file"""
        import json
        
        state_file = self.output_path / "pipeline_state.json"
        
        # Convert datetime objects to strings
        serializable_state = self.pipeline_state.copy()
        if serializable_state['start_time']:
            serializable_state['start_time'] = serializable_state['start_time'].isoformat()
        if serializable_state['end_time']:
            serializable_state['end_time'] = serializable_state['end_time'].isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(serializable_state, f, indent=2, default=str)
    
    def _create_results_index(self) -> Dict[str, str]:
        """Create an index of all generated results files"""
        
        results_index = {}
        
        # Scan output directory for generated files
        for file_path in self.output_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.output_path)
                results_index[str(relative_path)] = file_path.stat().st_size
        
        # Save index
        with open(self.output_path / "results_index.json", 'w') as f:
            import json
            json.dump(results_index, f, indent=2)
        
        return results_index
    
    def _cleanup_temporary_files(self):
        """Clean up temporary files if configured to do so"""
        
        if not self.config.save_intermediate:
            temp_dir = self.output_path / "results" / "intermediate"
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary intermediate files")
    
    def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline-level errors"""
        
        logger.error(f"Pipeline execution failed: {error}")
        logger.error(traceback.format_exc())
        
        # Save error state
        error_file = self.output_path / "pipeline_error.log"
        with open(error_file, 'w') as f:
            f.write(f"Pipeline Error Report\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error}\n\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline results"""
        
        if not self.pipeline_state['results']:
            return {'status': 'no_results'}
        
        return {
            'status': 'completed' if not self.pipeline_state['errors'] else 'completed_with_errors',
            'duration': self.pipeline_state['duration'],
            'stages_completed': len(self.pipeline_state['stages_completed']),
            'output_directory': str(self.output_path),
            'has_model_results': 'model_evaluation' in self.pipeline_state['results'],
            'has_safety_results': 'safety_evaluation' in self.pipeline_state['results'],
            'has_visualizations': 'visualizations' in self.pipeline_state['results'],
            'has_reports': 'reports' in self.pipeline_state['results']
        }