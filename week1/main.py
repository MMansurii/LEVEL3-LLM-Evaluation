#!/usr/bin/env python3
"""
LLM Evaluation Framework - Main Entry Point
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from config.settings import Settings
from src.data import DatasetDownloader, DatasetExplorer, DatasetAnalyzer
from src.visualization import Visualizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationPipeline:
    """Main evaluation pipeline"""
    
    def __init__(self, dataset_name: str = None):
        """Initialize pipeline"""
        self.dataset_name = dataset_name or Settings.DEFAULT_DATASET
        Settings.create_directories()
        
        self.downloader = DatasetDownloader(self.dataset_name)
        self.dataset = None
        self.explorer = None
        self.analyzer = None
        self.visualizer = Visualizer()
    
    def run(self, download_only: bool = False, analyze_only: bool = False, visualize: bool = True):
        """Run the complete pipeline"""
        logger.info("="*60)
        logger.info("LLM EVALUATION FRAMEWORK - WEEK 1")
        logger.info("="*60)
        
        # Step 1: Download
        if not analyze_only:
            logger.info("\n[Step 1] Downloading dataset...")
            self.dataset = self.downloader.download()
            
            if download_only:
                logger.info("Download complete. Exiting.")
                return
        else:
            logger.info("Loading existing dataset...")
            self.dataset = self.downloader.download()
        
        # Step 2: Explore
        logger.info("\n[Step 2] Exploring dataset...")
        self.explorer = DatasetExplorer(self.dataset)
        exploration_results = self.explorer.explore()
        
        # Step 3: Analyze
        logger.info("\n[Step 3] Analyzing dataset...")
        self.analyzer = DatasetAnalyzer(self.dataset)
        analysis_results = self.analyzer.analyze()
        
        # Step 4: Visualize
        if visualize:
            logger.info("\n[Step 4] Creating visualizations...")
            self.visualizer.create_all_visualizations(self.dataset, analysis_results)
        
        # Step 5: Generate reports
        logger.info("\n[Step 5] Generating reports...")
        self._generate_reports(exploration_results, analysis_results)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE!")
        logger.info(f"Results saved to: {Settings.OUTPUT_DIR}")
        logger.info("="*60)
    
    def _generate_reports(self, exploration_results, analysis_results):
        """Generate all reports"""
        # JSON report
        json_report = {
            "dataset": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "exploration": exploration_results,
            "analysis": analysis_results
        }
        
        json_path = Settings.REPORTS_DIR / "results.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {json_path}")
        
        # Text report
        self._generate_text_report(exploration_results, analysis_results)
    
    def _generate_text_report(self, exploration_results, analysis_results):
        """Generate detailed text report"""
        report_path = Settings.REPORTS_DIR / "exploration_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATASET EXPLORATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for split_name, results in exploration_results.items():
                f.write(f"\n{split_name.upper()} SPLIT\n")
                f.write("-"*40 + "\n")
                f.write(f"Samples: {results['num_samples']}\n")
                f.write(f"Columns: {', '.join(results['columns'])}\n")
                
                if 'text_columns' in results:
                    f.write(f"Text columns: {', '.join(results['text_columns'])}\n")
                if 'label_columns' in results:
                    f.write(f"Label columns: {', '.join(results['label_columns'])}\n")
                
                f.write("\n")
            
            f.write("\nSTATISTICAL ANALYSIS\n")
            f.write("="*40 + "\n")
            
            for split_name, results in analysis_results.items():
                if split_name == "cross_split":
                    continue
                    
                f.write(f"\n{split_name.upper()} Statistics:\n")
                
                for key, value in results.items():
                    if "_stats" in key:
                        col_name = key.replace("_stats", "")
                        f.write(f"\n  {col_name} text statistics:\n")
                        if 'word_count' in value:
                            f.write(f"    Word count: {value['word_count']['mean']:.1f} (mean)\n")
                            f.write(f"    Range: {value['word_count']['min']}-{value['word_count']['max']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
        
        logger.info(f"Text report saved to {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework - Dataset Exploration"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (default: QCRI/HumAID-all)"
    )
    
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download dataset"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing dataset"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization creation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        Settings.LOG_LEVEL = "DEBUG"
    
    pipeline = EvaluationPipeline(dataset_name=args.dataset)
    pipeline.run(
        download_only=args.download_only,
        analyze_only=args.analyze_only,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()
