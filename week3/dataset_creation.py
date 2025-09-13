#!/usr/bin/env python3
"""
Week 3: Custom Evaluation Dataset Creation
Main script for building high-quality, domain-specific evaluation data
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import time

from custom_dataset_builder import CustomDatasetBuilder, DataPoint

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dataset_creation.log')
        ]
    )

def step1_create_collection_framework(output_dir: str):
    """Step 1: Create framework and guidelines for data collection"""
    
    print("="*70)
    print("STEP 1: CREATING DATA COLLECTION FRAMEWORK")
    print("="*70)
    
    builder = CustomDatasetBuilder(output_dir)
    
    print("\n1️⃣ Generating collection prompts...")
    collection_prompts = builder.create_seed_collection_prompts()
    
    print(f"   ✅ Created {len(collection_prompts)} collection prompt categories")
    for prompt in collection_prompts:
        print(f"   📝 {prompt['category']}: {prompt['prompt']}")
    
    print("\n2️⃣ Creating annotation interface...")
    interface_path = builder.generate_annotation_interface()
    print(f"   ✅ Annotation interface created: {interface_path}")
    
    print("\n3️⃣ Creating annotation guidelines...")
    guidelines_path = builder.create_annotation_guidelines_doc()
    print(f"   ✅ Guidelines document created: {guidelines_path}")
    
    print(f"\n✅ STEP 1 COMPLETE")
    print(f"📁 Framework files created in: {output_dir}")
    print(f"🔗 Open annotation interface: {interface_path}")
    print(f"📖 Review guidelines: {guidelines_path}")
    
    return builder

def step2_manual_annotation_demo(builder: CustomDatasetBuilder):
    """Step 2: Demonstrate manual annotation process with sample data"""
    
    print("\n" + "="*70)
    print("STEP 2: MANUAL ANNOTATION DEMONSTRATION")
    print("="*70)
    
    # Create sample tweets for demonstration
    sample_tweets = [
        {
            'id': 'demo_001',
            'text': 'Fire department training exercise at Main St. Looks so realistic with all the smoke!',
            'notes': 'This is a challenging case because it mentions fire and emergency response but is actually a training exercise, not a real emergency.'
        },
        {
            'id': 'demo_002', 
            'text': 'Abuela still missing since the flooding started yesterday. La familia está muy preocupada. Please help us find her.',
            'notes': 'Mixed language content with clear emergency need - missing person case with cultural context.'
        },
        {
            'id': 'demo_003',
            'text': 'Great, another "emergency" downtown 🙄 Third time this week they block traffic for "urgent" construction',
            'notes': 'Sarcastic tone that could confuse models - not a real emergency despite using emergency language.'
        },
        {
            'id': 'demo_004',
            'text': 'Building collapse at 5th and Main - multiple people trapped, need rescue crews immediately!',
            'notes': 'Clear emergency case with high urgency and specific actionable information.'
        },
        {
            'id': 'demo_005',
            'text': 'Is this earthquake serious or just normal shaking? Can\'t tell if I should evacuate or not',
            'notes': 'Ambiguous case - unclear severity and appropriate response needed.'
        }
    ]
    
    print(f"\n📝 Demonstrating annotation process with {len(sample_tweets)} sample tweets...")
    
    # Create demonstration annotations
    demo_annotations = []
    
    annotations_data = [
        {
            'id': 'demo_001',
            'true_label': 'other_relevant_information',
            'confidence': 4,
            'category': 'ambiguous', 
            'urgency_level': 'low',
            'complexity': 'complex',
            'reasoning': 'Training exercise, not real emergency'
        },
        {
            'id': 'demo_002',
            'true_label': 'missing_or_found_people',
            'confidence': 5,
            'category': 'multilingual',
            'urgency_level': 'urgent', 
            'complexity': 'moderate',
            'reasoning': 'Clear missing person case despite mixed language'
        },
        {
            'id': 'demo_003',
            'true_label': 'not_humanitarian',
            'confidence': 4,
            'category': 'sarcastic',
            'urgency_level': 'low',
            'complexity': 'complex',
            'reasoning': 'Sarcasm about non-emergency situation'
        },
        {
            'id': 'demo_004', 
            'true_label': 'requests_or_urgent_needs',
            'confidence': 5,
            'category': 'simple',
            'urgency_level': 'immediate',
            'complexity': 'simple',
            'reasoning': 'Clear emergency with immediate action needed'
        },
        {
            'id': 'demo_005',
            'true_label': 'requests_or_urgent_needs',
            'confidence': 2,
            'category': 'ambiguous',
            'urgency_level': 'moderate',
            'complexity': 'complex',
            'reasoning': 'Uncertain severity, person needs guidance'
        }
    ]
    
    for tweet, annotation in zip(sample_tweets, annotations_data):
        demo_point = DataPoint(
            id=annotation['id'],
            text=tweet['text'],
            true_label=annotation['true_label'],
            confidence=annotation['confidence'],
            category=annotation['category'],
            source='demo_manual',
            disaster_type='mixed',
            urgency_level=annotation['urgency_level'],
            complexity=annotation['complexity'],
            annotator_notes=f"{tweet['notes']} | {annotation['reasoning']}",
            timestamp='2024-01-15T10:00:00Z'
        )
        demo_annotations.append(demo_point)
        
        print(f"\n📋 {annotation['id']}: {annotation['category'].upper()} case")
        print(f"   Text: \"{tweet['text'][:60]}...\"")
        print(f"   Label: {annotation['true_label']}")
        print(f"   Confidence: {annotation['confidence']}/5")
        print(f"   Urgency: {annotation['urgency_level']}")
        print(f"   Complexity: {annotation['complexity']}")
        print(f"   Reasoning: {annotation['reasoning']}")
    
    # Add to builder
    builder.collected_data.extend(demo_annotations)
    
    print(f"\n✅ STEP 2 COMPLETE")
    print(f"📊 {len(demo_annotations)} demonstration annotations created")
    print(f"🎯 Coverage: {len(set(dp.category for dp in demo_annotations))} different edge case types")
    
    return demo_annotations

def step3_data_augmentation(builder: CustomDatasetBuilder, target_size: int = 100):
    """Step 3: Generate augmented examples to expand dataset"""
    
    print("\n" + "="*70)
    print("STEP 3: DATA AUGMENTATION")
    print("="*70)
    
    if not builder.collected_data:
        print("❌ No collected data available for augmentation")
        return []
    
    print(f"\n🔄 Generating {target_size} augmented examples from {len(builder.collected_data)} base examples...")
    
    # Show augmentation strategies
    print(f"\n📋 Augmentation Strategies:")
    strategies = [
        "🔤 Paraphrasing: Semantic-preserving text variations",
        "📊 Severity Scaling: Varied urgency and severity levels", 
        "📍 Location Substitution: Different geographic contexts",
        "⏰ Temporal Variation: Different disaster phase contexts",
        "👁️ Perspective Shifts: First/third person variations",
        "📝 Detail Variation: Different information density levels"
    ]
    
    for strategy in strategies:
        print(f"   {strategy}")
    
    # Generate augmented data
    start_time = time.time()
    augmented_data = builder.generate_augmented_examples(
        builder.collected_data, 
        target_count=target_size
    )
    end_time = time.time()
    
    print(f"\n✅ Generated {len(augmented_data)} augmented examples in {end_time - start_time:.1f} seconds")
    
    # Show examples of each augmentation type
    print(f"\n📋 AUGMENTATION EXAMPLES:")
    
    augmentation_types = {}
    for dp in augmented_data:
        aug_type = dp.source.split('_')[0] if '_' in dp.source else dp.source
        if aug_type not in augmentation_types:
            augmentation_types[aug_type] = []
        augmentation_types[aug_type].append(dp)
    
    for aug_type, examples in augmentation_types.items():
        if examples:
            example = examples[0]
            print(f"\n🔧 {aug_type.upper()} Example:")
            print(f"   Original concept: {example.true_label}")
            print(f"   Generated text: \"{example.text[:80]}...\"")
            print(f"   Category: {example.category}")
            print(f"   Count: {len(examples)} examples")
    
    print(f"\n✅ STEP 3 COMPLETE")
    print(f"📈 Dataset expanded from {len(builder.collected_data)} to {len(builder.collected_data) + len(augmented_data)} examples")
    
    return augmented_data

def step4_quality_validation(builder: CustomDatasetBuilder):
    """Step 4: Validate dataset quality and generate recommendations"""
    
    print("\n" + "="*70)
    print("STEP 4: DATASET QUALITY VALIDATION")
    print("="*70)
    
    print(f"\n🔍 Analyzing dataset quality...")
    validation_results = builder.validate_dataset_quality()
    
    if 'error' in validation_results:
        print(f"❌ Validation error: {validation_results['error']}")
        return validation_results
    
    # Display key metrics
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total Samples: {validation_results['total_samples']}")
    print(f"   Collected: {validation_results['collected_samples']}")
    print(f"   Augmented: {validation_results['augmented_samples']}")
    
    # Label distribution
    print(f"\n🏷️ LABEL DISTRIBUTION:")
    label_dist = validation_results['distribution_analysis']['label_distribution']
    total = validation_results['total_samples']
    
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"   {label.replace('_', ' ').title():30} : {count:3d} ({percentage:5.1f}%)")
    
    # Edge case categories
    print(f"\n🎯 EDGE CASE CATEGORIES:")
    category_dist = validation_results['distribution_analysis']['category_distribution']
    
    for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"   {category.title():15} : {count:3d} ({percentage:5.1f}%)")
    
    # Quality metrics
    quality = validation_results['quality_metrics']
    annotation_quality = validation_results['annotation_quality']
    
    print(f"\n⚖️ QUALITY METRICS:")
    print(f"   Label Imbalance Ratio: {quality.get('label_imbalance_ratio', 'N/A'):.1f}")
    print(f"   Average Confidence: {annotation_quality.get('average_confidence', 'N/A'):.2f}/5.0")
    print(f"   Low Confidence Count: {annotation_quality.get('low_confidence_count', 'N/A')}")
    
    # Recommendations
    recommendations = validation_results.get('recommendations', [])
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print(f"\n✅ No major quality issues detected!")
    
    print(f"\n✅ STEP 4 COMPLETE")
    print(f"📋 Quality validation finished")
    
    return validation_results

def step5_export_and_documentation(builder: CustomDatasetBuilder):
    """Step 5: Export dataset and generate documentation"""
    
    print("\n" + "="*70)
    print("STEP 5: EXPORT AND DOCUMENTATION")
    print("="*70)
    
    print(f"\n📤 Exporting dataset in multiple formats...")
    export_paths = builder.export_dataset(format_type='all')
    
    print(f"\n📁 EXPORTED FILES:")
    for format_type, path in export_paths.items():
        print(f"   {format_type.upper():15} : {path}")
    
    print(f"\n📝 Generating comprehensive documentation...")
    report_content = builder.generate_dataset_report()
    
    print(f"\n✅ STEP 5 COMPLETE")
    print(f"📊 Dataset exported in {len(export_paths)} formats")
    print(f"📄 Documentation generated")
    
    return export_paths

def demonstrate_annotation_workflow():
    """Interactive demonstration of the annotation workflow"""
    
    print("\n" + "="*70)
    print("ANNOTATION WORKFLOW DEMONSTRATION")
    print("="*70)
    
    print("""
This demonstrates the complete annotation workflow:

1. 📝 COLLECTION PHASE:
   - Review collection prompts for different edge case types
   - Gather 50-100 diverse, challenging examples
   - Focus on cases that would challenge existing models

2. 🏷️ ANNOTATION PHASE:
   - Use the provided HTML interface for consistent annotation
   - Follow detailed guidelines for quality control
   - Include confidence scores and reasoning notes

3. 🔄 AUGMENTATION PHASE:
   - Generate variations while preserving semantic meaning
   - Create robustness test cases
   - Expand dataset size systematically

4. ✅ VALIDATION PHASE:
   - Check for quality issues and biases
   - Validate annotation consistency
   - Generate improvement recommendations

5. 📤 EXPORT PHASE:
   - Export in multiple formats for different use cases
   - Create comprehensive documentation
   - Prepare for evaluation integration
    """)

def run_week3_complete_workflow(output_dir: str):
    """Run the complete Week 3 workflow"""
    
    print("🚀 WEEK 3: CUSTOM EVALUATION DATASET CREATION")
    print("=" * 70)
    print("Goal: Create high-quality, domain-specific evaluation data")
    print("Focus: Edge cases, real-world complexity, annotation quality")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Framework Creation
    builder = step1_create_collection_framework(output_dir)
    
    # Step 2: Manual Annotation Demo
    demo_annotations = step2_manual_annotation_demo(builder)
    
    # Step 3: Data Augmentation
    augmented_data = step3_data_augmentation(builder, target_size=80)
    
    # Step 4: Quality Validation
    validation_results = step4_quality_validation(builder)
    
    # Step 5: Export and Documentation
    export_paths = step5_export_and_documentation(builder)
    
    # Final Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n" + "="*70)
    print("WEEK 3 COMPLETE - CUSTOM DATASET READY!")
    print("="*70)
    
    print(f"\n📊 FINAL DATASET STATISTICS:")
    if validation_results and 'total_samples' in validation_results:
        print(f"   📝 Total Examples: {validation_results['total_samples']}")
        print(f"   🎯 Manual Examples: {validation_results['collected_samples']}") 
        print(f"   🔄 Augmented Examples: {validation_results['augmented_samples']}")
        print(f"   ⏱️ Creation Time: {total_time/60:.1f} minutes")
    
    print(f"\n🎯 WHAT WE PROVED:")
    print(f"   ✅ Created targeted edge case evaluation data")
    print(f"   ✅ Demonstrated systematic annotation methodology")
    print(f"   ✅ Generated robust augmentation pipeline")
    print(f"   ✅ Validated dataset quality systematically")
    print(f"   ✅ Produced multiple export formats for integration")
    
    print(f"\n📁 KEY DELIVERABLES:")
    print(f"   📋 Annotation Guidelines: {builder.output_dir}/annotation_guidelines.md")
    print(f"   🔗 Annotation Interface: {builder.output_dir}/annotation_interface.html")
    print(f"   📊 Dataset Report: {builder.output_dir}/dataset_report.md")
    print(f"   📄 Main Dataset: {builder.output_dir}/custom_evaluation_dataset.json")
    print(f"   📈 Split Versions: train/val/test JSON files")
    
    print(f"\n🔬 NEXT STEPS FOR INTEGRATION:")
    print(f"   1. Load custom dataset alongside HumAID for comparison")
    print(f"   2. Evaluate models on edge cases vs standard cases")
    print(f"   3. Analyze performance gaps on complex vs simple examples")
    print(f"   4. Use confidence scores to calibrate model uncertainty")
    print(f"   5. Generate insights about model limitations and strengths")
    
    return builder, validation_results, export_paths

def main():
    """Main function for Week 3 dataset creation"""
    
    parser = argparse.ArgumentParser(
        description="Week 3: Custom Evaluation Dataset Creation for Disaster Response",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python week3_dataset_creation.py                    # Run complete workflow
  python week3_dataset_creation.py --demo             # Show annotation workflow demo
  python week3_dataset_creation.py --framework-only   # Create framework only
  python week3_dataset_creation.py --output-dir ./my_dataset  # Custom output directory
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['complete', 'framework-only', 'demo'],
        default='complete',
        help='Run mode: complete workflow, framework creation only, or demo'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./custom_evaluation_dataset',
        help='Output directory for dataset files'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=100,
        help='Target size for augmented dataset'
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
    
    logger.info("Starting Week 3: Custom Evaluation Dataset Creation")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'demo':
            demonstrate_annotation_workflow()
        elif args.mode == 'framework-only':
            step1_create_collection_framework(args.output_dir)
        else:  # complete
            run_week3_complete_workflow(args.output_dir)
            
    except KeyboardInterrupt:
        logger.info("Dataset creation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during dataset creation: {e}", exc_info=True)
        return 1
    
    logger.info("Week 3 dataset creation completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)