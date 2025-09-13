"""
Custom Evaluation Dataset Builder for Disaster Response Tweet Classification
Week 3: Building high-quality, domain-specific evaluation data
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import random
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Represents a single data point in our custom evaluation dataset"""
    id: str
    text: str
    true_label: str
    confidence: float  # Annotator confidence (1-5 scale)
    category: str  # Edge case category (ambiguous, temporal, multilingual, etc.)
    source: str  # Data source (collected, generated, augmented)
    disaster_type: str  # Earthquake, flood, fire, etc.
    urgency_level: str  # immediate, urgent, moderate, low
    complexity: str  # simple, moderate, complex
    annotator_notes: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class AnnotationGuidelines:
    """Guidelines for consistent annotation"""
    
    urgency_levels = {
        'immediate': 'Life-threatening situation requiring instant response (minutes)',
        'urgent': 'Serious situation requiring rapid response (hours)', 
        'moderate': 'Important but not immediately life-threatening (days)',
        'low': 'Informational or support-related (no time pressure)'
    }
    
    complexity_levels = {
        'simple': 'Clear, unambiguous classification',
        'moderate': 'Some context needed but generally clear',
        'complex': 'Requires careful analysis, potential for misclassification'
    }
    
    edge_case_categories = {
        'ambiguous': 'Could reasonably fit multiple categories',
        'temporal': 'Disaster phase context critical for classification',
        'multilingual': 'Contains multiple languages or translation issues',
        'sarcastic': 'Sarcasm or irony that could confuse models',
        'incomplete': 'Missing critical context or information',
        'conflicting': 'Contains contradictory information',
        'cultural': 'Requires cultural context to classify correctly',
        'technical': 'Contains specialized terminology or technical details'
    }

class CustomDatasetBuilder:
    """Main class for building custom evaluation dataset"""
    
    def __init__(self, output_dir: str = "./custom_evaluation_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.guidelines = AnnotationGuidelines()
        self.collected_data: List[DataPoint] = []
        self.augmented_data: List[DataPoint] = []
        
        # Standard HumAID labels for consistency
        self.standard_labels = [
            'injured_or_dead_people',
            'requests_or_urgent_needs', 
            'rescue_volunteering_or_donation_effort',
            'missing_or_found_people',
            'displaced_people_and_evacuations',
            'infrastructure_and_utility_damage',
            'caution_and_advice',
            'sympathy_and_support',
            'other_relevant_information',
            'not_humanitarian',
            'dont_know_cant_judge'
        ]
    
    def create_seed_collection_prompts(self) -> List[Dict[str, str]]:
        """Create prompts to guide manual data collection"""
        
        collection_prompts = [
            {
                'category': 'ambiguous_emergency',
                'prompt': 'Find tweets where it\'s unclear if this is a real emergency or not',
                'examples': [
                    'Fire department training exercise looks so real!',
                    'Earthquake drill at school - felt like the real thing',
                    'Is this flooding serious or just normal rain?'
                ]
            },
            {
                'category': 'temporal_context',
                'prompt': 'Find tweets where disaster phase (before/during/after) affects classification',
                'examples': [
                    'Preparing for hurricane - stocking up on supplies',
                    'Hurricane passed - assessing damage now',  
                    'Annual hurricane anniversary - remembering victims'
                ]
            },
            {
                'category': 'multilingual_mixed',
                'prompt': 'Find tweets mixing languages or with translation issues',
                'examples': [
                    'Terremoto! Earthquake! People trapped necesitamos ayuda',
                    'Help - ayuda - person hurt in incendio fire',
                    'SOS救命 need rescue from flood水災'
                ]
            },
            {
                'category': 'sarcasm_irony',
                'prompt': 'Find tweets using sarcasm that could confuse emergency classification',
                'examples': [
                    'Great, another "emergency" downtown 🙄',
                    'Oh wonderful, tornado warning during lunch break',
                    'Just what we needed - another earthquake today'
                ]
            },
            {
                'category': 'incomplete_context',
                'prompt': 'Find tweets missing critical information for proper classification',
                'examples': [
                    'They said evacuation but didn\'t say where or why',
                    'Someone got hurt but unclear how serious',
                    'Emergency services responded to incident'
                ]
            },
            {
                'category': 'conflicting_information',
                'prompt': 'Find tweets with contradictory or conflicting signals',
                'examples': [
                    'Major earthquake! Just kidding, everything\'s fine',
                    'Building collapsed but no injuries reported... wait, people trapped',
                    'Evacuation ordered then cancelled then ordered again'
                ]
            },
            {
                'category': 'cultural_context',
                'prompt': 'Find tweets requiring cultural knowledge for proper classification',
                'examples': [
                    'Community potluck for hurricane victims at mosque',
                    'Abuela missing since the flood - family searching',
                    'Traditional healing circle for trauma survivors'
                ]
            },
            {
                'category': 'technical_specialized',
                'prompt': 'Find tweets with technical language that might confuse models',
                'examples': [
                    'Magnitude 6.2 seismic event with P-wave arrival',
                    'HAZMAT team responding to chemical spill incident',
                    'CAT 3 hurricane with 120mph sustained winds tracking NE'
                ]
            }
        ]
        
        return collection_prompts
    
    def generate_annotation_interface(self) -> str:
        """Generate HTML interface for manual annotation"""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Disaster Response Tweet Annotation Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .tweet-box { border: 1px solid #ddd; padding: 15px; margin: 10px 0; background: #f9f9f9; }
        .tweet-text { font-size: 16px; font-weight: bold; margin-bottom: 10px; }
        .annotation-section { margin: 10px 0; }
        .label-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .urgency-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .complexity-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .category-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        button { padding: 8px 12px; margin: 2px; cursor: pointer; }
        button.selected { background: #007bff; color: white; }
        textarea { width: 100%; height: 60px; }
        .guidelines { background: #e7f3ff; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
        .save-btn { background: #28a745; color: white; padding: 10px 20px; font-size: 16px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Disaster Response Tweet Annotation</h1>
        
        <div class="guidelines">
            <h3>Annotation Guidelines</h3>
            <p><strong>Goal:</strong> Create high-quality ground truth for edge cases and complex scenarios</p>
            <ul>
                <li><strong>Confidence:</strong> Rate your certainty in the classification (1=very uncertain, 5=very certain)</li>
                <li><strong>Urgency:</strong> How quickly does this require response?</li>
                <li><strong>Complexity:</strong> How difficult is this to classify correctly?</li>
                <li><strong>Category:</strong> What type of edge case is this?</li>
            </ul>
        </div>
        
        <div id="annotation-interface">
            <!-- Annotation interface will be populated by JavaScript -->
        </div>
        
        <script>
            // JavaScript for interactive annotation interface
            let currentTweet = 0;
            let annotations = [];
            
            // Sample tweets for annotation (replace with your collected data)
            const tweets = [
                {
                    id: "sample_001",
                    text: "Fire department training exercise at Main St. Looks so real - smoke everywhere!",
                    source: "collected"
                },
                {
                    id: "sample_002", 
                    text: "Abuela still missing since the flooding started. Familia muy preocupada. Please help find her.",
                    source: "collected"
                }
                // Add more tweets here
            ];
            
            function loadTweet(index) {
                if (index >= tweets.length) {
                    showResults();
                    return;
                }
                
                const tweet = tweets[index];
                const interface = document.getElementById('annotation-interface');
                
                interface.innerHTML = `
                    <div class="tweet-box">
                        <div class="tweet-text">"${tweet.text}"</div>
                        <small>ID: ${tweet.id} | Source: ${tweet.source}</small>
                        
                        <div class="annotation-section">
                            <h4>Primary Label:</h4>
                            <div class="label-grid">
                                ${generateLabelButtons()}
                            </div>
                        </div>
                        
                        <div class="annotation-section">
                            <h4>Confidence (1-5):</h4>
                            <div class="confidence-grid">
                                ${generateConfidenceButtons()}
                            </div>
                        </div>
                        
                        <div class="annotation-section">
                            <h4>Urgency Level:</h4>
                            <div class="urgency-grid">
                                ${generateUrgencyButtons()}
                            </div>
                        </div>
                        
                        <div class="annotation-section">
                            <h4>Complexity:</h4>
                            <div class="complexity-grid">
                                ${generateComplexityButtons()}
                            </div>
                        </div>
                        
                        <div class="annotation-section">
                            <h4>Edge Case Category:</h4>
                            <div class="category-grid">
                                ${generateCategoryButtons()}
                            </div>
                        </div>
                        
                        <div class="annotation-section">
                            <h4>Notes:</h4>
                            <textarea id="notes" placeholder="Add any additional observations or reasoning..."></textarea>
                        </div>
                        
                        <div class="annotation-section">
                            <button class="save-btn" onclick="saveAnnotation()">Save & Next Tweet</button>
                            <span style="margin-left: 20px;">Tweet ${index + 1} of ${tweets.length}</span>
                        </div>
                    </div>
                `;
            }
            
            function generateLabelButtons() {
                const labels = [
                    'injured_or_dead_people', 'requests_or_urgent_needs', 'rescue_volunteering_or_donation_effort',
                    'missing_or_found_people', 'displaced_people_and_evacuations', 'infrastructure_and_utility_damage',
                    'caution_and_advice', 'sympathy_and_support', 'other_relevant_information',
                    'not_humanitarian', 'dont_know_cant_judge'
                ];
                
                return labels.map(label => 
                    `<button onclick="selectButton(this, 'label')" data-value="${label}">${label.replace(/_/g, ' ')}</button>`
                ).join('');
            }
            
            function generateConfidenceButtons() {
                return [1,2,3,4,5].map(conf => 
                    `<button onclick="selectButton(this, 'confidence')" data-value="${conf}">${conf}</button>`
                ).join('');
            }
            
            function generateUrgencyButtons() {
                const urgency = ['immediate', 'urgent', 'moderate', 'low'];
                return urgency.map(u => 
                    `<button onclick="selectButton(this, 'urgency')" data-value="${u}">${u}</button>`
                ).join('');
            }
            
            function generateComplexityButtons() {
                const complexity = ['simple', 'moderate', 'complex'];
                return complexity.map(c => 
                    `<button onclick="selectButton(this, 'complexity')" data-value="${c}">${c}</button>`
                ).join('');
            }
            
            function generateCategoryButtons() {
                const categories = ['ambiguous', 'temporal', 'multilingual', 'sarcastic', 'incomplete', 'conflicting', 'cultural', 'technical'];
                return categories.map(cat => 
                    `<button onclick="selectButton(this, 'category')" data-value="${cat}">${cat}</button>`
                ).join('');
            }
            
            function selectButton(button, type) {
                // Clear previous selections in this group
                const siblings = button.parentElement.querySelectorAll('button');
                siblings.forEach(b => b.classList.remove('selected'));
                
                // Select current button
                button.classList.add('selected');
            }
            
            function saveAnnotation() {
                const tweet = tweets[currentTweet];
                const annotation = {
                    id: tweet.id,
                    text: tweet.text,
                    true_label: getSelectedValue('label'),
                    confidence: getSelectedValue('confidence'),
                    urgency_level: getSelectedValue('urgency'),
                    complexity: getSelectedValue('complexity'),
                    category: getSelectedValue('category'),
                    annotator_notes: document.getElementById('notes').value,
                    timestamp: new Date().toISOString(),
                    source: tweet.source
                };
                
                annotations.push(annotation);
                currentTweet++;
                loadTweet(currentTweet);
            }
            
            function getSelectedValue(type) {
                const selected = document.querySelector(`button.selected[data-value]`);
                return selected ? selected.dataset.value : null;
            }
            
            function showResults() {
                document.getElementById('annotation-interface').innerHTML = `
                    <h2>Annotation Complete!</h2>
                    <p>You have annotated ${annotations.length} tweets.</p>
                    <textarea style="width: 100%; height: 300px;" readonly>${JSON.stringify(annotations, null, 2)}</textarea>
                    <p>Copy the JSON above and save it as annotations.json</p>
                `;
            }
            
            // Start annotation interface
            loadTweet(0);
        </script>
    </div>
</body>
</html>
        """
        
        # Save HTML interface
        interface_path = self.output_dir / "annotation_interface.html"
        with open(interface_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"Annotation interface saved to: {interface_path}")
        return str(interface_path)
    
    def load_manual_annotations(self, annotations_file: str) -> List[DataPoint]:
        """Load manually annotated data from JSON file"""
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            raw_annotations = json.load(f)
        
        data_points = []
        for ann in raw_annotations:
            data_point = DataPoint(
                id=ann['id'],
                text=ann['text'],
                true_label=ann['true_label'],
                confidence=float(ann.get('confidence', 3)),
                category=ann.get('category', 'unknown'),
                source=ann.get('source', 'manual'),
                disaster_type=ann.get('disaster_type', 'unknown'),
                urgency_level=ann.get('urgency_level', 'moderate'),
                complexity=ann.get('complexity', 'moderate'),
                annotator_notes=ann.get('annotator_notes', ''),
                timestamp=ann.get('timestamp', datetime.now().isoformat())
            )
            data_points.append(data_point)
        
        self.collected_data.extend(data_points)
        logger.info(f"Loaded {len(data_points)} manually annotated data points")
        
        return data_points
    
    def generate_augmented_examples(self, base_examples: List[DataPoint], 
                                  target_count: int = 200) -> List[DataPoint]:
        """Generate augmented examples using various techniques"""
        
        logger.info(f"Generating {target_count} augmented examples from {len(base_examples)} base examples")
        
        augmented = []
        
        # Augmentation strategies
        strategies = [
            self._paraphrase_augmentation,
            self._severity_variation,
            self._location_substitution,
            self._temporal_variation,
            self._perspective_shift,
            self._detail_variation
        ]
        
        examples_per_strategy = target_count // len(strategies)
        
        for strategy in strategies:
            strategy_examples = strategy(base_examples, examples_per_strategy)
            augmented.extend(strategy_examples)
        
        # Fill remaining with random strategy
        remaining = target_count - len(augmented)
        if remaining > 0:
            random_strategy = random.choice(strategies)
            extra_examples = random_strategy(base_examples, remaining)
            augmented.extend(extra_examples)
        
        self.augmented_data.extend(augmented)
        logger.info(f"Generated {len(augmented)} augmented examples")
        
        return augmented
    
    def validate_dataset_quality(self) -> Dict[str, Any]:
        """Validate the quality of the custom dataset"""
        
        all_data = self.collected_data + self.augmented_data
        
        if not all_data:
            return {'error': 'No data to validate'}
        
        validation_results = {
            'total_samples': len(all_data),
            'collected_samples': len(self.collected_data),
            'augmented_samples': len(self.augmented_data),
            'quality_metrics': {},
            'distribution_analysis': {},
            'annotation_quality': {},
            'recommendations': []
        }
        
        # Label distribution analysis
        label_counts = Counter(dp.true_label for dp in all_data)
        validation_results['distribution_analysis']['label_distribution'] = dict(label_counts)
        
        # Check for label imbalance
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        validation_results['quality_metrics']['label_imbalance_ratio'] = imbalance_ratio
        
        # Category distribution
        category_counts = Counter(dp.category for dp in all_data)
        validation_results['distribution_analysis']['category_distribution'] = dict(category_counts)
        
        # Complexity distribution
        complexity_counts = Counter(dp.complexity for dp in all_data)
        validation_results['distribution_analysis']['complexity_distribution'] = dict(complexity_counts)
        
        # Urgency distribution
        urgency_counts = Counter(dp.urgency_level for dp in all_data)
        validation_results['distribution_analysis']['urgency_distribution'] = dict(urgency_counts)
        
        # Annotation quality metrics
        confidences = [dp.confidence for dp in all_data if dp.confidence > 0]
        if confidences:
            validation_results['annotation_quality']['average_confidence'] = np.mean(confidences)
            validation_results['annotation_quality']['min_confidence'] = min(confidences)
            validation_results['annotation_quality']['low_confidence_count'] = sum(1 for c in confidences if c < 3)
        
        # Source distribution
        source_counts = Counter(dp.source for dp in all_data)
        validation_results['distribution_analysis']['source_distribution'] = dict(source_counts)
        
        # Generate recommendations
        recommendations = []
        
        if imbalance_ratio > 5:
            recommendations.append(f"High label imbalance detected (ratio: {imbalance_ratio:.1f}). Consider collecting more examples for underrepresented classes.")
        
        if validation_results['annotation_quality'].get('low_confidence_count', 0) > len(all_data) * 0.2:
            recommendations.append("More than 20% of annotations have low confidence (<3). Consider re-annotating unclear examples.")
        
        if len(set(dp.category for dp in all_data)) < 4:
            recommendations.append("Limited edge case category coverage. Consider adding more diverse challenging examples.")
        
        complex_examples = sum(1 for dp in all_data if dp.complexity == 'complex')
        if complex_examples < len(all_data) * 0.2:
            recommendations.append("Less than 20% complex examples. Consider adding more challenging cases for robust evaluation.")
        
        validation_results['recommendations'] = recommendations
        
        logger.info(f"Dataset validation complete. Total samples: {len(all_data)}")
        return validation_results
    
    def export_dataset(self, format_type: str = 'all') -> Dict[str, str]:
        """Export the custom dataset in various formats"""
        
        all_data = self.collected_data + self.augmented_data
        
        if not all_data:
            logger.error("No data to export")
            return {}
        
        export_paths = {}
        
        # JSON format (detailed)
        if format_type in ['json', 'all']:
            json_data = [dp.to_dict() for dp in all_data]
            json_path = self.output_dir / "custom_evaluation_dataset.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            export_paths['json'] = str(json_path)
            logger.info(f"Dataset exported to JSON: {json_path}")
        
        # CSV format (simplified)
        if format_type in ['csv', 'all']:
            df_data = []
            for dp in all_data:
                df_data.append({
                    'id': dp.id,
                    'text': dp.text,
                    'label': dp.true_label,
                    'confidence': dp.confidence,
                    'category': dp.category,
                    'urgency': dp.urgency_level,
                    'complexity': dp.complexity,
                    'source': dp.source
                })
            
            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / "custom_evaluation_dataset.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            export_paths['csv'] = str(csv_path)
            logger.info(f"Dataset exported to CSV: {csv_path}")
        
        # HuggingFace datasets format
        if format_type in ['huggingface', 'all']:
            hf_data = {
                'id': [dp.id for dp in all_data],
                'text': [dp.text for dp in all_data],
                'label': [dp.true_label for dp in all_data],
                'confidence': [dp.confidence for dp in all_data],
                'category': [dp.category for dp in all_data],
                'urgency_level': [dp.urgency_level for dp in all_data],
                'complexity': [dp.complexity for dp in all_data],
                'source': [dp.source for dp in all_data]
            }
            
            hf_path = self.output_dir / "custom_dataset_huggingface.json"
            with open(hf_path, 'w', encoding='utf-8') as f:
                json.dump(hf_data, f, indent=2)
            
            export_paths['huggingface'] = str(hf_path)
            logger.info(f"Dataset exported for HuggingFace: {hf_path}")
        
        # Split versions (train/validation/test)
        if format_type in ['splits', 'all']:
            # Stratified split to maintain label distribution
            train_data, val_data, test_data = self._create_stratified_splits(all_data)
            
            for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                split_path = self.output_dir / f"custom_dataset_{split_name}.json"
                split_export = [dp.to_dict() for dp in split_data]
                
                with open(split_path, 'w', encoding='utf-8') as f:
                    json.dump(split_export, f, indent=2, ensure_ascii=False)
                
                export_paths[f'{split_name}_split'] = str(split_path)
                logger.info(f"{split_name} split exported: {len(split_data)} samples")
        
        return export_paths
    
    def _create_stratified_splits(self, data: List[DataPoint], 
                                train_ratio: float = 0.7, 
                                val_ratio: float = 0.15) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint]]:
        """Create stratified splits maintaining label distribution"""
        
        # Group by label
        label_groups = defaultdict(list)
        for dp in data:
            label_groups[dp.true_label].append(dp)
        
        train_data, val_data, test_data = [], [], []
        
        for label, examples in label_groups.items():
            # Shuffle examples
            random.shuffle(examples)
            
            n_examples = len(examples)
            n_train = int(n_examples * train_ratio)
            n_val = int(n_examples * val_ratio)
            
            train_data.extend(examples[:n_train])
            val_data.extend(examples[n_train:n_train + n_val])
            test_data.extend(examples[n_train + n_val:])
        
        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data
    
    def generate_dataset_report(self) -> str:
        """Generate comprehensive dataset documentation"""
        
        validation_results = self.validate_dataset_quality()
        
        report_lines = [
            "# Custom Evaluation Dataset Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Overview",
            "",
            f"- **Total Samples**: {validation_results['total_samples']}",
            f"- **Manually Collected**: {validation_results['collected_samples']}",
            f"- **Augmented Examples**: {validation_results['augmented_samples']}",
            "",
            "## Purpose and Goals",
            "",
            "This custom evaluation dataset was created to address specific gaps in existing disaster response classification datasets:",
            "",
            "1. **Edge Case Coverage**: Focus on ambiguous, complex scenarios that challenge model robustness",
            "2. **Real-world Authenticity**: Authentic social media language and communication patterns",
            "3. **Fine-grained Analysis**: Detailed annotation including confidence, urgency, and complexity",
            "4. **Bias Detection**: Diverse examples to test for demographic and contextual biases",
            "",
            "## Data Collection Methodology",
            "",
            "### Manual Collection Strategy",
            "- **Targeted Collection**: Focused on specific edge case categories",
            "- **Expert Annotation**: Detailed manual annotation with confidence scores",
            "- **Quality Control**: Multiple validation passes for consistency",
            "",
            "### Augmentation Techniques",
            "- **Paraphrasing**: Semantic-preserving text variations",
            "- **Severity Scaling**: Varied urgency and severity levels",
            "- **Location Substitution**: Different geographic contexts",
            "- **Temporal Variation**: Different disaster phase contexts",
            "- **Perspective Shifts**: First/third person variations",
            "- **Detail Variation**: Different information density levels",
            "",
            "## Dataset Statistics",
            ""
        ]
        
        # Label distribution
        label_dist = validation_results['distribution_analysis']['label_distribution']
        report_lines.extend([
            "### Label Distribution",
            "",
            "| Label | Count | Percentage |",
            "|-------|-------|------------|"
        ])
        
        total_samples = validation_results['total_samples']
        for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_samples) * 100
            report_lines.append(f"| {label.replace('_', ' ').title()} | {count} | {percentage:.1f}% |")
        
        # Edge case categories
        if 'category_distribution' in validation_results['distribution_analysis']:
            category_dist = validation_results['distribution_analysis']['category_distribution']
            report_lines.extend([
                "",
                "### Edge Case Categories",
                "",
                "| Category | Count | Description |",
                "|----------|-------|-------------|"
            ])
            
            category_descriptions = self.guidelines.edge_case_categories
            for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
                description = category_descriptions.get(category, 'Unknown category')
                report_lines.append(f"| {category.title()} | {count} | {description} |")
        
        # Complexity and urgency distributions
        complexity_dist = validation_results['distribution_analysis'].get('complexity_distribution', {})
        urgency_dist = validation_results['distribution_analysis'].get('urgency_distribution', {})
        
        report_lines.extend([
            "",
            "### Complexity Distribution",
            ""
        ])
        
        for complexity, count in complexity_dist.items():
            percentage = (count / total_samples) * 100
            report_lines.append(f"- **{complexity.title()}**: {count} samples ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "### Urgency Distribution", 
            ""
        ])
        
        for urgency, count in urgency_dist.items():
            percentage = (count / total_samples) * 100
            report_lines.append(f"- **{urgency.title()}**: {count} samples ({percentage:.1f}%)")
        
        # Quality metrics
        quality_metrics = validation_results['quality_metrics']
        annotation_quality = validation_results['annotation_quality']
        
        report_lines.extend([
            "",
            "## Quality Assessment",
            "",
            f"- **Label Imbalance Ratio**: {quality_metrics.get('label_imbalance_ratio', 'N/A'):.1f}",
            f"- **Average Annotation Confidence**: {annotation_quality.get('average_confidence', 'N/A'):.2f}/5.0",
            f"- **Low Confidence Samples**: {annotation_quality.get('low_confidence_count', 'N/A')}",
            ""
        ])
        
        # Recommendations
        recommendations = validation_results.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "## Recommendations for Improvement",
                ""
            ])
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            
            report_lines.append("")
        
        # Sample examples
        if self.collected_data:
            report_lines.extend([
                "## Sample Data Points",
                ""
            ])
            
            # Show diverse examples
            sample_examples = []
            categories_shown = set()
            
            for dp in self.collected_data:
                if dp.category not in categories_shown and len(sample_examples) < 5:
                    sample_examples.append(dp)
                    categories_shown.add(dp.category)
            
            for i, example in enumerate(sample_examples, 1):
                report_lines.extend([
                    f"### Example {i}: {example.category.title()} Case",
                    f"**Text**: \"{example.text}\"",
                    f"**Label**: {example.true_label}",
                    f"**Urgency**: {example.urgency_level}",
                    f"**Complexity**: {example.complexity}",
                    f"**Confidence**: {example.confidence}/5.0",
                    f"**Notes**: {example.annotator_notes}",
                    ""
                ])
        
        # Usage guidelines
        report_lines.extend([
            "## Usage Guidelines",
            "",
            "### Evaluation Protocol",
            "1. **Baseline Comparison**: Compare against standard HumAID test set",
            "2. **Edge Case Analysis**: Focus on performance across different categories",
            "3. **Confidence Correlation**: Analyze model confidence vs annotation confidence",
            "4. **Error Analysis**: Examine failures by complexity and urgency levels",
            "",
            "### Interpretation Notes",
            "- **High Confidence Samples**: Expected to have minimal annotation disagreement",
            "- **Complex Cases**: Lower performance expected, focus on relative comparison",
            "- **Edge Cases**: Test specific model capabilities and limitations",
            "- **Augmented Data**: Use for robustness testing, not primary evaluation",
            "",
            "## Dataset Limitations",
            "",
            "1. **Size**: Relatively small dataset focused on specific challenging cases",
            "2. **Domain Scope**: Limited to disaster response tweet classification",
            "3. **Annotation Bias**: Single annotator perspective in initial version",
            "4. **Temporal Scope**: Limited temporal diversity in disaster types",
            "5. **Language**: Primarily English with limited multilingual examples",
            "",
            "## Future Improvements",
            "",
            "1. **Multi-annotator Agreement**: Add inter-annotator reliability measures",
            "2. **Expanded Categories**: Include more disaster types and contexts",
            "3. **Multilingual Coverage**: Expand beyond English examples",
            "4. **Temporal Diversity**: Include more disaster timeline variations",
            "5. **Expert Validation**: Domain expert review of challenging cases"
        ])
        
        report_content = '\n'.join(report_lines)
        
        # Save report
        report_path = self.output_dir / "dataset_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Dataset report saved to: {report_path}")
        return report_content
    
    def create_annotation_guidelines_doc(self) -> str:
        """Create detailed annotation guidelines document"""
        
        guidelines_content = """
# Disaster Response Tweet Annotation Guidelines

## Overview
This guide provides detailed instructions for creating high-quality annotations for disaster response tweet classification evaluation.

## Annotation Principles

### 1. Consistency
- Use the same criteria across all examples
- When uncertain, note the specific source of uncertainty
- Be consistent with urgency and complexity assessments

### 2. Context Awareness
- Consider the disaster phase (before, during, after)
- Account for cultural and linguistic context
- Recognize that emergency response needs vary by location and resources

### 3. Real-world Relevance
- Prioritize annotations that reflect actual emergency response needs
- Consider the perspective of emergency responders and affected communities
- Balance ideal classifications with practical response capabilities

## Label Definitions

### Primary Categories

**injured_or_dead_people**
- Direct reports of casualties
- Medical emergencies requiring immediate attention
- Fatality reports or serious injury descriptions

**requests_or_urgent_needs**
- Explicit calls for help or assistance
- Urgent resource needs (medical, rescue, supplies)
- Time-sensitive requests for emergency services

**rescue_volunteering_or_donation_effort**
- Offers of help or volunteer services
- Donation drives or resource collection
- Coordination of rescue or relief efforts

**missing_or_found_people**
- Reports of missing persons
- Found person notifications
- Search and rescue coordination

**displaced_people_and_evacuations**
- Evacuation notices or reports
- Shelter needs and displaced population information
- Temporary housing or relocation information

**infrastructure_and_utility_damage**
- Reports of structural damage
- Utility outages or infrastructure failures
- Transportation disruption information

**caution_and_advice**
- Safety warnings and advisories
- Preparedness information and guidance
- Risk awareness and prevention advice

**sympathy_and_support**
- Emotional support and condolences
- Prayers, thoughts, and moral support
- Community solidarity expressions

**other_relevant_information**
- General disaster-related information
- Updates and situational reports
- Relevant but uncategorized information

**not_humanitarian**
- Non-disaster related content
- Commercial or promotional content
- Irrelevant or off-topic information

## Urgency Level Assessment

### Immediate (Response needed in minutes)
- Life-threatening situations in progress
- Active rescue operations needed
- Medical emergencies requiring paramedics

### Urgent (Response needed in hours)
- Serious situations requiring prompt attention
- Resource needs that affect safety
- Time-sensitive coordination requirements

### Moderate (Response needed in days)
- Important but not immediately life-threatening
- Resource planning and logistics
- Infrastructure assessment and repair

### Low (No time pressure)
- Informational content
- Long-term planning and recovery
- General support and awareness

## Complexity Assessment

### Simple
- Clear, unambiguous meaning
- Single clear classification
- Standard emergency language

### Moderate
- Some interpretation required
- Multiple possible classifications with clear preference
- Context helps clarify meaning

### Complex
- Significant ambiguity or multiple valid interpretations
- Requires domain expertise to classify correctly
- Missing critical context for confident classification

## Edge Case Categories

### Ambiguous
- Could reasonably fit multiple categories
- Unclear intent or context
- Borderline cases requiring judgment calls

### Temporal
- Classification depends on disaster timeline
- Before/during/after context affects meaning
- Phase-specific language and priorities

### Multilingual
- Mixed language content
- Translation ambiguities
- Cultural communication patterns

### Sarcastic
- Ironic or sarcastic tone
- Could be misinterpreted as sincere
- Requires understanding of social context

### Incomplete
- Missing critical information
- Partial messages or unclear references
- Requires additional context for classification

### Conflicting
- Contains contradictory information
- Mixed signals about urgency or needs
- Internal inconsistencies in content

### Cultural
- Requires cultural knowledge for proper classification
- Community-specific terminology or practices
- Local context affects interpretation

### Technical
- Specialized terminology or technical language
- Professional emergency response language
- Domain-specific acronyms or procedures

## Confidence Scoring (1-5 Scale)

**5 - Very Confident**
- Completely clear and unambiguous
- Perfect fit for chosen category
- Would expect 100% annotator agreement

**4 - Confident**
- Clear classification with minor uncertainty
- Good fit for chosen category
- Would expect >90% annotator agreement

**3 - Moderately Confident**
- Reasonable classification with some uncertainty
- Acceptable fit but could see alternative interpretations
- Would expect ~80% annotator agreement

**2 - Low Confidence**
- Significant uncertainty about classification
- Could fit multiple categories
- Would expect ~60% annotator agreement

**1 - Very Low Confidence**
- High uncertainty, best guess classification
- Multiple equally valid interpretations
- Would expect <50% annotator agreement

## Common Annotation Challenges

### Challenge 1: Distinguishing Real vs Hypothetical
**Example**: "What would happen if earthquake hit downtown?"
**Guidance**: Mark as hypothetical unless there's clear indication of actual emergency

### Challenge 2: Temporal Context
**Example**: "Preparing for hurricane season"
**Guidance**: Consider if this is pre-disaster preparation vs. post-disaster recovery

### Challenge 3: Severity Assessment
**Example**: "Minor flooding on Main Street"
**Guidance**: Base urgency on described impact, not just the word "minor"

### Challenge 4: Cultural References
**Example**: "Community kitchen serving storm victims"
**Guidance**: Understand cultural practices around disaster response

### Challenge 5: Technical Language
**Example**: "CAT 3 hurricane tracking NE at 15mph"
**Guidance**: Classify based on emergency relevance, not technical complexity

## Quality Control Checklist

Before finalizing annotations:
- [ ] Is the primary label the best fit among all options?
- [ ] Does the urgency level match the described situation?
- [ ] Is the complexity assessment consistent with ambiguity level?
- [ ] Are notes helpful for understanding classification reasoning?
- [ ] Is the confidence score honest and well-calibrated?

## Examples by Category

[Include 2-3 examples for each category with detailed reasoning]

## Frequently Asked Questions

**Q: What if a tweet fits multiple categories equally well?**
A: Choose the category that would be most actionable for emergency responders, note the ambiguity, and mark confidence as low.

**Q: How do I handle tweets with poor grammar or spelling?**
A: Focus on the intended meaning rather than linguistic quality. Consider if language barriers might affect emergency response.

**Q: What about tweets that mention disasters but aren't requesting help?**
A: Consider the information value for emergency coordination and community awareness.

**Q: How do I assess urgency for unfamiliar disaster types?**
A: Focus on described human impact and time sensitivity rather than disaster classification.
        """
        
        guidelines_path = self.output_dir / "annotation_guidelines.md"
        with open(guidelines_path, 'w', encoding='utf-8') as f:
            f.write(guidelines_content)
        
        logger.info(f"Annotation guidelines saved to: {guidelines_path}")
        return str(guidelines_path)
    
    def _paraphrase_augmentation(self, examples: List[DataPoint], count: int) -> List[DataPoint]:
        """Generate paraphrased versions of examples"""
        
        augmented = []
        paraphrase_templates = [
            # Formality variations
            ("URGENT: {content}", "Emergency: {content}"),
            ("HELP: {content}", "Assistance needed: {content}"),
            ("SOS: {content}", "Urgent request: {content}"),
            
            # Structure variations
            ("{content} - please help!", "Please help - {content}"),
            ("{content}!!!", "{content}."),
            ("Breaking: {content}", "Update: {content}"),
            
            # Intensity variations
            ("desperate", "urgent"),
            ("catastrophic", "severe"),
            ("devastating", "serious")
        ]
        
        for i in range(count):
            base_example = random.choice(examples)
            
            # Apply paraphrasing
            new_text = base_example.text
            
            # Apply random template transformation
            if random.random() < 0.5:
                template_from, template_to = random.choice(paraphrase_templates)
                new_text = new_text.replace(template_from.split('{')[0], template_to.split('{')[0])
            
            # Create augmented example
            augmented_example = DataPoint(
                id=f"{base_example.id}_para_{i}",
                text=new_text,
                true_label=base_example.true_label,
                confidence=max(1, base_example.confidence - 0.5),  # Lower confidence for augmented
                category=base_example.category,
                source="paraphrase_augmented", 
                disaster_type=base_example.disaster_type,
                urgency_level=base_example.urgency_level,
                complexity=base_example.complexity,
                annotator_notes=f"Paraphrased from {base_example.id}",
                timestamp=datetime.now().isoformat()
            )
            
            augmented.append(augmented_example)
        
        return augmented
    
    def _severity_variation(self, examples: List[DataPoint], count: int) -> List[DataPoint]:
        """Generate examples with varied severity levels"""
        
        augmented = []
        severity_transformations = {
            # Increase severity
            'increase': {
                'injured': 'critically injured',
                'hurt': 'severely hurt', 
                'damaged': 'destroyed',
                'affected': 'devastated',
                'some': 'many',
                'a few': 'numerous'
            },
            # Decrease severity  
            'decrease': {
                'critically injured': 'injured',
                'severely hurt': 'hurt',
                'destroyed': 'damaged',
                'devastated': 'affected',
                'many': 'some',
                'numerous': 'a few'
            }
        }
        
        for i in range(count):
            base_example = random.choice(examples)
            transformation_type = random.choice(['increase', 'decrease'])
            
            new_text = base_example.text
            transformations = severity_transformations[transformation_type]
            
            for old_word, new_word in transformations.items():
                new_text = new_text.replace(old_word, new_word)
            
            # Adjust urgency based on severity change
            urgency_mapping = {
                'immediate': ['immediate', 'urgent', 'moderate', 'low'],
                'urgent': ['urgent', 'moderate', 'low'],
                'moderate': ['moderate', 'low'],
                'low': ['low']
            }
            
            current_urgency_idx = urgency_mapping['immediate'].index(base_example.urgency_level)
            
            if transformation_type == 'increase' and current_urgency_idx > 0:
                new_urgency = urgency_mapping['immediate'][current_urgency_idx - 1]
            elif transformation_type == 'decrease' and current_urgency_idx < 3:
                new_urgency = urgency_mapping['immediate'][current_urgency_idx + 1]
            else:
                new_urgency = base_example.urgency_level
            
            augmented_example = DataPoint(
                id=f"{base_example.id}_sev_{transformation_type}_{i}",
                text=new_text,
                true_label=base_example.true_label,
                confidence=base_example.confidence - 0.3,
                category=base_example.category,
                source=f"severity_{transformation_type}",
                disaster_type=base_example.disaster_type,
                urgency_level=new_urgency,
                complexity=base_example.complexity,
                annotator_notes=f"Severity {transformation_type} from {base_example.id}",
                timestamp=datetime.now().isoformat()
            )
            
            augmented.append(augmented_example)
        
        return augmented
    
    def _location_substitution(self, examples: List[DataPoint], count: int) -> List[DataPoint]:
        """Generate examples with different location contexts"""
        
        augmented = []
        location_substitutions = {
            # Urban locations
            'downtown': ['city center', 'metropolitan area', 'urban district'],
            'apartment': ['condo', 'high-rise', 'residential building'],
            'street': ['avenue', 'boulevard', 'road'],
            
            # Rural locations  
            'farm': ['ranch', 'countryside', 'rural area'],
            'village': ['small town', 'community', 'hamlet'],
            
            # Generic substitutions
            'building': ['structure', 'facility', 'complex'],
            'area': ['region', 'zone', 'vicinity'],
            'neighborhood': ['district', 'community', 'locality']
        }
        
        for i in range(count):
            base_example = random.choice(examples)
            new_text = base_example.text
            
            # Apply random location substitution
            for original_loc, alternatives in location_substitutions.items():
                if original_loc in new_text.lower():
                    new_location = random.choice(alternatives)
                    new_text = re.sub(original_loc, new_location, new_text, flags=re.IGNORECASE)
                    break
            
            augmented_example = DataPoint(
                id=f"{base_example.id}_loc_{i}",
                text=new_text,
                true_label=base_example.true_label,
                confidence=base_example.confidence,
                category=base_example.category,
                source="location_substituted",
                disaster_type=base_example.disaster_type,
                urgency_level=base_example.urgency_level,
                complexity=base_example.complexity,
                annotator_notes=f"Location substituted from {base_example.id}",
                timestamp=datetime.now().isoformat()
            )
            
            augmented.append(augmented_example)
        
        return augmented
    
    def _temporal_variation(self, examples: List[DataPoint], count: int) -> List[DataPoint]:
        """Generate examples with different temporal contexts"""
        
        augmented = []
        temporal_transformations = {
            'past': ['happened yesterday', 'occurred last night', 'took place earlier'],
            'present': ['happening now', 'currently ongoing', 'right now'],
            'future': ['expected tomorrow', 'predicted for tonight', 'forecast for later'],
            'duration': ['has been going on for hours', 'started this morning', 'began at dawn']
        }
        
        for i in range(count):
            base_example = random.choice(examples)
            
            # Add temporal context
            temporal_type = random.choice(list(temporal_transformations.keys()))
            temporal_phrase = random.choice(temporal_transformations[temporal_type])
            
            # Insert temporal information
            new_text = f"{base_example.text} - {temporal_phrase}"
            
            augmented_example = DataPoint(
                id=f"{base_example.id}_temp_{temporal_type}_{i}",
                text=new_text,
                true_label=base_example.true_label,
                confidence=base_example.confidence - 0.2,
                category='temporal',
                source=f"temporal_{temporal_type}",
                disaster_type=base_example.disaster_type,
                urgency_level=base_example.urgency_level,
                complexity='moderate',  # Temporal context adds complexity
                annotator_notes=f"Added {temporal_type} temporal context to {base_example.id}",
                timestamp=datetime.now().isoformat()
            )
            
            augmented.append(augmented_example)
        
        return augmented
    
    def _perspective_shift(self, examples: List[DataPoint], count: int) -> List[DataPoint]:
        """Generate examples from different perspectives"""
        
        augmented = []
        perspective_shifts = [
            # First person to third person
            ('I am', 'Someone is'),
            ('my family', 'a family'),
            ('we need', 'they need'),
            ('our house', 'a house'),
            
            # Third person to first person
            ('someone is', 'I am'),
            ('people are', 'we are'),
            ('a person', 'I'),
            ('families', 'my family')
        ]
        
        for i in range(count):
            base_example = random.choice(examples)
            new_text = base_example.text
            
            # Apply perspective shift
            shift_from, shift_to = random.choice(perspective_shifts)
            new_text = re.sub(shift_from, shift_to, new_text, flags=re.IGNORECASE)
            
            augmented_example = DataPoint(
                id=f"{base_example.id}_persp_{i}",
                text=new_text,
                true_label=base_example.true_label,
                confidence=base_example.confidence,
                category=base_example.category,
                source="perspective_shifted",
                disaster_type=base_example.disaster_type,
                urgency_level=base_example.urgency_level,
                complexity=base_example.complexity,
                annotator_notes=f"Perspective shifted from {base_example.id}",
                timestamp=datetime.now().isoformat()
            )
            
            augmented.append(augmented_example)
        
        return augmented
    
    def _detail_variation(self, examples: List[DataPoint], count: int) -> List[DataPoint]:
        """Generate examples with varied detail levels"""
        
        augmented = []
        
        detail_additions = [
            'emergency services on scene',
            'multiple casualties reported', 
            'evacuation in progress',
            'roads blocked',
            'power lines down',
            'water supply affected',
            'Red Cross responding',
            'volunteers needed'
        ]
        
        for i in range(count):
            base_example = random.choice(examples)
            
            # Add or remove details
            if random.random() < 0.7:  # 70% chance to add details
                additional_detail = random.choice(detail_additions)
                new_text = f"{base_example.text} - {additional_detail}"
                detail_type = "added"
            else:  # 30% chance to remove details (simplify)
                # Remove adjectives and adverbs to simplify
                new_text = re.sub(r'\b(very|extremely|severely|critically|completely)\s+', '', base_example.text)
                detail_type = "removed"
            
            augmented_example = DataPoint(
                id=f"{base_example.id}_detail_{detail_type}_{i}",
                text=new_text,
                true_label=base_example.true_label,
                confidence=base_example.confidence - 0.1,
                category=base_example.category,
                source=f"detail_{detail_type}",
                disaster_type=base_example.disaster_type,
                urgency_level=base_example.urgency_level,
                complexity=base_example.complexity,
                annotator_notes=f"Detail {detail_type} from {base_example.id}",
                timestamp=datetime.now().isoformat()
            )
            
            augmented.append(augmented_example)
        