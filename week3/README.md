# Week 3: Building a Custom Evaluation Dataset

## 🎯 Overview

Week 3 focuses on creating a high-quality, domain-specific evaluation dataset that addresses gaps in existing datasets and tests model capabilities on edge cases and real-world complexity.

## 📋 Goals and Objectives

### Primary Goal
Create a **50-100 example** custom evaluation dataset that reveals model limitations not captured by standard benchmarks.

### What We Want to Prove
1. **Edge case handling**: How do models perform on ambiguous or complex scenarios?
2. **Real-world robustness**: Performance on authentic social media language vs academic datasets
3. **Fine-grained classification**: Distinguishing between similar emergency types
4. **Annotation quality impact**: How annotation confidence correlates with model performance

### Why This Matters
- **Nobody reads through eval datasets** - but we will, and we'll find the interesting cases
- **Famous datasets have errors** - even MNIST has labeling mistakes, we'll do better
- **Standard metrics miss critical failures** - edge cases reveal true model capabilities

---

## 🔄 Complete Workflow

### Phase 1: Framework Creation 
```bash
# Create annotation framework and guidelines
python week3_dataset_creation.py --mode framework-only
```

**Deliverables:**
- Annotation interface (HTML)
- Detailed annotation guidelines
- Collection prompts for edge cases

### Phase 2: Manual Data Collection (2-3 hours)
```bash
# Use the generated annotation interface
open custom_evaluation_dataset/annotation_interface.html
```

**Process:**
1. **Collect 50-100 examples** targeting specific edge cases
2. **Annotate systematically** using provided interface
3. **Include confidence scores** and detailed reasoning
4. **Focus on challenging cases** that would trip up models

### Phase 3: Data Augmentation 
```bash
# Generate augmented examples
python week3_dataset_creation.py --target-size 100
```

**Techniques:**
- **Paraphrasing**: Semantic-preserving variations
- **Severity scaling**: Different urgency levels
- **Location substitution**: Urban/rural contexts
- **Temporal variation**: Disaster phase contexts
- **Perspective shifts**: First/third person
- **Detail variation**: Information density changes

### Phase 4: Quality Validation 
Automated quality checks and recommendations for improvement.

### Phase 5: Export and Documentation 
Multiple formats ready for model evaluation integration.

---

## 🎯 Edge Case Categories We Target

### 1. **Ambiguous Cases**
```
Example: "Fire department training exercise looks so real!"
Challenge: Real emergency vs. training/simulation
Why Important: Tests context understanding
```

### 2. **Temporal Context**
```
Example: "Preparing for hurricane season - stocking supplies"
Challenge: Before/during/after disaster phases
Why Important: Tests temporal reasoning
```

### 3. **Multilingual/Mixed Language**
```
Example: "Terremoto! Earthquake! Necesitamos ayuda please help!"
Challenge: Code-switching and translation
Why Important: Tests language robustness
```

### 4. **Sarcasm/Irony**
```
Example: "Great, another 'emergency' downtown 🙄"
Challenge: Non-literal language use
Why Important: Tests pragmatic understanding
```

### 5. **Incomplete Information**
```
Example: "Emergency services responding to incident downtown"
Challenge: Missing critical details
Why Important: Tests inference capabilities
```

### 6. **Conflicting Signals**
```
Example: "Building collapsed! Just kidding, false alarm"
Challenge: Contradictory information
Why Important: Tests contradiction handling
```

### 7. **Cultural Context**
```
Example: "Community kitchen serving hurricane victims at mosque"
Challenge: Cultural practices and contexts
Why Important: Tests cultural awareness
```

### 8. **Technical/Specialized Language**
```
Example: "CAT 3 hurricane with 120mph sustained winds tracking NE"
Challenge: Domain-specific terminology
Why Important: Tests technical comprehension
```

---

## 📊 Annotation Framework

### Primary Labels (Standard HumAID)
- `injured_or_dead_people`
- `requests_or_urgent_needs` 
- `rescue_volunteering_or_donation_effort`
- `missing_or_found_people`
- `displaced_people_and_evacuations`
- `infrastructure_and_utility_damage`
- `caution_and_advice`
- `sympathy_and_support`
- `other_relevant_information`
- `not_humanitarian`
- `dont_know_cant_judge`

### Additional Annotation Dimensions

#### Confidence Scale (1-5)
- **5**: Very confident, would expect 100% agreement
- **4**: Confident, would expect >90% agreement  
- **3**: Moderately confident, ~80% agreement
- **2**: Low confidence, ~60% agreement
- **1**: Very uncertain, <50% agreement

#### Urgency Levels
- **Immediate**: Response needed in minutes (life-threatening)
- **Urgent**: Response needed in hours (serious situation)
- **Moderate**: Response needed in days (important but not critical)
- **Low**: No time pressure (informational)

#### Complexity Assessment
- **Simple**: Clear, unambiguous classification
- **Moderate**: Some interpretation required but generally clear
- **Complex**: Significant ambiguity, requires domain expertise

---

## 🔬 Data Augmentation Strategies

### 1. Paraphrasing Augmentation
```python
Original: "URGENT: Person trapped under debris"
Paraphrase: "Emergency: Individual stuck beneath rubble"
Purpose: Test semantic robustness
```

### 2. Severity Scaling
```python
Original: "Person injured in collapse"
Increase: "Person critically injured in collapse" 
Decrease: "Person slightly hurt in collapse"
Purpose: Test severity calibration
```

### 3. Location Substitution
```python
Original: "Emergency in downtown area"
Rural: "Emergency in rural village"
Urban: "Emergency in metropolitan district"
Purpose: Test geographic bias
```

### 4. Temporal Variation
```python
Original: "Building collapsed"
Past: "Building collapsed yesterday"
Present: "Building collapsing right now"
Future: "Building expected to collapse"
Purpose: Test temporal understanding
```

### 5. Perspective Shifts
```python
Original: "I am trapped in building"
Third Person: "Someone is trapped in building"
Collective: "We are trapped in building"
Purpose: Test perspective handling
```

### 6. Detail Variation
```python
Original: "Person hurt in accident"
More Detail: "Person hurt in accident - emergency services on scene"
Less Detail: "Accident downtown"
Purpose: Test information density handling
```

---

## 📈 Quality Validation Framework

### Automated Quality Checks

#### Label Distribution Analysis
- Check for severe class imbalance (>5:1 ratio)
- Ensure coverage of all major categories
- Identify underrepresented edge cases

#### Annotation Quality Metrics
- Average confidence scores across examples
- Proportion of low-confidence annotations (<3)
- Consistency within edge case categories

#### Dataset Completeness
- Coverage of all defined edge case types
- Balanced complexity distribution
- Appropriate urgency level spread

### Quality Thresholds
- **Label Imbalance**: <5:1 ratio between most/least common
- **Low Confidence**: <20% of examples with confidence <3
- **Edge Case Coverage**: At least 4 different categories represented
- **Complex Examples**: At least 20% marked as complex

---

## 🛠️ Implementation Details

### Files Created
1. **`custom_dataset_builder.py`** - Core framework for dataset creation
2. **`week3_dataset_creation.py`** - Main execution script
3. **Generated Files:**
   - `annotation_interface.html` - Interactive annotation tool
   - `annotation_guidelines.md` - Detailed annotation instructions
   - `custom_evaluation_dataset.json` - Main dataset file
   - `dataset_report.md` - Comprehensive documentation

### Usage Commands
```bash
# Complete workflow
python week3_dataset_creation.py

# Create framework only
python week3_dataset_creation.py --mode framework-only

# Show demonstration
python week3_dataset_creation.py --mode demo

# Custom output directory
python week3_dataset_creation.py --output-dir ./my_dataset

# Adjust target size
python week3_dataset_creation.py --target-size 150
```

### Integration with Existing Pipeline
```python
# Load custom dataset alongside HumAID
from datasets import load_dataset
import json

# Load standard dataset
humaid = load_dataset("QCRI/HumAID-all")

# Load custom dataset
with open("custom_evaluation_dataset.json", "r") as f:
    custom_data = json.load(f)

# Compare performance
standard_results = evaluate_model(model, humaid["test"])
custom_results = evaluate_model(model, custom_data)

# Analyze edge case performance
edge_case_results = analyze_by_category(custom_results)
```

---

## 📊 Expected Outcomes

### Dataset Characteristics
- **Size**: 50-100 manually annotated + 100-200 augmented examples
- **Quality**: High-confidence annotations with detailed reasoning
- **Coverage**: All major edge case categories represented
- **Complexity**: 20-30% complex cases requiring domain expertise

### Insights We'll Gain
1. **Model Limitations**: Specific failure modes not caught by standard evaluation
2. **Edge Case Performance**: How models handle ambiguous or complex scenarios
3. **Robustness Gaps**: Sensitivity to paraphrasing, context, and complexity
4. **Annotation Quality Impact**: Correlation between human confidence and model performance

### Research Value
- **Novel Evaluation Methodology**: Systematic approach to edge case evaluation
- **Domain-Specific Insights**: Disaster response specific challenges and solutions
- **Reproducible Framework**: Methodology applicable to other domains
- **Quality-Aware Evaluation**: Incorporating annotation confidence into model assessment

---

## 🎓 Learning Outcomes

### Technical Skills
- **Dataset Design**: Creating targeted evaluation data
- **Annotation Methodology**: Systematic, high-quality labeling
- **Quality Validation**: Automated quality assessment
- **Data Augmentation**: Systematic variation generation

### Domain Knowledge
- **Disaster Response**: Understanding emergency communication patterns
- **Edge Case Analysis**: Identifying and categorizing challenging scenarios
- **Evaluation Design**: Creating evaluation that reveals model limitations
- **Real-World AI**: Bridging academic evaluation and practical deployment

### Research Methods
- **Problem Definition**: Identifying gaps in existing evaluation
- **Systematic Collection**: Methodical approach to data gathering
- **Quality Control**: Ensuring annotation consistency and reliability
- **Documentation**: Comprehensive dataset documentation and reporting

---

## 🔮 Future Extensions

### Immediate Next Steps
1. **Collect Real Data**: Move from demo examples to actual collection
2. **Multi-Annotator Agreement**: Add inter-annotator reliability measures
3. **Model Evaluation**: Use dataset to evaluate BERT and LLaMA
4. **Error Analysis**: Deep dive into model failures on edge cases

### Advanced Extensions
1. **Active Learning**: Use model uncertainty to guide data collection
2. **Adversarial Examples**: Systematically generated failure cases
3. **Multi-Domain**: Extend methodology to other disaster types
4. **Multilingual**: Expand beyond English examples
5. **Temporal Evolution**: Track model performance over time

### Research Directions
1. **Evaluation Methodology**: Best practices for edge case evaluation
2. **Annotation Theory**: Optimal annotation strategies for AI evaluation
3. **Model Robustness**: Understanding and improving edge case performance
4. **Domain Adaptation**: Applying insights to other critical AI applications

---

## 💡 Key Insights and Best Practices

### Annotation Best Practices
1. **Start Small**: Begin with clear examples, gradually add complexity
2. **Document Reasoning**: Always explain annotation decisions
3. **Calibrate Confidence**: Be honest about uncertainty
4. **Iterate and Refine**: Review and improve annotation guidelines

### Dataset Design Principles
1. **Purpose-Driven**: Each example should test specific model capabilities
2. **Real-World Grounded**: Authentic language and scenarios
3. **Systematically Diverse**: Cover all important edge case types
4. **Quality Over Quantity**: Better to have fewer high-quality examples

### Evaluation Philosophy
1. **Find the Failures**: Focus on where models break down
2. **Understand the Why**: Analyze why certain cases are challenging
3. **Connect to Practice**: Link evaluation insights to real-world deployment
4. **Iterative Improvement**: Use insights to improve both models and evaluation

This Week 3 framework provides a systematic approach to creating high-quality evaluation data that reveals model limitations and guides improvement efforts. The focus on edge cases and annotation quality ensures that the resulting dataset provides valuable insights not available from standard benchmarks.