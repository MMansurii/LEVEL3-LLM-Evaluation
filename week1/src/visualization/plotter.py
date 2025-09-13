import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter
from wordcloud import WordCloud

from config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_style(Settings.STYLE)


class Visualizer:
    """Create individual visualizations for dataset analysis - each plot as separate image"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Settings.VIZ_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_visualizations(self, dataset, analysis_results: Dict[str, Any]):
        """Create all visualizations as separate image files"""
        logger.info("Creating individual visualization files...")
        
        # Get train split for analysis
        train_df = dataset['train'].to_pandas()
        
        # Find text and label columns
        text_col = self._find_text_column(train_df)
        label_col = self._find_label_column(train_df)
        event_col = self._find_event_column(train_df)
        
        # Prepare splits data
        splits = [s for s in dataset.keys() if s != 'dev']
        sizes = [len(dataset[split]) for split in splits]
        
        # Create each plot as separate image
        self._create_split_sizes_plot(splits, sizes)
        self._create_split_proportions_plot(splits, sizes)
        
        if text_col:
            self._create_word_count_distribution(train_df, text_col)
            self._create_character_length_distribution(train_df, text_col)
            self._create_sentence_count_distribution(train_df, text_col)
            self._create_word_cloud(train_df, text_col)
            self._create_text_quality_indicators(train_df, text_col)
            self._create_text_length_by_split(dataset, splits, text_col)
            self._create_average_word_count_by_split(dataset, splits, text_col)
        
        if label_col:
            self._create_label_distribution(train_df, label_col)
            self._create_class_balance_analysis(train_df, label_col)
        
        if event_col:
            self._create_event_distribution(train_df, event_col)
        
        self._create_dataset_summary_stats(train_df, text_col, label_col)
        self._create_comprehensive_metrics(dataset, analysis_results, text_col)
        
        logger.info(f"All individual visualizations saved to {self.output_dir}")
    
    def _find_text_column(self, df):
        """Find the text column in the dataframe"""
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'tweet', 'message', 'content']):
                return col
        return None
    
    def _find_label_column(self, df):
        """Find the label column in the dataframe"""
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['label', 'class', 'category']):
                return col
        return None
    
    def _find_event_column(self, df):
        """Find the event column in the dataframe"""
        for col in ['event_name', 'crisis', 'disaster_type', 'event']:
            if col in df.columns:
                return col
        return None
    
    def _create_split_sizes_plot(self, splits, sizes):
        """Create dataset split sizes bar chart"""
        plt.figure(figsize=(10, 6))
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(splits)]
        bars = plt.bar(splits, sizes, color=colors, edgecolor='black', linewidth=1.5)
        
        plt.title('Dataset Split Sizes', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Split', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        
        # Add value labels on bars
        for i, (split, size) in enumerate(zip(splits, sizes)):
            plt.text(i, size + max(sizes)*0.02, f'{size:,}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = self.output_dir / "01_dataset_split_sizes.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Split sizes plot saved to {output_path}")
    
    def _create_split_proportions_plot(self, splits, sizes):
        """Create dataset split proportions pie chart"""
        plt.figure(figsize=(8, 8))
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(splits)]
        
        wedges, texts, autotexts = plt.pie(sizes, labels=splits, autopct='%1.1f%%',
                                          colors=colors, startangle=90, 
                                          textprops={'fontsize': 12})
        
        plt.title('Dataset Split Proportions', fontsize=16, fontweight='bold', pad=20)
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "02_dataset_split_proportions.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Split proportions plot saved to {output_path}")
    
    def _create_word_count_distribution(self, train_df, text_col):
        """Create word count distribution histogram"""
        plt.figure(figsize=(12, 6))
        word_counts = train_df[text_col].fillna("").str.split().str.len()
        
        plt.hist(word_counts, bins=50, color='lightgreen', edgecolor='black', 
                alpha=0.7, density=True)
        plt.title(f'Word Count Distribution ({text_col})', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Words', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Add statistics lines
        plt.axvline(word_counts.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {word_counts.mean():.1f}')
        plt.axvline(word_counts.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {word_counts.median():.1f}')
        plt.axvline(word_counts.quantile(0.95), color='orange', linestyle='--', linewidth=2,
                   label=f'95th percentile: {word_counts.quantile(0.95):.1f}')
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "03_word_count_distribution.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Word count distribution plot saved to {output_path}")
    
    def _create_character_length_distribution(self, train_df, text_col):
        """Create character length distribution histogram"""
        plt.figure(figsize=(12, 6))
        char_lengths = train_df[text_col].fillna("").str.len()
        
        plt.hist(char_lengths, bins=50, color='lightcoral', edgecolor='black', 
                alpha=0.7, density=True)
        plt.title('Character Length Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Characters', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Add statistics lines
        plt.axvline(char_lengths.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {char_lengths.mean():.0f}')
        plt.axvline(char_lengths.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {char_lengths.median():.0f}')
        plt.axvline(char_lengths.quantile(0.95), color='orange', linestyle='--', linewidth=2,
                   label=f'95th percentile: {char_lengths.quantile(0.95):.0f}')
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "04_character_length_distribution.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Character length distribution plot saved to {output_path}")
    
    def _create_sentence_count_distribution(self, train_df, text_col):
        """Create sentence count distribution histogram"""
        plt.figure(figsize=(10, 6))
        sentence_counts = train_df[text_col].fillna("").str.count(r'[.!?]+') + 1
        sentence_counts = sentence_counts.clip(upper=20)  # Cap for better visualization
        
        plt.hist(sentence_counts, bins=20, color='orange', edgecolor='black', alpha=0.7)
        plt.title('Sentence Count Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Sentences', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        plt.axvline(sentence_counts.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {sentence_counts.mean():.1f}')
        plt.axvline(sentence_counts.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {sentence_counts.median():.1f}')
        
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.output_dir / "05_sentence_count_distribution.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Sentence count distribution plot saved to {output_path}")
    
    def _create_word_cloud(self, train_df, text_col):
        """Create word cloud visualization"""
        plt.figure(figsize=(12, 8))
        sample_size = min(2000, len(train_df))
        all_text = ' '.join(train_df[text_col].fillna("").sample(sample_size, random_state=42).tolist())
        
        if all_text.strip():
            try:
                wordcloud = WordCloud(width=800, height=600, 
                                     background_color='white',
                                     max_words=200,
                                     colormap='viridis',
                                     relative_scaling=0.5,
                                     random_state=42).generate(all_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud (Sample of 2000 texts)', fontsize=16, fontweight='bold', pad=20)
                
                output_path = self.output_dir / "06_word_cloud.png"
                plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
                plt.close()
                logger.info(f"Word cloud saved to {output_path}")
            except Exception as e:
                logger.warning(f"Word cloud generation failed: {e}")
                plt.close()
    
    def _create_text_quality_indicators(self, train_df, text_col):
        """Create text quality indicators bar chart"""
        plt.figure(figsize=(12, 6))
        texts = train_df[text_col].fillna("")
        
        quality_data = {
            'URLs': (texts.str.contains(r'http[s]?://\S+', regex=True)).mean() * 100,
            'Mentions (@)': (texts.str.contains(r'@\w+', regex=True)).mean() * 100,
            'Hashtags (#)': (texts.str.contains(r'#\w+', regex=True)).mean() * 100,
            'Numbers': (texts.str.contains(r'\d+', regex=True)).mean() * 100,
            'Empty Texts': (texts == "").mean() * 100,
            'Special Chars': (texts.str.contains(r'[^a-zA-Z0-9\s]', regex=True)).mean() * 100
        }
        
        colors = ['coral', 'lightblue', 'lightgreen', 'gold', 'lightgray', 'plum']
        bars = plt.bar(quality_data.keys(), quality_data.values(), 
                      color=colors, edgecolor='black', linewidth=1)
        
        plt.title('Text Quality Indicators (% of samples)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Quality Indicator', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, quality_data.values()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = self.output_dir / "07_text_quality_indicators.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Text quality indicators plot saved to {output_path}")
    
    def _create_text_length_by_split(self, dataset, splits, text_col):
        """Create text length distribution by split (box plot)"""
        plt.figure(figsize=(12, 6))
        split_data = []
        split_names = []
        
        for split in splits[:5]:  # Limit to 5 splits for readability
            try:
                split_df = dataset[split].to_pandas()
                if text_col in split_df.columns:
                    lengths = split_df[text_col].fillna("").str.split().str.len()
                    split_data.append(lengths)
                    split_names.append(split)
            except:
                continue
        
        if split_data:
            bp = plt.boxplot(split_data, labels=split_names, patch_artist=True, 
                           showfliers=True, medianprops={'color': 'red', 'linewidth': 2})
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title('Word Count Distribution by Split', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Split', fontsize=12)
            plt.ylabel('Word Count', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_path = self.output_dir / "08_word_count_by_split.png"
            plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
            plt.close()
            logger.info(f"Word count by split plot saved to {output_path}")
    
    def _create_average_word_count_by_split(self, dataset, splits, text_col):
        """Create average word count by split bar chart"""
        plt.figure(figsize=(10, 6))
        splits_to_compare = []
        mean_words = []
        
        for split_name in splits[:5]:
            try:
                split_df = dataset[split_name].to_pandas()
                if text_col in split_df.columns:
                    mean_word_count = split_df[text_col].fillna("").str.split().str.len().mean()
                    splits_to_compare.append(split_name)
                    mean_words.append(mean_word_count)
            except:
                continue
        
        if splits_to_compare:
            colors = ['lightsteelblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(splits_to_compare)]
            bars = plt.bar(splits_to_compare, mean_words, color=colors, edgecolor='black', linewidth=1.5)
            
            plt.title('Average Word Count by Split', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Split', fontsize=12)
            plt.ylabel('Mean Word Count', fontsize=12)
            
            # Add value labels
            for bar, value in zip(bars, mean_words):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(mean_words)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            output_path = self.output_dir / "09_average_word_count_by_split.png"
            plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
            plt.close()
            logger.info(f"Average word count by split plot saved to {output_path}")
    
    def _create_label_distribution(self, train_df, label_col):
        """Create label distribution bar chart"""
        plt.figure(figsize=(14, 8))
        label_counts = train_df[label_col].value_counts()
        
        if len(label_counts) <= 25:
            bars = plt.bar(range(len(label_counts)), label_counts.values, 
                          color='lightsteelblue', edgecolor='black', linewidth=1)
            plt.xticks(range(len(label_counts)), 
                      [str(x)[:20] for x in label_counts.index], 
                      rotation=45, ha='right')
            plt.title(f'Label Distribution ({label_col})', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, label_counts.values)):
                plt.text(i, value + max(label_counts.values)*0.01, str(value),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            # Too many classes, show top 20
            top_20 = label_counts.head(20)
            bars = plt.bar(range(len(top_20)), top_20.values, 
                          color='lightsteelblue', edgecolor='black', linewidth=1)
            plt.xticks(range(len(top_20)), 
                      [str(x)[:15] for x in top_20.index], 
                      rotation=45, ha='right')
            plt.title(f'Top 20 Labels ({label_col})', fontsize=16, fontweight='bold', pad=20)
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_path = self.output_dir / "10_label_distribution.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Label distribution plot saved to {output_path}")
    
    def _create_class_balance_analysis(self, train_df, label_col):
        """Create class balance analysis horizontal bar chart"""
        plt.figure(figsize=(12, 10))
        label_counts = train_df[label_col].value_counts()
        
        if len(label_counts) <= 20:
            expected_count = len(train_df) / len(label_counts)
            colors = ['green' if count >= expected_count * 0.5 else 'red' 
                     for count in label_counts.values]
            
            bars = plt.barh(range(len(label_counts)), label_counts.values, 
                           color=colors, alpha=0.7, edgecolor='black')
            plt.yticks(range(len(label_counts)), 
                      [str(x)[:30] for x in label_counts.index])
            plt.axvline(expected_count, color='blue', linestyle='--', linewidth=2,
                       label=f'Perfect Balance ({expected_count:.0f})')
            
            plt.title('Class Balance Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Sample Count', fontsize=12)
            plt.ylabel('Class', fontsize=12)
            
            # Add imbalance ratio text
            imbalance_ratio = label_counts.max() / label_counts.min()
            plt.text(0.02, 0.98, f'Imbalance Ratio: {imbalance_ratio:.2f}', 
                    transform=plt.gca().transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
            
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            output_path = self.output_dir / "11_class_balance_analysis.png"
            plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
            plt.close()
            logger.info(f"Class balance analysis plot saved to {output_path}")
    
    def _create_event_distribution(self, train_df, event_col):
        """Create event distribution horizontal bar chart"""
        plt.figure(figsize=(12, 8))
        event_counts = train_df[event_col].value_counts().head(15)
        
        bars = plt.barh(range(len(event_counts)), event_counts.values, color='lightsalmon', edgecolor='black')
        plt.yticks(range(len(event_counts)), 
                  [str(x)[:40] for x in event_counts.index])
        plt.title(f'Top 15 {event_col.replace("_", " ").title()}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(event_col.replace("_", " ").title(), fontsize=12)
        
        # Add value labels
        for i, value in enumerate(event_counts.values):
            plt.text(value + max(event_counts.values)*0.01, i, str(value),
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        output_path = self.output_dir / f"12_{event_col}_distribution.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Event distribution plot saved to {output_path}")
    
    def _create_dataset_summary_stats(self, train_df, text_col, label_col):
        """Create dataset summary statistics table visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate comprehensive metrics
        stats_data = []
        
        # Basic stats
        stats_data.append(['Metric', 'Value'])
        stats_data.append(['Total Samples', f'{len(train_df):,}'])
        
        # if text_col:
        #     texts = train_df[text_col].fillna("")
        #     stats_data.extend([
        #         ['Empty Texts', f'{(texts == "").sum():,}'],
        #         ['Duplicate Texts', f'{texts.duplicated().sum():,}'],
        #         ['Average Words per Text', f'{texts.str.split().str.len().mean():.2f}'],
        #         ['Median Words per Text', f'{texts.str.split().str.len().median():.0f}'],
        #         ['Average Characters per Text', f'{texts.str.len().mean():.0f}'],
        #         ['Max Words in Text', f'{texts.str.split().str.len().max():.0f}'],
        #         ['Min Words in Text', f'{texts.str.split().str.len().min():.0f}'],
        #         ['Texts with URLs', f'{texts.str.contains(r"http[s]?://\S+", regex=True).sum():,}'],
        #         ['Texts with Mentions', f'{texts.str.contains(r"@\w+", regex=True).sum():,}'],
        #         ['Texts with Hashtags', f'{texts.str.contains(r"#\w+", regex=True).sum():,}']
        #     ])
        if text_col:
            texts = train_df[text_col].fillna("")

            urls_count = texts.str.contains(r"http[s]?://\S+", regex=True).sum()
            mentions_count = texts.str.contains(r"@\w+", regex=True).sum()
            hashtags_count = texts.str.contains(r"#\w+", regex=True).sum()

            stats_data.extend([
                ['Empty Texts', f'{(texts == "").sum():,}'],
                ['Duplicate Texts', f'{texts.duplicated().sum():,}'],
                ['Average Words per Text', f'{texts.str.split().str.len().mean():.2f}'],
                ['Median Words per Text', f'{texts.str.split().str.len().median():.0f}'],
                ['Average Characters per Text', f'{texts.str.len().mean():.0f}'],
                ['Max Words in Text', f'{texts.str.split().str.len().max():.0f}'],
                ['Min Words in Text', f'{texts.str.split().str.len().min():.0f}'],
                ['Texts with URLs', f'{urls_count:,}'],
                ['Texts with Mentions', f'{mentions_count:,}'],
                ['Texts with Hashtags', f'{hashtags_count:,}']
            ])


        if label_col:
            label_counts = train_df[label_col].value_counts()
            stats_data.extend([
                ['Number of Classes', f'{len(label_counts):,}'],
                ['Most Common Class', f'{label_counts.index[0]}'],
                ['Most Common Class Count', f'{label_counts.iloc[0]:,}'],
                ['Least Common Class Count', f'{label_counts.iloc[-1]:,}'],
                ['Class Imbalance Ratio', f'{label_counts.iloc[0] / label_counts.iloc[-1]:.2f}']
            ])
        
        # Create table
        table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                        cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(stats_data)):
            for j in range(2):
                if i == 0:  # Header
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                elif i % 2 == 0:  # Even rows
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:  # Odd rows
                    table[(i, j)].set_facecolor('#ffffff')
        
        plt.title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=30)
        plt.tight_layout()
        
        output_path = self.output_dir / "13_dataset_summary_stats.png"
        plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Dataset summary statistics saved to {output_path}")
    
    def _create_comprehensive_metrics(self, dataset, analysis_results, text_col):
        """Create comprehensive metrics comparison across splits"""
        plt.figure(figsize=(14, 8))
        
        # Collect metrics from all splits
        split_metrics = {}
        for split_name in dataset.keys():
            if split_name == 'dev':  # Skip duplicate
                continue
            try:
                split_df = dataset[split_name].to_pandas()
                if text_col and text_col in split_df.columns:
                    texts = split_df[text_col].fillna("")
                    split_metrics[split_name] = {
                        'avg_words': texts.str.split().str.len().mean(),
                        'avg_chars': texts.str.len().mean(),
                        'empty_ratio': (texts == "").mean() * 100,
                        'duplicate_ratio': texts.duplicated().mean() * 100
                    }
            except:
                continue
        
        if split_metrics:
            splits = list(split_metrics.keys())
            metrics = ['avg_words', 'avg_chars', 'empty_ratio', 'duplicate_ratio']
            metric_labels = ['Avg Words', 'Avg Characters', 'Empty %', 'Duplicate %']
            
            x = np.arange(len(splits))
            width = 0.2
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = [split_metrics[split][metric] for split in splits]
                plt.bar(x + i * width, values, width, label=label, color=colors[i], edgecolor='black')
            
            plt.title('Comprehensive Metrics Across Splits', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Split', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.xticks(x + width * 1.5, splits)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            output_path = self.output_dir / "14_comprehensive_metrics.png"
            plt.savefig(output_path, dpi=Settings.DPI, bbox_inches='tight')
            plt.close()
            logger.info(f"Comprehensive metrics plot saved to {output_path}")
    
    def _create_detailed_analysis_plots(self, dataset, analysis_results):
        """This method is kept for compatibility but individual plots are created in create_all_visualizations"""
        pass