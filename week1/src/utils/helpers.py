import pandas as pd
from typing import List, Dict, Any


def find_text_columns(df: pd.DataFrame) -> List[str]:
    """Find text columns in DataFrame"""
    text_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
            if isinstance(sample, str) and len(sample) > 20:
                text_columns.append(col)
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['text', 'content', 'message', 'tweet']):
            if col not in text_columns and df[col].dtype == 'object':
                text_columns.append(col)
    
    return text_columns


def find_label_columns(df: pd.DataFrame) -> List[str]:
    """Find label/class columns in DataFrame"""
    label_columns = []
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['label', 'class', 'category', 'target']):
            label_columns.append(col)
        elif df[col].nunique() < 100 and df[col].nunique() > 1:
            if col not in label_columns:
                label_columns.append(col)
    
    return label_columns


def calculate_text_statistics(series: pd.Series) -> Dict[str, Any]:
    """Calculate text statistics"""
    texts = series.fillna("")
    
    word_counts = texts.str.split().str.len()
    char_lengths = texts.str.len()
    
    return {
        "word_count": {
            "mean": float(word_counts.mean()),
            "std": float(word_counts.std()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "median": float(word_counts.median())
        },
        "char_length": {
            "mean": float(char_lengths.mean()),
            "std": float(char_lengths.std()),
            "min": int(char_lengths.min()),
            "max": int(char_lengths.max()),
            "median": float(char_lengths.median())
        },
        "empty_count": int((texts == "").sum()),
        "duplicate_count": int(texts.duplicated().sum()),
        "unique_count": int(texts.nunique())
    }
