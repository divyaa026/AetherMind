"""Prepare and unify depression and suicide datasets for multimodal training.

This script:
- Loads source CSVs from the `datasets/` directory
- Normalizes schema to: cleaned_text (str), label (0/1)
- Generates simple synthetic behavioral features from labels
- Saves a combined CSV: `multimodal_dataset.csv`
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Paths (use forward slashes for cross-platform compatibility)
DATASETS_DIR = Path('datasets')
DEPRESSION_PATH = DATASETS_DIR / 'depression_dataset_reddit_cleaned.csv'
SUICIDE_PATH = DATASETS_DIR / 'Suicide_Detection.csv'


# Simple text cleaning
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ' '.join(text.lower().strip().split())


# Load datasets
depression_df = pd.read_csv(DEPRESSION_PATH)
suicide_df = pd.read_csv(SUICIDE_PATH)


# Normalize depression dataset
# Expected columns: clean_text, is_depression (0/1)
if 'clean_text' not in depression_df.columns or 'is_depression' not in depression_df.columns:
    raise ValueError("depression_dataset_reddit_cleaned.csv must contain 'clean_text' and 'is_depression' columns")

depression_df['cleaned_text'] = depression_df['clean_text'].astype(str).apply(clean_text)
# Map labels robustly to integers
depression_df['label'] = depression_df['is_depression'].map({"0": 0, "1": 1, 0: 0, 1: 1}).astype(int)
depression_small = depression_df[['cleaned_text', 'label']].copy()


# Normalize suicide dataset
# Expected columns: text, class (values: 'suicide' | 'non-suicide')
# Drop any unnamed index column if present
suicide_df = suicide_df.loc[:, ~suicide_df.columns.str.contains('^Unnamed')]  # drop stray index cols

if 'text' not in suicide_df.columns or 'class' not in suicide_df.columns:
    raise ValueError("Suicide_Detection.csv must contain 'text' and 'class' columns")

suicide_df['cleaned_text'] = suicide_df['text'].astype(str).apply(clean_text)
suicide_df['label'] = suicide_df['class'].map({'suicide': 1, 'non-suicide': 0})
suicide_df = suicide_df.dropna(subset=['label'])  # remove rows with unknown labels
suicide_df['label'] = suicide_df['label'].astype(int)
suicide_small = suicide_df[['cleaned_text', 'label']].copy()


# Combine aligned datasets
full_df = pd.concat([depression_small, suicide_small], ignore_index=True)


def generate_behavioral_features(label):
    if label == 1:  # Crisis state
        return {
            # Overlap with non-crisis: 4-6 hrs
            'screen_time': np.random.normal(5.5, 1.5),
            
            # Overlapping app switches
            'app_switches': np.random.poisson(30),
            
            # Partial night activity overlap
            'night_activity': np.random.uniform(0.4, 0.8)
        }
    else:  # Non-crisis
        return {
            'screen_time': np.random.normal(4.0, 1.2),
            'app_switches': np.random.poisson(35),
            'night_activity': np.random.uniform(0.2, 0.6)
        }

full_df['behavioral_features'] = full_df['label'].apply(generate_behavioral_features)

# Expand behavioral features into columns and finalize
behavior_df = pd.json_normalize(full_df['behavioral_features'])
final_df = pd.concat([full_df[['cleaned_text', 'label']].reset_index(drop=True), behavior_df.reset_index(drop=True)], axis=1)

# Save prepared data
final_df.to_csv('multimodal_dataset.csv', index=False)
