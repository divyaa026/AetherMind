#!/usr/bin/env python3
"""
MindGuard Data Ingestion Pipeline

This script processes and prepares mental health datasets for the MindGuard system.
It handles data cleaning, validation, and preparation for ML training.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import hashlib
from dataclasses import dataclass
import click

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.utils.text_processing import TextProcessor
from ml.utils.data_validation import DataValidator
from ml.utils.privacy import PrivacyProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    name: str
    file_path: str
    text_column: str
    label_column: str
    label_mapping: Dict[str, int]
    min_text_length: int = 10
    max_text_length: int = 2000
    test_split: float = 0.2
    validation_split: float = 0.1


class DataIngestionPipeline:
    """Main data ingestion pipeline for MindGuard"""
    
    def __init__(self, output_dir: str = "processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.text_processor = TextProcessor()
        self.data_validator = DataValidator()
        self.privacy_processor = PrivacyProcessor()
        
        # Dataset configurations
        self.dataset_configs = {
            "depression_reddit": DatasetConfig(
                name="depression_reddit",
                file_path="datasets/depression_dataset_reddit_cleaned.csv",
                text_column="clean_text",
                label_column="is_depression",
                label_mapping={"0": 0, "1": 1, 0: 0, 1: 1},
                min_text_length=20,
                max_text_length=1000
            ),
            "suicide_detection": DatasetConfig(
                name="suicide_detection",
                file_path="datasets/Suicide_Detection.csv",
                text_column="text",
                label_column="class",
                label_mapping={"suicide": 1, "non-suicide": 0},
                min_text_length=10,
                max_text_length=2000
            )
        }
    
    def load_dataset(self, config: DatasetConfig) -> pd.DataFrame:
        """Load dataset from file"""
        logger.info(f"Loading dataset: {config.name}")
        
        try:
            df = pd.read_csv(config.file_path)
            logger.info(f"Loaded {len(df)} records from {config.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset {config.name}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = text.lower()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def validate_data(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Validate and filter data"""
        logger.info(f"Validating dataset: {config.name}")
        
        initial_count = len(df)
        
        # Check required columns
        required_columns = [config.text_column, config.label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing values
        df = df.dropna(subset=[config.text_column, config.label_column])
        
        # Clean text
        df[config.text_column] = df[config.text_column].apply(self.clean_text)
        
        # Filter by text length
        df = df[
            (df[config.text_column].str.len() >= config.min_text_length) &
            (df[config.text_column].str.len() <= config.max_text_length)
        ]
        
        # Validate labels - be more flexible with label types
        valid_labels = set(config.label_mapping.keys())
        # Convert labels to string for comparison
        df[config.label_column] = df[config.label_column].astype(str)
        df = df[df[config.label_column].isin(valid_labels)]
        
        # Map labels to integers
        df['label'] = df[config.label_column].map(config.label_mapping)
        
        # Remove rows with invalid labels
        df = df.dropna(subset=['label'])
        
        final_count = len(df)
        logger.info(f"Data validation complete: {initial_count} -> {final_count} records")
        
        return df
    
    def apply_privacy_protection(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Apply privacy protection measures"""
        logger.info("Applying privacy protection measures")
        
        # Generate anonymized IDs
        df['anonymized_id'] = df.index.map(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16])
        
        # Remove any potential PII from text (basic implementation)
        df['processed_text'] = df[config.text_column].apply(self.privacy_processor.remove_pii)
        
        return df
    
    def split_data(self, df: pd.DataFrame, config: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        logger.info("Splitting data into train/validation/test sets")
        
        # Stratified split to maintain label distribution
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=config.test_split,
            stratify=df['label'],
            random_state=42
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=config.validation_split / (1 - config.test_split),
            stratify=train_val['label'],
            random_state=42
        )
        
        logger.info(f"Data split complete:")
        logger.info(f"  Train: {len(train)} records")
        logger.info(f"  Validation: {len(val)} records")
        logger.info(f"  Test: {len(test)} records")
        
        return train, val, test
    
    def save_processed_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: DatasetConfig):
        """Save processed data to files"""
        logger.info(f"Saving processed data for {config.name}")
        
        dataset_dir = self.output_dir / config.name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save splits
        train.to_csv(dataset_dir / "train.csv", index=False)
        val.to_csv(dataset_dir / "validation.csv", index=False)
        test.to_csv(dataset_dir / "test.csv", index=False)
        
        # Save metadata
        metadata = {
            "dataset_name": config.name,
            "processed_at": datetime.now().isoformat(),
            "splits": {
                "train": int(len(train)),
                "validation": int(len(val)),
                "test": int(len(test))
            },
            "label_distribution": {
                "train": {str(k): int(v) for k, v in train['label'].value_counts().to_dict().items()},
                "validation": {str(k): int(v) for k, v in val['label'].value_counts().to_dict().items()},
                "test": {str(k): int(v) for k, v in test['label'].value_counts().to_dict().items()}
            },
            "text_statistics": {
                "avg_length": float(train['processed_text'].str.len().mean()),
                "min_length": int(train['processed_text'].str.len().min()),
                "max_length": int(train['processed_text'].str.len().max())
            }
        }
        
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved to {dataset_dir}")
    
    def process_dataset(self, config: DatasetConfig) -> Dict[str, pd.DataFrame]:
        """Process a single dataset"""
        logger.info(f"Processing dataset: {config.name}")
        
        # Load data
        df = self.load_dataset(config)
        
        # Validate and clean
        df = self.validate_data(df, config)
        
        # Apply privacy protection
        df = self.apply_privacy_protection(df, config)
        
        # Split data
        train, val, test = self.split_data(df, config)
        
        # Save processed data
        self.save_processed_data(train, val, test, config)
        
        return {
            "train": train,
            "validation": val,
            "test": test
        }
    
    def run_pipeline(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run the complete data ingestion pipeline"""
        logger.info("Starting MindGuard data ingestion pipeline")
        
        processed_datasets = {}
        
        for dataset_name, config in self.dataset_configs.items():
            try:
                if Path(config.file_path).exists():
                    processed_datasets[dataset_name] = self.process_dataset(config)
                else:
                    logger.warning(f"Dataset file not found: {config.file_path}")
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue
        
        # Generate summary report
        self.generate_summary_report(processed_datasets)
        
        logger.info("Data ingestion pipeline completed successfully")
        return processed_datasets
    
    def generate_summary_report(self, processed_datasets: Dict[str, Dict[str, pd.DataFrame]]):
        """Generate a summary report of all processed datasets"""
        logger.info("Generating summary report")
        
        summary = {
            "pipeline_run_at": datetime.now().isoformat(),
            "total_datasets": len(processed_datasets),
            "datasets": {}
        }
        
        for dataset_name, splits in processed_datasets.items():
            total_records = sum(len(split) for split in splits.values())
            summary["datasets"][dataset_name] = {
                "total_records": int(total_records),
                "splits": {name: int(len(split)) for name, split in splits.items()},
                "label_distribution": {str(k): int(v) for k, v in splits["train"]["label"].value_counts().to_dict().items()}
            }
        
        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_file}")


@click.command()
@click.option('--output-dir', default='processed_data', help='Output directory for processed data')
@click.option('--dataset', help='Process specific dataset only')
def main(output_dir: str, dataset: Optional[str]):
    """MindGuard Data Ingestion Pipeline"""
    
    pipeline = DataIngestionPipeline(output_dir)
    
    if dataset:
        if dataset not in pipeline.dataset_configs:
            logger.error(f"Unknown dataset: {dataset}")
            return
        
        config = pipeline.dataset_configs[dataset]
        try:
            pipeline.process_dataset(config)
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
    else:
        pipeline.run_pipeline()


if __name__ == "__main__":
    main()
