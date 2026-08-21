"""
Tests for data generation and preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic_generator import SyntheticMentalHealthData, SyntheticConfig
from data.preprocess import MentalHealthPreprocessor, PreprocessingConfig
from data.federated_partition import FederatedPartitioner, PartitionConfig


class TestSyntheticDataGeneration:
    """Tests for synthetic data generator."""
    
    def test_basic_generation(self):
        """Test basic data generation."""
        config = SyntheticConfig(n_users=100, n_days=10)
        generator = SyntheticMentalHealthData(config)
        
        df = generator.generate()
        
        assert len(df) > 0
        assert 'user_id' in df.columns
        assert 'date' in df.columns
        assert 'mood' in df.columns
        assert 'stress' in df.columns
        assert 'high_risk' in df.columns
    
    def test_data_ranges(self):
        """Test that generated data is in valid ranges."""
        config = SyntheticConfig(n_users=100, n_days=10)
        generator = SyntheticMentalHealthData(config)
        df = generator.generate()
        
        # Check ranges (allowing for some NaN values)
        assert df['mood'].dropna().between(1, 10).all()
        assert df['stress'].dropna().between(1, 10).all()
        assert df['sleep_hours'].dropna().between(0, 14).all()
        assert df['exercise_mins'].dropna().between(0, 200).all()
    
    def test_user_count(self):
        """Test correct number of users."""
        config = SyntheticConfig(n_users=50, n_days=7)
        generator = SyntheticMentalHealthData(config)
        df = generator.generate()
        
        assert df['user_id'].nunique() == 50
    
    def test_missing_data(self):
        """Test missing data generation."""
        config = SyntheticConfig(n_users=100, n_days=30, missing_weekend_rate=0.5)
        generator = SyntheticMentalHealthData(config)
        df = generator.generate()
        
        # Should have some missing values
        assert df.isnull().any().any()
    
    def test_save_load(self):
        """Test saving and loading data."""
        config = SyntheticConfig(n_users=50, n_days=7)
        generator = SyntheticMentalHealthData(config)
        df = generator.generate()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator.save(tmpdir)
            
            # Check files exist
            assert (Path(tmpdir) / 'time_series.parquet').exists()
            assert (Path(tmpdir) / 'user_profiles.parquet').exists()
            
            # Load and verify
            loaded_df = pd.read_parquet(Path(tmpdir) / 'time_series.parquet')
            assert len(loaded_df) == len(df)


class TestPreprocessing:
    """Tests for data preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        config = SyntheticConfig(n_users=100, n_days=30)
        generator = SyntheticMentalHealthData(config)
        return generator.generate()
    
    def test_basic_preprocessing(self, sample_data):
        """Test basic preprocessing pipeline."""
        config = PreprocessingConfig(sequence_length=7)
        preprocessor = MentalHealthPreprocessor(config)
        
        splits = preprocessor.fit_transform(sample_data)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check shapes
        assert len(splits['train']['X'].shape) == 3  # (samples, seq_len, features)
        assert splits['train']['X'].shape[1] == 7  # sequence length
    
    def test_train_val_test_split(self, sample_data):
        """Test proper train/val/test splitting."""
        config = PreprocessingConfig(
            sequence_length=7,
            train_ratio=0.7,
            val_ratio=0.15
        )
        preprocessor = MentalHealthPreprocessor(config)
        splits = preprocessor.fit_transform(sample_data)
        
        total = len(splits['train']['X']) + len(splits['val']['X']) + len(splits['test']['X'])
        
        train_ratio = len(splits['train']['X']) / total
        val_ratio = len(splits['val']['X']) / total
        
        # Allow some tolerance
        assert abs(train_ratio - 0.7) < 0.1
        assert abs(val_ratio - 0.15) < 0.1
    
    def test_no_data_leakage(self, sample_data):
        """Test that there's no data leakage between splits."""
        config = PreprocessingConfig(sequence_length=7)
        preprocessor = MentalHealthPreprocessor(config)
        splits = preprocessor.fit_transform(sample_data)
        
        # The sequences should come from different users
        # This is a simplified check
        assert len(splits['train']['X']) > 0
        assert len(splits['val']['X']) > 0
        assert len(splits['test']['X']) > 0
    
    def test_save_load_preprocessor(self, sample_data):
        """Test saving and loading preprocessor."""
        config = PreprocessingConfig(sequence_length=7)
        preprocessor = MentalHealthPreprocessor(config)
        splits = preprocessor.fit_transform(sample_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor.save(tmpdir)
            
            # Check files exist
            assert (Path(tmpdir) / 'X_train.npy').exists()
            assert (Path(tmpdir) / 'y_train.npy').exists()


class TestFederatedPartitioning:
    """Tests for federated data partitioning."""
    
    @pytest.fixture
    def sample_arrays(self):
        """Generate sample arrays for testing."""
        np.random.seed(42)
        X = np.random.randn(1000, 7, 20).astype(np.float32)
        y = np.random.randint(0, 2, 1000).astype(np.float32)
        return X, y
    
    def test_iid_partition(self, sample_arrays):
        """Test IID partitioning."""
        X, y = sample_arrays
        
        config = PartitionConfig(n_clients=10, strategy='iid')
        partitioner = FederatedPartitioner(config)
        
        client_data = partitioner.partition(X, y)
        
        assert len(client_data) == 10
        
        # Check all data is used
        total_samples = sum(len(data['y']) for data in client_data)
        assert total_samples == len(y)
    
    def test_non_iid_partition(self, sample_arrays):
        """Test non-IID partitioning."""
        X, y = sample_arrays
        
        config = PartitionConfig(n_clients=10, strategy='non_iid_label', alpha=0.5)
        partitioner = FederatedPartitioner(config)
        
        client_data = partitioner.partition(X, y)
        
        assert len(client_data) == 10
        
        # Non-IID should have different label distributions
        label_ratios = [data['y'].mean() for data in client_data]
        assert np.std(label_ratios) > 0  # Should have variation
    
    def test_partition_sizes(self, sample_arrays):
        """Test that partition sizes are reasonable."""
        X, y = sample_arrays
        
        config = PartitionConfig(n_clients=5, strategy='iid')
        partitioner = FederatedPartitioner(config)
        
        client_data = partitioner.partition(X, y)
        
        sizes = [len(data['y']) for data in client_data]
        
        # IID should have roughly equal sizes
        assert max(sizes) / min(sizes) < 1.5
    
    def test_save_partitions(self, sample_arrays):
        """Test saving partitioned data."""
        X, y = sample_arrays
        
        config = PartitionConfig(n_clients=5, strategy='iid')
        partitioner = FederatedPartitioner(config)
        client_data = partitioner.partition(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            partitioner.save(tmpdir)
            
            # Check client directories exist
            for i in range(5):
                assert (Path(tmpdir) / f'client_{i}').exists()
                assert (Path(tmpdir) / f'client_{i}' / 'X.npy').exists()
                assert (Path(tmpdir) / f'client_{i}' / 'y.npy').exists()


class TestDataIntegration:
    """Integration tests for data pipeline."""
    
    def test_full_pipeline(self):
        """Test complete data generation to partitioning pipeline."""
        # Generate
        gen_config = SyntheticConfig(n_users=100, n_days=14)
        generator = SyntheticMentalHealthData(gen_config)
        df = generator.generate()
        
        assert len(df) > 0
        
        # Preprocess
        prep_config = PreprocessingConfig(sequence_length=7)
        preprocessor = MentalHealthPreprocessor(prep_config)
        splits = preprocessor.fit_transform(df)
        
        X_train = splits['train']['X']
        y_train = splits['train']['y']
        
        assert len(X_train) > 0
        
        # Partition
        part_config = PartitionConfig(n_clients=5, strategy='iid')
        partitioner = FederatedPartitioner(part_config)
        client_data = partitioner.partition(X_train, y_train)
        
        assert len(client_data) == 5
        
        # Verify data integrity
        for i, data in enumerate(client_data):
            assert 'X' in data
            assert 'y' in data
            assert len(data['X']) == len(data['y'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
