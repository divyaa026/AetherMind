"""
Main experiment runner for federated mental health prediction.
Orchestrates data generation, training, and evaluation.
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Local imports
from data.synthetic_generator import SyntheticMentalHealthData, SyntheticConfig
from data.preprocess import MentalHealthPreprocessor, PreprocessingConfig
from data.federated_partition import FederatedPartitioner, PartitionConfig
from models.architecture import create_model
from models.train_local import LocalTrainer, TrainingConfig
from federated.coordinator import FederatedCoordinator, FederatedConfig
from evaluation.metrics import MetricsCalculator
from evaluation.privacy_eval import PrivacyEvaluator
from evaluation.comparative_analysis import ComparativeAnalyzer, ExperimentResult


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(config: dict, output_dir: Path) -> dict:
    """
    Setup experiment directories and logging.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory
        
    Returns:
        Updated config with paths
    """
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    config['output_dir'] = str(output_dir)
    config['data_dir'] = str(output_dir / 'data')
    config['models_dir'] = str(output_dir / 'models')
    config['results_dir'] = str(output_dir / 'results')
    
    return config


def generate_data(config: dict) -> dict:
    """
    Generate synthetic data for experiments.
    
    Args:
        config: Data configuration
        
    Returns:
        Data generation results
    """
    print("\n" + "="*60)
    print("Phase 1: Generating Synthetic Data")
    print("="*60)
    
    data_config = SyntheticConfig(
        n_users=config.get('n_users', 10000),
        n_days=config.get('n_days', 30),
        missing_weekend_rate=config.get('missing_rate', 0.3)
    )
    
    generator = SyntheticMentalHealthData(data_config)
    df = generator.generate()
    
    # Save raw data
    data_dir = Path(config['data_dir'])
    generator.save(str(data_dir / 'raw'))
    
    return {
        'n_samples': len(df),
        'n_users': data_config.n_users,
        'n_days': data_config.n_days,
        'features': list(df.columns)
    }


def preprocess_data(config: dict) -> dict:
    """
    Preprocess data for training.
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        Preprocessing results
    """
    print("\n" + "="*60)
    print("Phase 2: Preprocessing Data")
    print("="*60)
    
    data_dir = Path(config['data_dir'])
    
    preprocess_config = PreprocessingConfig(
        sequence_length=config.get('sequence_length', 7),
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15)
    )
    
    preprocessor = MentalHealthPreprocessor(preprocess_config)
    
    # Load and process data
    import pandas as pd
    df = pd.read_parquet(data_dir / 'raw' / 'time_series.parquet')
    
    splits = preprocessor.fit_transform(df)
    preprocessor.save(str(data_dir / 'processed'))
    
    return {
        'train_size': len(splits['train']['X']),
        'val_size': len(splits['val']['X']),
        'test_size': len(splits['test']['X']),
        'feature_dim': splits['train']['X'].shape[2],
        'sequence_length': splits['train']['X'].shape[1]
    }


def partition_data(config: dict) -> dict:
    """
    Partition data for federated learning.
    
    Args:
        config: Partition configuration
        
    Returns:
        Partition results
    """
    print("\n" + "="*60)
    print("Phase 3: Partitioning Data for Federated Learning")
    print("="*60)
    
    data_dir = Path(config['data_dir'])
    
    partition_config = PartitionConfig(
        n_clients=config.get('n_clients', 10),
        strategy=config.get('partition_strategy', 'iid'),
        alpha=config.get('dirichlet_alpha', 0.5)
    )
    
    partitioner = FederatedPartitioner(partition_config)
    
    # Load preprocessed data
    X_train = np.load(data_dir / 'processed' / 'X_train.npy')
    y_train = np.load(data_dir / 'processed' / 'y_train.npy')
    
    client_data = partitioner.partition(X_train, y_train)
    partitioner.save(str(data_dir / 'partitions'))
    
    return {
        'n_clients': partition_config.n_clients,
        'strategy': partition_config.strategy,
        'samples_per_client': [len(data['y']) for data in client_data]
    }


def run_federated_training(config: dict) -> dict:
    """
    Run federated training experiment.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    print("\n" + "="*60)
    print("Phase 4: Federated Training")
    print("="*60)
    
    data_dir = Path(config['data_dir'])
    
    # Create federated config
    fed_config = FederatedConfig(
        partition_dir=str(data_dir / 'partitions'),
        n_rounds=config.get('n_rounds', 50),
        local_epochs=config.get('local_epochs', 5),
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 0.001),
        use_dp=config.get('use_dp', True),
        dp_epsilon=config.get('dp_epsilon', 1.0),
        dp_delta=config.get('dp_delta', 1e-5),
        aggregation_strategy=config.get('aggregation_strategy', 'fedavg'),
        output_dir=config['results_dir'],
        experiment_name=config.get('experiment_name', 'federated_run')
    )
    
    # Create and run coordinator
    coordinator = FederatedCoordinator(fed_config)
    coordinator.setup_from_partitions(
        partition_dir=str(data_dir / 'partitions'),
        val_data_path=str(data_dir / 'processed')
    )
    
    summary = coordinator.train()
    
    return summary


def evaluate_model(config: dict) -> dict:
    """
    Evaluate trained model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Evaluation results
    """
    print("\n" + "="*60)
    print("Phase 5: Model Evaluation")
    print("="*60)
    
    data_dir = Path(config['data_dir'])
    results_dir = Path(config['results_dir'])
    
    # Load test data
    X_test = np.load(data_dir / 'processed' / 'X_test.npy')
    y_test = np.load(data_dir / 'processed' / 'y_test.npy')
    
    # Load model
    checkpoint = torch.load(
        results_dir / config['experiment_name'] / 'final_model.pt'
    )
    
    input_dim = X_test.shape[2]
    model = create_model(input_dim)
    
    for name, param in model.named_parameters():
        if name in checkpoint['global_state']:
            param.data = checkpoint['global_state'][name]
    
    # Compute metrics
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(X_test))
        probs = torch.sigmoid(logits).squeeze().numpy()
    
    calculator = MetricsCalculator()
    metrics = calculator.compute_all_metrics(y_test, probs)
    
    # Privacy evaluation
    X_train = np.load(data_dir / 'processed' / 'X_train.npy')
    y_train = np.load(data_dir / 'processed' / 'y_train.npy')
    
    privacy_evaluator = PrivacyEvaluator(
        target_epsilon=config.get('dp_epsilon', 1.0),
        target_delta=config.get('dp_delta', 1e-5)
    )
    
    privacy_result = privacy_evaluator.evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dp_params={
            'noise_multiplier': config.get('noise_multiplier', 1.0),
            'sampling_probability': config.get('batch_size', 32) / len(X_train),
            'n_steps': config.get('n_rounds', 50) * config.get('local_epochs', 5)
        }
    )
    
    # Save results
    results = {
        'classification': metrics['classification'],
        'calibration': metrics['calibration'],
        'privacy': privacy_result.to_dict()
    }
    
    with open(results_dir / config['experiment_name'] / 'evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_experiment(config_path: str = None, config: dict = None) -> dict:
    """
    Run complete experiment pipeline.
    
    Args:
        config_path: Path to config YAML file
        config: Configuration dictionary (overrides config_path)
        
    Returns:
        Complete experiment results
    """
    # Load config
    if config is None:
        if config_path:
            config = load_config(config_path)
        else:
            config = {}
    
    # Set defaults
    config.setdefault('experiment_name', f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    config.setdefault('seed', 42)
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Setup experiment
    output_dir = Path(config.get('output_base', './experiments')) / config['experiment_name']
    config = setup_experiment(config, output_dir)
    
    results = {
        'experiment_name': config['experiment_name'],
        'start_time': datetime.now().isoformat()
    }
    
    try:
        # Phase 1: Generate data
        if config.get('generate_data', True):
            results['data_generation'] = generate_data(config)
        
        # Phase 2: Preprocess
        if config.get('preprocess', True):
            results['preprocessing'] = preprocess_data(config)
        
        # Phase 3: Partition
        if config.get('partition', True):
            results['partitioning'] = partition_data(config)
        
        # Phase 4: Train
        if config.get('train', True):
            results['training'] = run_federated_training(config)
        
        # Phase 5: Evaluate
        if config.get('evaluate', True):
            results['evaluation'] = evaluate_model(config)
        
        results['status'] = 'completed'
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        raise
    
    finally:
        results['end_time'] = datetime.now().isoformat()
        
        # Save final results
        with open(output_dir / 'experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run federated mental health prediction experiments'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--experiment-name', '-n',
        type=str,
        help='Experiment name'
    )
    
    parser.add_argument(
        '--n-clients',
        type=int,
        default=10,
        help='Number of federated clients'
    )
    
    parser.add_argument(
        '--n-rounds',
        type=int,
        default=50,
        help='Number of federated rounds'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1.0,
        help='Privacy budget epsilon'
    )
    
    parser.add_argument(
        '--no-dp',
        action='store_true',
        help='Disable differential privacy'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments',
        help='Output directory'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Build config from args
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # Override with command line args
    if args.experiment_name:
        config['experiment_name'] = args.experiment_name
    config['n_clients'] = args.n_clients
    config['n_rounds'] = args.n_rounds
    config['dp_epsilon'] = args.epsilon
    config['use_dp'] = not args.no_dp
    config['output_base'] = args.output_dir
    config['seed'] = args.seed
    
    # Run experiment
    results = run_experiment(config=config)
    
    print("\n" + "="*60)
    print("Experiment Complete")
    print("="*60)
    print(f"Status: {results['status']}")
    print(f"Output: {config['output_dir']}")
    
    if 'evaluation' in results:
        eval_results = results['evaluation']
        print(f"\nClassification Results:")
        print(f"  Accuracy: {eval_results['classification']['accuracy']:.4f}")
        print(f"  AUC-ROC: {eval_results['classification']['auc_roc']:.4f}")
        print(f"  F1: {eval_results['classification']['f1']:.4f}")
        
        print(f"\nPrivacy Results:")
        print(f"  Epsilon: {eval_results['privacy']['epsilon']:.4f}")
        print(f"  MIA Advantage: {eval_results['privacy']['membership_inference_advantage']:.4f}")
        print(f"  Risk Level: {eval_results['privacy']['privacy_risk_level']}")


if __name__ == "__main__":
    main()
