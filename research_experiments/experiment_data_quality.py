"""
Experiment 1: Data Quality Validation

This module validates the synthetic mental health dataset for federated learning:
- Class distribution analysis (global and per-client)
- Feature-label correlations
- Temporal autocorrelation checks
- Non-IID heterogeneity validation

Author: Research Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import pointbiserialr
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger('ResearchExperiments.DataQuality')


class DataQualityValidator:
    """
    Comprehensive data quality validation for federated learning datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data quality validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config['paths']['data_dir'])
        self.results_dir = Path(config['paths']['results_dir'])
        self.figures_dir = Path(config['paths']['figures_dir']) / "data_quality"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.features = config['data']['features']
        self.label = config['data']['label']
        
    def load_data(self) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load the synthetic dataset (global and per-client partitions).
        
        Returns:
            Tuple of (global_dataframe, list_of_client_dataframes)
        """
        logger.info("Loading synthetic mental health dataset...")
        
        # Try to load from different possible locations
        possible_paths = [
            self.data_dir / "synthetic_data.csv",
            self.data_dir / "mental_health_synthetic.csv",
            Path("../federated-mental-health/data/synthetic_fixed/synthetic_data.csv"),
        ]
        
        global_df = None
        for path in possible_paths:
            if path.exists():
                global_df = pd.read_csv(path)
                logger.info(f"Loaded global dataset from: {path}")
                break
        
        if global_df is None:
            # Generate synthetic data if not found
            logger.warning("Dataset not found. Generating synthetic data...")
            global_df = self._generate_synthetic_data()
        
        # Partition data into clients
        num_clients = self.config['federated']['num_clients']
        client_dfs = self._partition_data(global_df, num_clients)
        
        logger.info(f"Dataset loaded: {len(global_df)} samples, {num_clients} clients")
        
        return global_df, client_dfs
    
    def _generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate realistic mental health data using new generator.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with realistic synthetic data
        """
        from realistic_data_generator import RealisticDataGenerator
        
        # Use realistic data generator with 3% positive class ratio
        generator = RealisticDataGenerator(
            seed=self.config['reproducibility']['seed'],
            positive_ratio=0.03
        )
        
        n_users = 100
        df = generator.generate_data(n_samples=n_samples, n_users=n_users)
        
        # Validate data quality with logistic regression baseline
        validation_results = generator.validate_with_logistic_regression(df)
        
        logger.info(f"Generated realistic data: {len(df)} samples")
        logger.info(f"Data quality: {validation_results['quality']}")
        logger.info(f"Improvement over random: {validation_results['improvement_factor']:.2f}x")
        
        return df
    
    def _partition_data(self, df: pd.DataFrame, num_clients: int) -> List[pd.DataFrame]:
        """
        Partition data into client datasets (non-IID by user).
        
        Args:
            df: Global dataframe
            num_clients: Number of clients
            
        Returns:
            List of client dataframes
        """
        if 'user_id' in df.columns:
            # Partition by user for natural non-IID
            unique_users = df['user_id'].unique()
            np.random.shuffle(unique_users)
            
            users_per_client = len(unique_users) // num_clients
            client_dfs = []
            
            for i in range(num_clients):
                start_idx = i * users_per_client
                end_idx = start_idx + users_per_client if i < num_clients - 1 else len(unique_users)
                client_users = unique_users[start_idx:end_idx]
                client_df = df[df['user_id'].isin(client_users)].copy()
                client_dfs.append(client_df)
        else:
            # Simple random partition
            client_size = len(df) // num_clients
            client_dfs = []
            for i in range(num_clients):
                start_idx = i * client_size
                end_idx = start_idx + client_size if i < num_clients - 1 else len(df)
                client_dfs.append(df.iloc[start_idx:end_idx].copy())
        
        return client_dfs
    
    def analyze_class_distribution(self, global_df: pd.DataFrame, 
                                  client_dfs: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze class distribution globally and per-client.
        
        Args:
            global_df: Global dataframe
            client_dfs: List of client dataframes
            
        Returns:
            Dictionary with class distribution metrics
        """
        logger.info("Analyzing class distribution...")
        
        # Global distribution
        global_positive = global_df[self.label].mean()
        
        # Per-client distribution
        client_positive_ratios = [df[self.label].mean() for df in client_dfs]
        
        # Check for extreme imbalance - adjusted for realistic mental health data
        # Mental health crises are rare: 2-5% is realistic and acceptable
        realistic_min = 0.02  # 2%
        realistic_max = 0.08  # 8%
        
        extreme_imbalance = (global_positive < realistic_min or 
                           global_positive > realistic_max)
        
        results = {
            'global_positive_ratio': float(global_positive),
            'global_negative_ratio': float(1 - global_positive),
            'client_positive_ratios': [float(r) for r in client_positive_ratios],
            'client_positive_mean': float(np.mean(client_positive_ratios)),
            'client_positive_std': float(np.std(client_positive_ratios)),
            'client_positive_min': float(np.min(client_positive_ratios)),
            'client_positive_max': float(np.max(client_positive_ratios)),
            'is_balanced': bool(not extreme_imbalance),
            'extreme_imbalance_flag': bool(extreme_imbalance)
        }
        
        logger.info(f"Global positive ratio: {global_positive:.2%}")
        logger.info(f"Client ratios - Mean: {results['client_positive_mean']:.2%}, "
                   f"Std: {results['client_positive_std']:.3f}")
        
        return results
    
    def analyze_feature_correlations(self, global_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Point-Biserial correlations between features and binary label.
        
        Args:
            global_df: Global dataframe
            
        Returns:
            Dictionary of feature correlations
        """
        logger.info("Computing feature-label correlations...")
        
        correlations = {}
        
        for feature in self.features:
            if feature in global_df.columns:
                # Point-biserial correlation for continuous-binary
                corr, pval = pointbiserialr(global_df[self.label], global_df[feature])
                correlations[feature] = float(corr)
        
        # Sort by absolute correlation
        sorted_corrs = dict(sorted(correlations.items(), 
                                  key=lambda x: abs(x[1]), 
                                  reverse=True))
        
        logger.info("Top 3 correlations:")
        for feature, corr in list(sorted_corrs.items())[:3]:
            logger.info(f"  {feature}: {corr:.3f}")
        
        return sorted_corrs
    
    def analyze_temporal_correlation(self, global_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check temporal autocorrelation in sequential features.
        
        Args:
            global_df: Global dataframe
            
        Returns:
            Dictionary with temporal correlation metrics
        """
        logger.info("Analyzing temporal autocorrelation...")
        
        lag = self.config['experiments']['data_quality']['temporal_lag']
        temporal_results = {}
        
        # If we have user_id and day, compute within-user autocorrelation
        if 'user_id' in global_df.columns and 'day' in global_df.columns:
            for feature in self.features:
                if feature in global_df.columns:
                    autocorrs = []
                    
                    for user_id in global_df['user_id'].unique():
                        user_data = global_df[global_df['user_id'] == user_id].sort_values('day')
                        
                        if len(user_data) > lag:
                            feature_values = user_data[feature].values
                            # Lag-1 autocorrelation
                            corr = np.corrcoef(feature_values[:-lag], feature_values[lag:])[0, 1]
                            if not np.isnan(corr):
                                autocorrs.append(corr)
                    
                    if autocorrs:
                        temporal_results[feature] = {
                            'mean_autocorr': float(np.mean(autocorrs)),
                            'std_autocorr': float(np.std(autocorrs)),
                            'has_temporal_structure': bool(float(np.mean(autocorrs)) > 0.3)
                        }
        else:
            logger.warning("No temporal structure detected in data (missing user_id/day)")
            temporal_results['note'] = "Temporal analysis requires user_id and day columns"
        
        return temporal_results
    
    def analyze_non_iid(self, client_dfs: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate non-IID heterogeneity across clients using distribution divergence.
        
        Args:
            client_dfs: List of client dataframes
            
        Returns:
            Dictionary with non-IID metrics
        """
        logger.info("Validating non-IID heterogeneity...")
        
        # Focus on stress_level as key feature
        key_feature = 'stress_level'
        
        if key_feature not in client_dfs[0].columns:
            key_feature = self.features[0]  # Fallback to first feature
        
        # Compute histograms for each client
        bins = np.linspace(0, 1, 11)  # 10 bins
        client_distributions = []
        
        for df in client_dfs:
            hist, _ = np.histogram(df[key_feature], bins=bins, density=True)
            hist = hist / hist.sum()  # Normalize to probability distribution
            client_distributions.append(hist)
        
        # Compute pairwise JS divergences
        js_divergences = []
        for i in range(len(client_distributions)):
            for j in range(i + 1, len(client_distributions)):
                js_div = jensenshannon(client_distributions[i], client_distributions[j])
                js_divergences.append(js_div)
        
        mean_js_div = np.mean(js_divergences)
        
        # Non-IID confirmed if mean JS divergence > 0.1
        is_non_iid = mean_js_div > 0.1
        
        results = {
            'key_feature': key_feature,
            'mean_js_divergence': float(mean_js_div),
            'max_js_divergence': float(np.max(js_divergences)),
            'min_js_divergence': float(np.min(js_divergences)),
            'is_non_iid': bool(is_non_iid),
            'heterogeneity_level': 'High' if mean_js_div > 0.2 else 'Moderate' if mean_js_div > 0.1 else 'Low',
            'status': 'Confirmed' if is_non_iid else 'Not detected'
        }
        
        logger.info(f"Mean JS divergence: {mean_js_div:.3f} - Non-IID: {is_non_iid}")
        
        return results
    
    def generate_diagnostic_plots(self, global_df: pd.DataFrame, 
                                 client_dfs: List[pd.DataFrame],
                                 results: Dict[str, Any]):
        """
        Generate diagnostic visualizations.
        
        Args:
            global_df: Global dataframe
            client_dfs: List of client dataframes
            results: Results dictionary
        """
        logger.info("Generating diagnostic plots...")
        
        # Plot 1: Class distribution across clients
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1a: Client class distribution
        client_ratios = results['class_distribution']['client_positive_ratios']
        axes[0, 0].bar(range(len(client_ratios)), client_ratios, color='steelblue', alpha=0.7)
        axes[0, 0].axhline(results['class_distribution']['global_positive_ratio'], 
                          color='red', linestyle='--', label='Global Mean')
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Positive Class Ratio')
        axes[0, 0].set_title('Class Distribution Across Clients')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 1b: Feature-label correlations
        correlations = results['feature_correlations']
        features = list(correlations.keys())
        corr_values = list(correlations.values())
        colors = ['green' if c > 0 else 'red' for c in corr_values]
        
        axes[0, 1].barh(features, corr_values, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Point-Biserial Correlation')
        axes[0, 1].set_title('Feature-Label Correlations')
        axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 1c: Feature distributions by label
        feature_to_plot = self.features[0] if self.features else 'stress_level'
        if feature_to_plot in global_df.columns:
            global_df[global_df[self.label] == 0][feature_to_plot].hist(
                ax=axes[1, 0], bins=30, alpha=0.6, label='Low Risk', color='blue'
            )
            global_df[global_df[self.label] == 1][feature_to_plot].hist(
                ax=axes[1, 0], bins=30, alpha=0.6, label='High Risk', color='red'
            )
            axes[1, 0].set_xlabel(feature_to_plot)
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'{feature_to_plot} Distribution by Risk Level')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 1d: Non-IID divergence heatmap
        # Compute pairwise divergences for visualization
        n_clients = min(10, len(client_dfs))  # Limit to 10 for visualization
        divergence_matrix = np.zeros((n_clients, n_clients))
        
        key_feature = results['non_iid_metrics']['key_feature']
        bins = np.linspace(0, 1, 11)
        
        distributions = []
        for i in range(n_clients):
            hist, _ = np.histogram(client_dfs[i][key_feature], bins=bins, density=True)
            hist = hist / (hist.sum() + 1e-10)
            distributions.append(hist)
        
        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    divergence_matrix[i, j] = jensenshannon(distributions[i], distributions[j])
        
        im = axes[1, 1].imshow(divergence_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xlabel('Client ID')
        axes[1, 1].set_ylabel('Client ID')
        axes[1, 1].set_title('Client Distribution Divergence (JS Divergence)')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plot_file = self.figures_dir / "data_quality_diagnostics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Diagnostic plot saved: {plot_file}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete data quality validation experiment.
        
        Returns:
            Dictionary with all validation results
        """
        logger.info("="*60)
        logger.info("EXPERIMENT 1: DATA QUALITY VALIDATION")
        logger.info("="*60)
        
        # Load data
        global_df, client_dfs = self.load_data()
        
        # Run all validation tests
        results = {
            'total_samples': len(global_df),
            'num_clients': len(client_dfs),
            'features': self.features,
            'label': self.label,
        }
        
        # Test 1: Class distribution
        results['class_distribution'] = self.analyze_class_distribution(global_df, client_dfs)
        
        # Test 2: Feature correlations
        results['feature_correlations'] = self.analyze_feature_correlations(global_df)
        
        # Test 3: Temporal correlation
        results['temporal_correlations'] = self.analyze_temporal_correlation(global_df)
        
        # Test 4: Non-IID validation
        results['non_iid_metrics'] = self.analyze_non_iid(client_dfs)
        
        # Generate visualizations
        self.generate_diagnostic_plots(global_df, client_dfs, results)
        
        # Save diagnostic report
        report_file = self.figures_dir / "data_diagnostic_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Diagnostic report saved: {report_file}")
        logger.info("Data quality validation completed successfully")
        
        return results


def run_data_quality_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for data quality experiment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment results
    """
    validator = DataQualityValidator(config)
    return validator.run()


if __name__ == "__main__":
    # For standalone testing
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_data_quality_experiment(config)
    print("\nData Quality Validation Results:")
    print(json.dumps(results, indent=2))
