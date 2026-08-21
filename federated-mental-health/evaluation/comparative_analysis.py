"""
Comparative analysis for federated learning experiments.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import copy


@dataclass
class ExperimentResult:
    """Container for a single experiment result."""
    name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    privacy: Dict[str, float]
    training_time: float
    n_rounds: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'config': self.config,
            'metrics': self.metrics,
            'privacy': self.privacy,
            'training_time': self.training_time,
            'n_rounds': self.n_rounds
        }


class ComparativeAnalyzer:
    """
    Comparative analysis across multiple federated learning experiments.
    
    Enables:
    - Comparison across DP configurations
    - Comparison across aggregation strategies
    - Comparison with centralized baselines
    - Statistical significance testing
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.experiments: List[ExperimentResult] = []
        self.baselines: Dict[str, ExperimentResult] = {}
    
    def add_experiment(self, result: ExperimentResult) -> None:
        """Add an experiment result."""
        self.experiments.append(result)
    
    def add_baseline(self, name: str, result: ExperimentResult) -> None:
        """Add a baseline result."""
        self.baselines[name] = result
    
    def compare_by_epsilon(self) -> Dict[str, Any]:
        """
        Compare experiments grouped by privacy epsilon.
        
        Returns:
            Comparison by epsilon levels
        """
        # Group by epsilon
        epsilon_groups: Dict[float, List[ExperimentResult]] = {}
        
        for exp in self.experiments:
            eps = exp.privacy.get('epsilon', float('inf'))
            eps_rounded = round(eps, 1)
            
            if eps_rounded not in epsilon_groups:
                epsilon_groups[eps_rounded] = []
            epsilon_groups[eps_rounded].append(exp)
        
        # Compute stats per group
        comparison = {}
        for eps, exps in sorted(epsilon_groups.items()):
            accs = [e.metrics.get('accuracy', 0) for e in exps]
            aucs = [e.metrics.get('auc_roc', 0) for e in exps]
            f1s = [e.metrics.get('f1', 0) for e in exps]
            
            comparison[eps] = {
                'n_experiments': len(exps),
                'accuracy': {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs)),
                    'min': float(np.min(accs)),
                    'max': float(np.max(accs))
                },
                'auc_roc': {
                    'mean': float(np.mean(aucs)),
                    'std': float(np.std(aucs))
                },
                'f1': {
                    'mean': float(np.mean(f1s)),
                    'std': float(np.std(f1s))
                }
            }
        
        return comparison
    
    def compare_by_aggregation(self) -> Dict[str, Any]:
        """
        Compare experiments by aggregation strategy.
        
        Returns:
            Comparison by aggregation method
        """
        # Group by aggregation
        agg_groups: Dict[str, List[ExperimentResult]] = {}
        
        for exp in self.experiments:
            agg = exp.config.get('aggregation_strategy', 'unknown')
            
            if agg not in agg_groups:
                agg_groups[agg] = []
            agg_groups[agg].append(exp)
        
        # Compute stats per group
        comparison = {}
        for agg, exps in agg_groups.items():
            accs = [e.metrics.get('accuracy', 0) for e in exps]
            aucs = [e.metrics.get('auc_roc', 0) for e in exps]
            times = [e.training_time for e in exps]
            
            comparison[agg] = {
                'n_experiments': len(exps),
                'accuracy': {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs))
                },
                'auc_roc': {
                    'mean': float(np.mean(aucs)),
                    'std': float(np.std(aucs))
                },
                'training_time': {
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times))
                }
            }
        
        return comparison
    
    def compare_to_baseline(self, baseline_name: str = 'centralized') -> Dict[str, Any]:
        """
        Compare federated experiments to baseline.
        
        Args:
            baseline_name: Name of baseline to compare against
            
        Returns:
            Comparison with baseline
        """
        if baseline_name not in self.baselines:
            return {'error': f'Baseline {baseline_name} not found'}
        
        baseline = self.baselines[baseline_name]
        baseline_acc = baseline.metrics.get('accuracy', 0)
        baseline_auc = baseline.metrics.get('auc_roc', 0)
        
        comparisons = []
        for exp in self.experiments:
            exp_acc = exp.metrics.get('accuracy', 0)
            exp_auc = exp.metrics.get('auc_roc', 0)
            
            comparisons.append({
                'name': exp.name,
                'accuracy_gap': baseline_acc - exp_acc,
                'auc_gap': baseline_auc - exp_auc,
                'accuracy_ratio': exp_acc / baseline_acc if baseline_acc > 0 else 0,
                'auc_ratio': exp_auc / baseline_auc if baseline_auc > 0 else 0,
                'epsilon': exp.privacy.get('epsilon', float('inf'))
            })
        
        # Summary statistics
        acc_gaps = [c['accuracy_gap'] for c in comparisons]
        auc_gaps = [c['auc_gap'] for c in comparisons]
        
        return {
            'baseline': baseline.to_dict(),
            'comparisons': comparisons,
            'summary': {
                'mean_accuracy_gap': float(np.mean(acc_gaps)),
                'mean_auc_gap': float(np.mean(auc_gaps)),
                'best_federated_accuracy': max(
                    e.metrics.get('accuracy', 0) for e in self.experiments
                ),
                'best_federated_auc': max(
                    e.metrics.get('auc_roc', 0) for e in self.experiments
                )
            }
        }
    
    def compute_pareto_frontier(self,
                                 privacy_metric: str = 'epsilon',
                                 utility_metric: str = 'auc_roc'
                                 ) -> List[ExperimentResult]:
        """
        Compute Pareto frontier of privacy-utility tradeoff.
        
        Args:
            privacy_metric: Privacy metric (lower is better)
            utility_metric: Utility metric (higher is better)
            
        Returns:
            Pareto-optimal experiments
        """
        # Sort by privacy (ascending) 
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: x.privacy.get(privacy_metric, float('inf'))
        )
        
        pareto = []
        max_utility = -float('inf')
        
        for exp in sorted_exps:
            utility = exp.metrics.get(utility_metric, 0)
            if utility > max_utility:
                pareto.append(exp)
                max_utility = utility
        
        return pareto
    
    def statistical_comparison(self,
                               exp1_results: List[float],
                               exp2_results: List[float]
                               ) -> Dict[str, Any]:
        """
        Statistical comparison between two experiment groups.
        
        Args:
            exp1_results: Results from first group
            exp2_results: Results from second group
            
        Returns:
            Statistical comparison
        """
        from scipy import stats
        
        # T-test
        t_stat, t_pvalue = stats.ttest_ind(exp1_results, exp2_results)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(
            exp1_results, exp2_results, alternative='two-sided'
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(exp1_results)**2 + np.std(exp2_results)**2) / 2
        )
        cohens_d = (np.mean(exp1_results) - np.mean(exp2_results)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_test': {'statistic': t_stat, 'p_value': t_pvalue},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
            'cohens_d': cohens_d,
            'significant_005': t_pvalue < 0.05,
            'significant_001': t_pvalue < 0.01
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Returns:
            Complete analysis report
        """
        report = {
            'summary': {
                'n_experiments': len(self.experiments),
                'n_baselines': len(self.baselines)
            },
            'best_results': {},
            'epsilon_comparison': self.compare_by_epsilon(),
            'aggregation_comparison': self.compare_by_aggregation()
        }
        
        # Find best results
        if self.experiments:
            best_acc = max(self.experiments, key=lambda x: x.metrics.get('accuracy', 0))
            best_auc = max(self.experiments, key=lambda x: x.metrics.get('auc_roc', 0))
            best_f1 = max(self.experiments, key=lambda x: x.metrics.get('f1', 0))
            
            report['best_results'] = {
                'accuracy': {'value': best_acc.metrics.get('accuracy', 0), 'experiment': best_acc.name},
                'auc_roc': {'value': best_auc.metrics.get('auc_roc', 0), 'experiment': best_auc.name},
                'f1': {'value': best_f1.metrics.get('f1', 0), 'experiment': best_f1.name}
            }
        
        # Baseline comparisons
        if self.baselines:
            report['baseline_comparisons'] = {}
            for baseline_name in self.baselines:
                report['baseline_comparisons'][baseline_name] = self.compare_to_baseline(baseline_name)
        
        # Pareto frontier
        pareto = self.compute_pareto_frontier()
        report['pareto_frontier'] = [
            {'name': e.name, 'epsilon': e.privacy.get('epsilon', 0), 'auc': e.metrics.get('auc_roc', 0)}
            for e in pareto
        ]
        
        return report
    
    def save_report(self, path: str) -> None:
        """Save report to file."""
        report = self.generate_report()
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


class AblationStudy:
    """
    Systematic ablation study for federated learning components.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        """
        Initialize ablation study.
        
        Args:
            base_config: Base configuration
        """
        self.base_config = base_config
        self.ablations: Dict[str, List[ExperimentResult]] = {}
    
    def add_ablation(self, 
                     component: str, 
                     results: List[ExperimentResult]) -> None:
        """
        Add ablation results for a component.
        
        Args:
            component: Component name
            results: Results with different values
        """
        self.ablations[component] = results
    
    def analyze_component(self, component: str) -> Dict[str, Any]:
        """
        Analyze impact of a component.
        
        Args:
            component: Component to analyze
            
        Returns:
            Ablation analysis
        """
        if component not in self.ablations:
            return {'error': f'Component {component} not found'}
        
        results = self.ablations[component]
        
        # Extract values and metrics
        values = []
        accs = []
        aucs = []
        
        for exp in results:
            value = exp.config.get(component)
            values.append(value)
            accs.append(exp.metrics.get('accuracy', 0))
            aucs.append(exp.metrics.get('auc_roc', 0))
        
        # Compute sensitivity
        acc_range = max(accs) - min(accs) if accs else 0
        auc_range = max(aucs) - min(aucs) if aucs else 0
        
        return {
            'component': component,
            'values_tested': values,
            'accuracy_range': acc_range,
            'auc_range': auc_range,
            'best_value': values[np.argmax(aucs)] if aucs else None,
            'sensitivity': 'high' if auc_range > 0.05 else 'moderate' if auc_range > 0.02 else 'low',
            'results': [
                {
                    'value': v,
                    'accuracy': a,
                    'auc': u
                }
                for v, a, u in zip(values, accs, aucs)
            ]
        }
    
    def get_importance_ranking(self) -> List[Dict[str, Any]]:
        """
        Rank components by importance.
        
        Returns:
            Ranked components
        """
        rankings = []
        
        for component in self.ablations:
            analysis = self.analyze_component(component)
            rankings.append({
                'component': component,
                'auc_range': analysis['auc_range'],
                'sensitivity': analysis['sensitivity']
            })
        
        # Sort by impact
        rankings.sort(key=lambda x: x['auc_range'], reverse=True)
        
        return rankings
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate ablation study report."""
        return {
            'base_config': self.base_config,
            'components_studied': list(self.ablations.keys()),
            'importance_ranking': self.get_importance_ranking(),
            'component_analyses': {
                comp: self.analyze_component(comp)
                for comp in self.ablations
            }
        }


class CrossValidationAnalyzer:
    """
    Cross-validation analysis for federated learning.
    """
    
    def __init__(self, n_folds: int = 5):
        """
        Initialize CV analyzer.
        
        Args:
            n_folds: Number of CV folds
        """
        self.n_folds = n_folds
        self.fold_results: List[Dict[str, float]] = []
    
    def add_fold_result(self, metrics: Dict[str, float]) -> None:
        """Add results from a fold."""
        self.fold_results.append(metrics)
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute CV statistics.
        
        Returns:
            CV statistics
        """
        if not self.fold_results:
            return {'error': 'No fold results'}
        
        metrics = {}
        for key in self.fold_results[0].keys():
            values = [f[key] for f in self.fold_results if key in f]
            metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_95': (
                    float(np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values))),
                    float(np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)))
                )
            }
        
        return {
            'n_folds': len(self.fold_results),
            'metrics': metrics
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Comparative Analysis Demonstration")
    print("=" * 60)
    
    # Create synthetic experiment results
    analyzer = ComparativeAnalyzer()
    
    # Add experiments with different epsilon values
    for i, eps in enumerate([0.5, 1.0, 2.0, 4.0, 8.0]):
        # Higher epsilon -> better utility, worse privacy
        acc = 0.75 + 0.05 * np.log(1 + eps) + np.random.randn() * 0.02
        auc = 0.80 + 0.04 * np.log(1 + eps) + np.random.randn() * 0.02
        f1 = 0.70 + 0.05 * np.log(1 + eps) + np.random.randn() * 0.02
        
        result = ExperimentResult(
            name=f'federated_eps_{eps}',
            config={'epsilon': eps, 'aggregation_strategy': 'fedavg'},
            metrics={'accuracy': acc, 'auc_roc': auc, 'f1': f1},
            privacy={'epsilon': eps, 'delta': 1e-5},
            training_time=100 + np.random.randn() * 10,
            n_rounds=50
        )
        analyzer.add_experiment(result)
    
    # Add different aggregation strategies
    for agg in ['fedavg', 'median', 'trimmed_mean']:
        acc = 0.78 + np.random.randn() * 0.02
        auc = 0.82 + np.random.randn() * 0.02
        
        result = ExperimentResult(
            name=f'federated_{agg}',
            config={'epsilon': 1.0, 'aggregation_strategy': agg},
            metrics={'accuracy': acc, 'auc_roc': auc, 'f1': 0.75},
            privacy={'epsilon': 1.0, 'delta': 1e-5},
            training_time=100,
            n_rounds=50
        )
        analyzer.add_experiment(result)
    
    # Add baseline
    baseline = ExperimentResult(
        name='centralized',
        config={'centralized': True},
        metrics={'accuracy': 0.85, 'auc_roc': 0.88, 'f1': 0.82},
        privacy={'epsilon': float('inf'), 'delta': 0},
        training_time=50,
        n_rounds=100
    )
    analyzer.add_baseline('centralized', baseline)
    
    # Generate report
    report = analyzer.generate_report()
    
    print("\nComparison by Epsilon:")
    for eps, stats in report['epsilon_comparison'].items():
        print(f"  ε={eps}: acc={stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
    
    print("\nComparison by Aggregation:")
    for agg, stats in report['aggregation_comparison'].items():
        print(f"  {agg}: acc={stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
    
    print("\nPareto Frontier:")
    for point in report['pareto_frontier']:
        print(f"  {point['name']}: ε={point['epsilon']:.1f}, AUC={point['auc']:.4f}")
    
    print("\nBest Results:")
    for metric, info in report['best_results'].items():
        print(f"  {metric}: {info['value']:.4f} ({info['experiment']})")
