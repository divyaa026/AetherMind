#!/usr/bin/env python3
"""
Visualization utilities for federated learning experiments.
Generates plots for training curves, privacy-utility tradeoffs, and results comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def set_plot_style():
    """Set consistent plot style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_training_curves(metrics_file: str, output_dir: str):
    """
    Plot training curves from experiment metrics.
    
    Args:
        metrics_file: Path to training_metrics.json
        output_dir: Directory to save plots
    """
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rounds = [m['round'] for m in metrics]
    accuracy = [m['accuracy'] for m in metrics]
    auc_roc = [m['auc_roc'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    
    # Plot accuracy and AUC over rounds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(rounds, accuracy, 'b-o', linewidth=2, markersize=6, label='Accuracy')
    axes[0].fill_between(rounds, [a * 0.95 for a in accuracy], [min(a * 1.05, 1.0) for a in accuracy], 
                         alpha=0.2, color='blue')
    axes[0].set_xlabel('Federated Round')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Test Accuracy vs. Federated Rounds')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC-ROC plot
    axes[1].plot(rounds, auc_roc, 'g-s', linewidth=2, markersize=6, label='AUC-ROC')
    axes[1].plot(rounds, f1, 'r-^', linewidth=2, markersize=6, label='F1 Score')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    axes[1].set_xlabel('Federated Round')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Classification Metrics vs. Federated Rounds')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png')
    plt.savefig(output_path / 'training_curves.pdf')
    print(f"Saved training curves to {output_path}")
    plt.close()


def plot_privacy_utility_tradeoff(experiments: List[Dict], output_dir: str):
    """
    Plot privacy-utility tradeoff for different epsilon values.
    
    Args:
        experiments: List of experiment results with different epsilon values
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Example data structure expected:
    # [{'epsilon': 0.5, 'accuracy': 0.85, 'auc': 0.82}, ...]
    
    if not experiments:
        print("No experiment data provided for privacy-utility plot")
        return
    
    epsilons = [e['epsilon'] for e in experiments]
    accuracies = [e['accuracy'] for e in experiments]
    aucs = [e['auc_roc'] for e in experiments]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epsilons, accuracies, 'b-o', linewidth=2, markersize=8, label='Accuracy')
    ax.plot(epsilons, aucs, 'g-s', linewidth=2, markersize=8, label='AUC-ROC')
    
    ax.set_xlabel('Privacy Budget (epsilon)')
    ax.set_ylabel('Performance')
    ax.set_title('Privacy-Utility Tradeoff')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, eps in enumerate(epsilons):
        ax.annotate(f'eps={eps}', (epsilons[i], accuracies[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'privacy_utility_tradeoff.png')
    plt.savefig(output_path / 'privacy_utility_tradeoff.pdf')
    print(f"Saved privacy-utility plot to {output_path}")
    plt.close()


def plot_client_data_distribution(partitions: Dict, output_dir: str):
    """
    Plot data distribution across federated clients.
    
    Args:
        partitions: Dict with client_id -> n_samples
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    clients = list(range(len(partitions)))
    samples = [partitions[i] for i in clients]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(clients)))
    bars = ax.bar(clients, samples, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Data Distribution Across Federated Clients')
    ax.set_xticks(clients)
    
    # Add value labels on bars
    for bar, val in zip(bars, samples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
               str(val), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'client_distribution.png')
    plt.savefig(output_path / 'client_distribution.pdf')
    print(f"Saved client distribution plot to {output_path}")
    plt.close()


def plot_convergence_comparison(experiments: Dict[str, List], output_dir: str):
    """
    Compare convergence of different aggregation strategies.
    
    Args:
        experiments: Dict mapping strategy name to list of round metrics
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'fedavg': 'blue', 'fedavg_momentum': 'green', 'median': 'red', 'trimmed_mean': 'purple'}
    
    for strategy, metrics in experiments.items():
        rounds = [m['round'] for m in metrics]
        accuracy = [m['accuracy'] for m in metrics]
        color = colors.get(strategy, 'gray')
        ax.plot(rounds, accuracy, '-o', linewidth=2, markersize=5, 
               label=strategy.replace('_', ' ').title(), color=color)
    
    ax.set_xlabel('Federated Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Convergence Comparison of Aggregation Strategies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'convergence_comparison.png')
    plt.savefig(output_path / 'convergence_comparison.pdf')
    print(f"Saved convergence comparison to {output_path}")
    plt.close()


def generate_results_table(experiments: List[Dict], output_dir: str):
    """
    Generate LaTeX table with experiment results.
    
    Args:
        experiments: List of experiment results
        output_dir: Directory to save table
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Federated Learning Experiment Results}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
\textbf{Experiment} & \textbf{Epsilon} & \textbf{Accuracy} & \textbf{AUC-ROC} & \textbf{F1} & \textbf{MCC} \\
\midrule
"""
    
    for exp in experiments:
        name = exp.get('name', 'Unknown')
        epsilon = exp.get('epsilon', 'N/A')
        acc = exp.get('accuracy', 0)
        auc = exp.get('auc_roc', 0)
        f1 = exp.get('f1', 0)
        mcc = exp.get('mcc', 0)
        
        latex += f"{name} & {epsilon} & {acc:.4f} & {auc:.4f} & {f1:.4f} & {mcc:.4f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path / 'results_table.tex', 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX table to {output_path / 'results_table.tex'}")


def visualize_experiment(experiment_dir: str, output_dir: Optional[str] = None):
    """
    Generate all visualizations for an experiment.
    
    Args:
        experiment_dir: Directory containing experiment results
        output_dir: Output directory for plots (default: {experiment_dir}/visualizations)
    """
    set_plot_style()
    
    exp_path = Path(experiment_dir)
    out_path = Path(output_dir) if output_dir else exp_path / 'visualizations'
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load experiment results
    results_file = exp_path / 'experiment_results.json'
    metrics_file = exp_path / 'training_metrics.json'
    
    if metrics_file.exists():
        plot_training_curves(str(metrics_file), str(out_path))
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Plot client distribution if available
        if 'phases' in results and 'partitioning' in results['phases']:
            samples = results['phases']['partitioning'].get('samples_per_client', [])
            if samples:
                partitions = {i: s for i, s in enumerate(samples)}
                plot_client_data_distribution(partitions, str(out_path))
        
        # Generate summary
        summary = {
            'experiment': results.get('experiment_name', 'Unknown'),
            'timestamp': results.get('timestamp', 'Unknown'),
            'final_metrics': results.get('final_metrics', {})
        }
        
        with open(out_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment Summary:")
        print(f"  Name: {summary['experiment']}")
        print(f"  Timestamp: {summary['timestamp']}")
        if summary['final_metrics']:
            print(f"  Final Accuracy: {summary['final_metrics'].get('accuracy', 'N/A'):.4f}")
            print(f"  Final AUC-ROC: {summary['final_metrics'].get('auc_roc', 'N/A'):.4f}")
    
    print(f"\nAll visualizations saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate visualizations for federated learning experiments')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualize_experiment(args.experiment_dir, args.output_dir)
