"""
Privacy-specific evaluation for federated mental health prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import copy

import sys
sys.path.append('..')
from privacy.attacks import (
    MembershipInferenceAttack, 
    GradientLeakageAttack,
    DPDefenseEvaluator,
    run_privacy_audit
)
from privacy.dp_accounting import (
    RDPAccountant, 
    FederatedPrivacyAccountant,
    PrivacyBudget
)


@dataclass
class PrivacyEvaluationResult:
    """Container for privacy evaluation results."""
    epsilon: float
    delta: float
    membership_inference_accuracy: float
    membership_inference_advantage: float
    privacy_risk_level: str
    recommendations: List[str]
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'membership_inference_accuracy': self.membership_inference_accuracy,
            'membership_inference_advantage': self.membership_inference_advantage,
            'privacy_risk_level': self.privacy_risk_level,
            'recommendations': self.recommendations,
            'details': self.details
        }


class PrivacyEvaluator:
    """
    Comprehensive privacy evaluation for federated learning models.
    
    Evaluates:
    - Theoretical privacy guarantees (DP accounting)
    - Empirical privacy (attack simulations)
    - Privacy-utility tradeoffs
    """
    
    def __init__(self,
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5):
        """
        Initialize privacy evaluator.
        
        Args:
            target_epsilon: Target privacy budget epsilon
            target_delta: Target privacy budget delta
        """
        self.target = PrivacyBudget(target_epsilon, target_delta)
        self.mia = MembershipInferenceAttack()
        self.defense_evaluator = DPDefenseEvaluator()
    
    def evaluate_theoretical_privacy(self,
                                      noise_multiplier: float,
                                      sampling_probability: float,
                                      n_steps: int,
                                      n_epochs: int = 1
                                      ) -> Dict[str, Any]:
        """
        Evaluate theoretical privacy guarantees.
        
        Args:
            noise_multiplier: Noise multiplier σ/max_grad_norm
            sampling_probability: Batch sampling probability
            n_steps: Steps per epoch
            n_epochs: Number of training epochs
            
        Returns:
            Theoretical privacy analysis
        """
        accountant = RDPAccountant(target_delta=self.target.delta)
        
        total_steps = n_steps * n_epochs
        accountant.add_mechanism(
            noise_multiplier=noise_multiplier,
            sampling_probability=sampling_probability,
            n_steps=total_steps
        )
        
        epsilon, delta, order = accountant.get_privacy_spent()
        
        return {
            'epsilon': epsilon,
            'delta': delta,
            'optimal_rdp_order': order,
            'within_budget': epsilon <= self.target.epsilon,
            'remaining_epsilon': max(0, self.target.epsilon - epsilon),
            'parameters': {
                'noise_multiplier': noise_multiplier,
                'sampling_probability': sampling_probability,
                'total_steps': total_steps
            }
        }
    
    def evaluate_empirical_privacy(self,
                                    model: nn.Module,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_test: np.ndarray,
                                    y_test: np.ndarray
                                    ) -> Dict[str, Any]:
        """
        Evaluate empirical privacy via attacks.
        
        Args:
            model: Trained model to evaluate
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Empirical privacy evaluation
        """
        # Split test data
        n = min(len(X_test) // 2, len(X_train) // 2)
        X_holdout, X_attack = X_test[:n], X_test[n:2*n]
        y_holdout, y_attack = y_test[:n], y_test[n:2*n]
        
        # Train attack model
        self.mia.train_attack(
            model,
            X_train[:n], y_train[:n],
            X_holdout, y_holdout
        )
        
        # Execute attack
        is_member = np.concatenate([
            np.ones(n),
            np.zeros(len(X_attack))
        ])
        X_combined = np.vstack([X_train[:n], X_attack])
        y_combined = np.concatenate([y_train[:n], y_attack])
        
        result = self.mia.attack(model, X_combined, y_combined, is_member)
        
        return {
            'membership_inference': {
                'accuracy': result.success_rate,
                'auc': result.auc,
                'advantage': result.advantage,
                'details': result.details
            }
        }
    
    def compute_privacy_risk_level(self,
                                   mia_advantage: float,
                                   epsilon: float
                                   ) -> Tuple[str, List[str]]:
        """
        Determine privacy risk level and recommendations.
        
        Args:
            mia_advantage: Membership inference advantage
            epsilon: Privacy epsilon
            
        Returns:
            (risk_level, recommendations)
        """
        recommendations = []
        
        # Evaluate risk based on both empirical and theoretical metrics
        if mia_advantage < 0.05 and epsilon <= 1.0:
            risk_level = "LOW"
            recommendations.append("Privacy protection is strong")
        
        elif mia_advantage < 0.1 and epsilon <= 3.0:
            risk_level = "MODERATE"
            if mia_advantage >= 0.05:
                recommendations.append("Consider increasing noise multiplier")
            if epsilon > 1.0:
                recommendations.append("Consider reducing training epochs")
        
        elif mia_advantage < 0.2 and epsilon <= 8.0:
            risk_level = "ELEVATED"
            recommendations.append("Increase DP noise significantly")
            recommendations.append("Reduce batch size or training epochs")
            recommendations.append("Consider data augmentation")
        
        else:
            risk_level = "HIGH"
            recommendations.append("Privacy protection is insufficient")
            recommendations.append("Substantially increase noise multiplier")
            recommendations.append("Consider using more aggressive DP parameters")
            recommendations.append("Reduce model complexity")
            recommendations.append("Consider training with fewer epochs")
        
        return risk_level, recommendations
    
    def evaluate(self,
                 model: nn.Module,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 dp_params: Optional[Dict[str, float]] = None
                 ) -> PrivacyEvaluationResult:
        """
        Comprehensive privacy evaluation.
        
        Args:
            model: Model to evaluate
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            dp_params: DP parameters (noise_multiplier, sampling_prob, n_steps)
            
        Returns:
            Privacy evaluation result
        """
        results = {}
        
        # Theoretical evaluation
        if dp_params:
            theoretical = self.evaluate_theoretical_privacy(
                noise_multiplier=dp_params.get('noise_multiplier', 1.0),
                sampling_probability=dp_params.get('sampling_probability', 0.01),
                n_steps=dp_params.get('n_steps', 1000)
            )
            results['theoretical'] = theoretical
            epsilon = theoretical['epsilon']
        else:
            epsilon = float('inf')
        
        # Empirical evaluation
        empirical = self.evaluate_empirical_privacy(
            model, X_train, y_train, X_test, y_test
        )
        results['empirical'] = empirical
        
        mia_advantage = empirical['membership_inference']['advantage']
        mia_accuracy = empirical['membership_inference']['accuracy']
        
        # Risk assessment
        risk_level, recommendations = self.compute_privacy_risk_level(
            mia_advantage, epsilon
        )
        
        return PrivacyEvaluationResult(
            epsilon=epsilon,
            delta=self.target.delta,
            membership_inference_accuracy=mia_accuracy,
            membership_inference_advantage=mia_advantage,
            privacy_risk_level=risk_level,
            recommendations=recommendations,
            details=results
        )


class PrivacyUtilityTradeoff:
    """
    Analyzes privacy-utility tradeoffs across different DP configurations.
    """
    
    def __init__(self):
        """Initialize tradeoff analyzer."""
        self.results: List[Dict] = []
    
    def add_result(self,
                   epsilon: float,
                   accuracy: float,
                   f1: float,
                   auc: float,
                   mia_advantage: float,
                   config: Dict[str, Any]) -> None:
        """
        Add a result point to the analysis.
        
        Args:
            epsilon: Privacy epsilon
            accuracy: Model accuracy
            f1: Model F1 score
            auc: Model AUC
            mia_advantage: MIA advantage
            config: Configuration used
        """
        self.results.append({
            'epsilon': epsilon,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'mia_advantage': mia_advantage,
            'config': config
        })
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the privacy-utility tradeoff.
        
        Returns:
            Tradeoff analysis
        """
        if not self.results:
            return {'error': 'No results to analyze'}
        
        # Sort by epsilon
        sorted_results = sorted(self.results, key=lambda x: x['epsilon'])
        
        # Find Pareto-optimal points
        pareto_optimal = []
        max_utility = -float('inf')
        
        for result in sorted_results:
            utility = result['auc']  # Use AUC as utility metric
            if utility > max_utility:
                pareto_optimal.append(result)
                max_utility = utility
        
        # Compute tradeoff metrics
        epsilons = [r['epsilon'] for r in sorted_results]
        aucs = [r['auc'] for r in sorted_results]
        
        # Privacy-utility slope
        if len(epsilons) > 1 and epsilons[-1] != epsilons[0]:
            slope = (aucs[-1] - aucs[0]) / (epsilons[-1] - epsilons[0])
        else:
            slope = 0
        
        # Optimal operating point (best F1 with epsilon <= 1)
        low_epsilon_results = [r for r in sorted_results if r['epsilon'] <= 1.0]
        if low_epsilon_results:
            optimal = max(low_epsilon_results, key=lambda x: x['f1'])
        else:
            optimal = max(sorted_results, key=lambda x: x['f1'])
        
        return {
            'all_results': sorted_results,
            'pareto_optimal': pareto_optimal,
            'optimal_point': optimal,
            'privacy_utility_slope': slope,
            'epsilon_range': (min(epsilons), max(epsilons)),
            'auc_range': (min(aucs), max(aucs)),
            'n_experiments': len(self.results)
        }
    
    def recommend_config(self, 
                         max_epsilon: float = 1.0,
                         min_auc: float = 0.7) -> Optional[Dict]:
        """
        Recommend configuration meeting constraints.
        
        Args:
            max_epsilon: Maximum acceptable epsilon
            min_auc: Minimum acceptable AUC
            
        Returns:
            Recommended configuration or None
        """
        valid_results = [
            r for r in self.results
            if r['epsilon'] <= max_epsilon and r['auc'] >= min_auc
        ]
        
        if not valid_results:
            return None
        
        # Return best F1 among valid
        return max(valid_results, key=lambda x: x['f1'])


class FederatedPrivacyEvaluator:
    """
    Privacy evaluation for federated learning scenarios.
    """
    
    def __init__(self,
                 n_clients: int,
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5):
        """
        Initialize federated privacy evaluator.
        
        Args:
            n_clients: Number of federated clients
            target_epsilon: Target privacy budget
            target_delta: Target delta
        """
        self.n_clients = n_clients
        self.target = PrivacyBudget(target_epsilon, target_delta)
        
        self.accountant = FederatedPrivacyAccountant(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            n_clients=n_clients
        )
    
    def evaluate_round(self,
                       participating_clients: List[int],
                       client_noise_multiplier: float,
                       server_noise_multiplier: float,
                       n_local_steps: int,
                       batch_size: int,
                       client_dataset_sizes: Dict[int, int]) -> Dict[str, Any]:
        """
        Evaluate privacy for a federated round.
        
        Args:
            participating_clients: Clients in this round
            client_noise_multiplier: Client-side noise
            server_noise_multiplier: Server-side noise
            n_local_steps: Local training steps
            batch_size: Batch size
            client_dataset_sizes: Dataset sizes per client
            
        Returns:
            Privacy evaluation for round
        """
        self.accountant.add_round(
            client_noise_multiplier=client_noise_multiplier,
            server_noise_multiplier=server_noise_multiplier,
            n_local_steps=n_local_steps,
            batch_size=batch_size,
            client_dataset_sizes=client_dataset_sizes,
            participating_clients=participating_clients
        )
        
        eps, delta = self.accountant.get_total_privacy()
        
        return {
            'total_epsilon': eps,
            'total_delta': delta,
            'within_budget': self.accountant.is_budget_available(),
            'remaining_epsilon': max(0, self.target.epsilon - eps)
        }
    
    def get_per_client_privacy(self) -> Dict[int, Dict]:
        """Get privacy spent per client."""
        return {
            client_id: {
                'epsilon': eps,
                'delta': delta
            }
            for client_id, (eps, delta) in [
                (cid, self.accountant.get_client_privacy(cid))
                for cid in self.accountant.client_accountants.keys()
            ]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete privacy summary."""
        return self.accountant.get_summary()


def run_comprehensive_privacy_evaluation(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dp_params: Dict[str, float],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive privacy evaluation.
    
    Args:
        model: Model to evaluate
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        dp_params: DP parameters
        output_path: Path to save results
        
    Returns:
        Comprehensive privacy evaluation
    """
    evaluator = PrivacyEvaluator(
        target_epsilon=dp_params.get('target_epsilon', 1.0),
        target_delta=dp_params.get('target_delta', 1e-5)
    )
    
    result = evaluator.evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dp_params=dp_params
    )
    
    output = result.to_dict()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
    
    return output


if __name__ == "__main__":
    import torch.nn as nn
    
    print("=" * 60)
    print("Privacy Evaluation Demonstration")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_train = 500
    n_test = 200
    input_dim = 20
    seq_len = 7
    
    X_train = np.random.randn(n_train, seq_len, input_dim).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train).astype(np.float32)
    
    X_test = np.random.randn(n_test, seq_len, input_dim).astype(np.float32)
    y_test = np.random.randint(0, 2, n_test).astype(np.float32)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1]), None
    
    model = SimpleModel(input_dim)
    
    # Train briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        logits, _ = model(torch.FloatTensor(X_train))
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), torch.FloatTensor(y_train)
        )
        loss.backward()
        optimizer.step()
    
    # Evaluate privacy
    print("\nEvaluating privacy...")
    
    evaluator = PrivacyEvaluator(target_epsilon=1.0, target_delta=1e-5)
    
    result = evaluator.evaluate(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dp_params={
            'noise_multiplier': 1.0,
            'sampling_probability': 0.01,
            'n_steps': 1000
        }
    )
    
    print(f"\nPrivacy Evaluation Results:")
    print(f"  Epsilon: {result.epsilon:.4f}")
    print(f"  Delta: {result.delta:.2e}")
    print(f"  MIA Accuracy: {result.membership_inference_accuracy:.4f}")
    print(f"  MIA Advantage: {result.membership_inference_advantage:.4f}")
    print(f"  Risk Level: {result.privacy_risk_level}")
    print(f"  Recommendations:")
    for rec in result.recommendations:
        print(f"    - {rec}")
