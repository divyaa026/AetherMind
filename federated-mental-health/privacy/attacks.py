"""
Privacy attack implementations and defenses.
Used for evaluating the privacy guarantees of federated learning.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import copy


class AttackType(Enum):
    """Types of privacy attacks."""
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_INVERSION = "model_inversion"
    GRADIENT_LEAKAGE = "gradient_leakage"
    ATTRIBUTE_INFERENCE = "attribute_inference"


@dataclass
class AttackResult:
    """Result of a privacy attack."""
    attack_type: AttackType
    success_rate: float  # Attack success rate
    auc: float  # Area under ROC curve
    advantage: float  # Attacker advantage over random guessing
    details: Dict[str, Any]


class MembershipInferenceAttack:
    """
    Membership Inference Attack.
    
    Determines whether a sample was used in training.
    Based on Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017)
    """
    
    def __init__(self, 
                 attack_model: str = 'logistic',
                 use_loss: bool = True,
                 use_confidence: bool = True,
                 use_entropy: bool = True):
        """
        Initialize membership inference attack.
        
        Args:
            attack_model: Type of attack model ('logistic', 'rf')
            use_loss: Use loss values as features
            use_confidence: Use prediction confidence as features
            use_entropy: Use prediction entropy as features
        """
        self.attack_model_type = attack_model
        self.use_loss = use_loss
        self.use_confidence = use_confidence
        self.use_entropy = use_entropy
        
        self.attack_model = None
        self.is_trained = False
    
    def _extract_features(self, 
                         model: nn.Module,
                         X: np.ndarray,
                         y: np.ndarray) -> np.ndarray:
        """
        Extract attack features from model predictions.
        
        Args:
            model: Target model
            X: Input data
            y: True labels
            
        Returns:
            Attack features
        """
        model.eval()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        with torch.no_grad():
            logits, _ = model(X_tensor)
            probs = torch.sigmoid(logits).squeeze()
        
        features = []
        
        # Confidence (prediction probability)
        if self.use_confidence:
            confidence = torch.where(
                y_tensor == 1, probs, 1 - probs
            ).numpy()
            features.append(confidence.reshape(-1, 1))
        
        # Loss
        if self.use_loss:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(), y_tensor, reduction='none'
            ).numpy()
            features.append(loss.reshape(-1, 1))
        
        # Entropy
        if self.use_entropy:
            eps = 1e-7
            probs_np = probs.numpy()
            probs_np = np.clip(probs_np, eps, 1 - eps)
            entropy = -probs_np * np.log(probs_np) - (1 - probs_np) * np.log(1 - probs_np)
            features.append(entropy.reshape(-1, 1))
        
        return np.hstack(features)
    
    def train_attack(self,
                     target_model: nn.Module,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_holdout: np.ndarray,
                     y_holdout: np.ndarray) -> None:
        """
        Train the attack model.
        
        Args:
            target_model: Model to attack
            X_train: Training data (members)
            y_train: Training labels
            X_holdout: Holdout data (non-members)
            y_holdout: Holdout labels
        """
        # Extract features for members (training data)
        member_features = self._extract_features(target_model, X_train, y_train)
        member_labels = np.ones(len(X_train))
        
        # Extract features for non-members (holdout data)
        non_member_features = self._extract_features(target_model, X_holdout, y_holdout)
        non_member_labels = np.zeros(len(X_holdout))
        
        # Combine
        attack_X = np.vstack([member_features, non_member_features])
        attack_y = np.concatenate([member_labels, non_member_labels])
        
        # Shuffle
        indices = np.random.permutation(len(attack_y))
        attack_X = attack_X[indices]
        attack_y = attack_y[indices]
        
        # Train attack model
        if self.attack_model_type == 'logistic':
            self.attack_model = LogisticRegression(random_state=42)
        else:
            self.attack_model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
        
        self.attack_model.fit(attack_X, attack_y)
        self.is_trained = True
    
    def attack(self,
               target_model: nn.Module,
               X: np.ndarray,
               y: np.ndarray,
               is_member: np.ndarray) -> AttackResult:
        """
        Execute membership inference attack.
        
        Args:
            target_model: Model to attack
            X: Data to test
            y: True labels
            is_member: Ground truth membership
            
        Returns:
            Attack results
        """
        if not self.is_trained:
            raise RuntimeError("Attack model not trained")
        
        # Extract features
        features = self._extract_features(target_model, X, y)
        
        # Predict membership
        predictions = self.attack_model.predict(features)
        probabilities = self.attack_model.predict_proba(features)[:, 1]
        
        # Compute metrics
        accuracy = accuracy_score(is_member, predictions)
        auc = roc_auc_score(is_member, probabilities)
        advantage = 2 * accuracy - 1  # Advantage over random guessing
        
        return AttackResult(
            attack_type=AttackType.MEMBERSHIP_INFERENCE,
            success_rate=accuracy,
            auc=auc,
            advantage=advantage,
            details={
                'n_samples': len(X),
                'n_members': int(is_member.sum()),
                'n_non_members': int((1 - is_member).sum()),
                'features_used': {
                    'loss': self.use_loss,
                    'confidence': self.use_confidence,
                    'entropy': self.use_entropy
                }
            }
        )


class GradientLeakageAttack:
    """
    Gradient Leakage Attack.
    
    Attempts to reconstruct training data from gradients.
    Based on Zhu et al., "Deep Leakage from Gradients" (2019)
    """
    
    def __init__(self,
                 n_iterations: int = 300,
                 learning_rate: float = 1.0,
                 tv_weight: float = 0.001):
        """
        Initialize gradient leakage attack.
        
        Args:
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for reconstruction
            tv_weight: Weight for total variation regularization
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.tv_weight = tv_weight
    
    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation for regularization."""
        diff1 = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff2 = x[:, :, :, 1:] - x[:, :, :, :-1]
        return torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2))
    
    def _compute_gradient(self,
                          model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor) -> List[torch.Tensor]:
        """Compute gradients for input."""
        model.zero_grad()
        
        logits, _ = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), y
        )
        
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        return list(grads)
    
    def attack(self,
               model: nn.Module,
               target_gradients: List[torch.Tensor],
               input_shape: Tuple[int, ...],
               true_x: Optional[torch.Tensor] = None) -> AttackResult:
        """
        Attempt to reconstruct training data from gradients.
        
        Args:
            model: Target model
            target_gradients: Gradients to attack
            input_shape: Shape of input to reconstruct
            true_x: True input (for evaluation)
            
        Returns:
            Attack results
        """
        device = next(model.parameters()).device
        
        # Initialize random input and label
        dummy_x = torch.randn(*input_shape, device=device, requires_grad=True)
        dummy_y = torch.randn(input_shape[0], device=device, requires_grad=True)
        
        optimizer = torch.optim.LBFGS([dummy_x, dummy_y], lr=self.learning_rate)
        
        history = []
        
        for i in range(self.n_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Compute gradients for dummy input
                dummy_grads = self._compute_gradient(
                    model, dummy_x, torch.sigmoid(dummy_y)
                )
                
                # Match gradients
                grad_diff = 0
                for dg, tg in zip(dummy_grads, target_gradients):
                    grad_diff += torch.sum((dg - tg) ** 2)
                
                # Total variation regularization
                if len(input_shape) == 4:  # Image data
                    tv = self.tv_weight * self._total_variation(dummy_x)
                else:
                    tv = 0
                
                total_loss = grad_diff + tv
                total_loss.backward()
                
                return total_loss
            
            loss = optimizer.step(closure)
            history.append(loss.item())
        
        # Evaluate reconstruction
        if true_x is not None:
            mse = torch.mean((dummy_x - true_x) ** 2).item()
            psnr = 10 * np.log10(1 / (mse + 1e-10))
        else:
            mse = float('nan')
            psnr = float('nan')
        
        return AttackResult(
            attack_type=AttackType.GRADIENT_LEAKAGE,
            success_rate=0.0,  # Not directly applicable
            auc=0.0,
            advantage=0.0,
            details={
                'final_loss': history[-1],
                'mse': mse,
                'psnr': psnr,
                'reconstructed': dummy_x.detach().cpu().numpy(),
                'convergence_history': history
            }
        )


class AttributeInferenceAttack:
    """
    Attribute Inference Attack.
    
    Infers sensitive attributes about training data.
    """
    
    def __init__(self, n_shadow_models: int = 5):
        """
        Initialize attribute inference attack.
        
        Args:
            n_shadow_models: Number of shadow models to train
        """
        self.n_shadow_models = n_shadow_models
        self.attack_models: Dict[str, Any] = {}
    
    def train_attack(self,
                     shadow_predictions: List[np.ndarray],
                     shadow_attributes: List[np.ndarray],
                     attribute_name: str) -> None:
        """
        Train attack model for attribute inference.
        
        Args:
            shadow_predictions: Predictions from shadow models
            shadow_attributes: True attributes for shadow data
            attribute_name: Name of attribute to infer
        """
        # Combine shadow data
        all_preds = np.vstack(shadow_predictions)
        all_attrs = np.concatenate(shadow_attributes)
        
        # Train attack model
        attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
        attack_model.fit(all_preds, all_attrs)
        
        self.attack_models[attribute_name] = attack_model
    
    def attack(self,
               target_predictions: np.ndarray,
               true_attributes: np.ndarray,
               attribute_name: str) -> AttackResult:
        """
        Execute attribute inference attack.
        
        Args:
            target_predictions: Predictions from target model
            true_attributes: Ground truth attributes
            attribute_name: Attribute to infer
            
        Returns:
            Attack results
        """
        if attribute_name not in self.attack_models:
            raise ValueError(f"No attack model for attribute: {attribute_name}")
        
        attack_model = self.attack_models[attribute_name]
        
        # Predict attributes
        predictions = attack_model.predict(target_predictions)
        probabilities = attack_model.predict_proba(target_predictions)
        
        # Compute metrics
        accuracy = accuracy_score(true_attributes, predictions)
        
        # AUC only for binary attributes
        if len(np.unique(true_attributes)) == 2:
            auc = roc_auc_score(true_attributes, probabilities[:, 1])
        else:
            auc = 0.0
        
        return AttackResult(
            attack_type=AttackType.ATTRIBUTE_INFERENCE,
            success_rate=accuracy,
            auc=auc,
            advantage=accuracy - (1 / len(np.unique(true_attributes))),
            details={
                'attribute_name': attribute_name,
                'n_classes': len(np.unique(true_attributes))
            }
        )


class DPDefenseEvaluator:
    """
    Evaluates effectiveness of differential privacy defenses.
    """
    
    def __init__(self):
        """Initialize defense evaluator."""
        self.mia = MembershipInferenceAttack()
    
    def evaluate_dp_defense(self,
                           model_no_dp: nn.Module,
                           model_with_dp: nn.Module,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate DP defense by comparing attack success.
        
        Args:
            model_no_dp: Model trained without DP
            model_with_dp: Model trained with DP
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Comparison of attack success
        """
        results = {}
        
        # Split test data for attack training
        n = len(X_test) // 2
        X_holdout, X_attack = X_test[:n], X_test[n:]
        y_holdout, y_attack = y_test[:n], y_test[n:]
        
        # Attack model without DP
        self.mia.train_attack(
            model_no_dp,
            X_train[:n], y_train[:n],
            X_holdout, y_holdout
        )
        
        # Create membership labels
        is_member = np.concatenate([
            np.ones(min(n, len(X_train))),
            np.zeros(len(X_attack))
        ])
        X_combined = np.vstack([X_train[:n], X_attack])
        y_combined = np.concatenate([y_train[:n], y_attack])
        
        no_dp_result = self.mia.attack(model_no_dp, X_combined, y_combined, is_member)
        results['no_dp'] = {
            'accuracy': no_dp_result.success_rate,
            'auc': no_dp_result.auc,
            'advantage': no_dp_result.advantage
        }
        
        # Attack model with DP
        self.mia.train_attack(
            model_with_dp,
            X_train[:n], y_train[:n],
            X_holdout, y_holdout
        )
        
        dp_result = self.mia.attack(model_with_dp, X_combined, y_combined, is_member)
        results['with_dp'] = {
            'accuracy': dp_result.success_rate,
            'auc': dp_result.auc,
            'advantage': dp_result.advantage
        }
        
        # Compute protection
        results['protection'] = {
            'accuracy_reduction': results['no_dp']['accuracy'] - results['with_dp']['accuracy'],
            'auc_reduction': results['no_dp']['auc'] - results['with_dp']['auc'],
            'advantage_reduction': results['no_dp']['advantage'] - results['with_dp']['advantage']
        }
        
        return results


def run_privacy_audit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    delta: float
) -> Dict[str, Any]:
    """
    Run comprehensive privacy audit on a trained model.
    
    Args:
        model: Model to audit
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        epsilon: Privacy budget epsilon
        delta: Privacy budget delta
        
    Returns:
        Audit results
    """
    results = {
        'privacy_budget': {'epsilon': epsilon, 'delta': delta},
        'attacks': {}
    }
    
    # Membership inference attack
    mia = MembershipInferenceAttack()
    
    # Split data
    n = len(X_test) // 2
    X_holdout, X_attack = X_test[:n], X_test[n:]
    y_holdout, y_attack = y_test[:n], y_test[n:]
    
    # Train attack
    train_subset_size = min(n, len(X_train))
    mia.train_attack(
        model,
        X_train[:train_subset_size], y_train[:train_subset_size],
        X_holdout, y_holdout
    )
    
    # Execute attack
    is_member = np.concatenate([
        np.ones(train_subset_size),
        np.zeros(len(X_attack))
    ])
    X_combined = np.vstack([X_train[:train_subset_size], X_attack])
    y_combined = np.concatenate([y_train[:train_subset_size], y_attack])
    
    mia_result = mia.attack(model, X_combined, y_combined, is_member)
    
    results['attacks']['membership_inference'] = {
        'success_rate': mia_result.success_rate,
        'auc': mia_result.auc,
        'advantage': mia_result.advantage,
        'details': mia_result.details
    }
    
    # Privacy assessment
    results['assessment'] = {
        'mia_success': mia_result.success_rate > 0.6,
        'high_risk': mia_result.advantage > 0.2,
        'recommendation': 'PASS' if mia_result.advantage < 0.1 else 'REVIEW' if mia_result.advantage < 0.2 else 'FAIL'
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Privacy Attack Demonstration")
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
    
    # Create a simple model
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
    
    # Run privacy audit
    print("\nRunning privacy audit...")
    audit_results = run_privacy_audit(
        model, X_train, y_train, X_test, y_test,
        epsilon=1.0, delta=1e-5
    )
    
    print(f"\nMembership Inference Attack:")
    mia_results = audit_results['attacks']['membership_inference']
    print(f"  Success Rate: {mia_results['success_rate']:.4f}")
    print(f"  AUC: {mia_results['auc']:.4f}")
    print(f"  Advantage: {mia_results['advantage']:.4f}")
    
    print(f"\nPrivacy Assessment:")
    print(f"  Recommendation: {audit_results['assessment']['recommendation']}")
