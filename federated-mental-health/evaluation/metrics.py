"""
Comprehensive evaluation metrics for federated mental health prediction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import warnings


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    mcc: float  # Matthews Correlation Coefficient
    kappa: float  # Cohen's Kappa
    specificity: float
    npv: float  # Negative Predictive Value
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'mcc': self.mcc,
            'kappa': self.kappa,
            'specificity': self.specificity,
            'npv': self.npv,
            'confusion_matrix': self.confusion_matrix.tolist()
        }


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    reliability_diagram: Dict[str, np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ece': self.ece,
            'mce': self.mce,
            'brier_score': self.brier_score,
            'reliability_diagram': {
                k: v.tolist() for k, v in self.reliability_diagram.items()
            }
        }


@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""
    demographic_parity: float
    equalized_odds: float
    predictive_parity: float
    individual_fairness: float
    group_metrics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'demographic_parity': self.demographic_parity,
            'equalized_odds': self.equalized_odds,
            'predictive_parity': self.predictive_parity,
            'individual_fairness': self.individual_fairness,
            'group_metrics': self.group_metrics
        }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for mental health prediction.
    
    Computes:
    - Classification metrics (accuracy, F1, AUC, etc.)
    - Calibration metrics (ECE, reliability)
    - Fairness metrics (demographic parity, equalized odds)
    - Clinical relevance metrics
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 n_calibration_bins: int = 10):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Classification threshold
            n_calibration_bins: Number of bins for calibration
        """
        self.threshold = threshold
        self.n_calibration_bins = n_calibration_bins
    
    def compute_classification_metrics(self,
                                        y_true: np.ndarray,
                                        y_prob: np.ndarray,
                                        threshold: Optional[float] = None
                                        ) -> ClassificationMetrics:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Classification metrics
        """
        thresh = threshold or self.threshold
        y_pred = (y_prob >= thresh).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_true, y_prob)
        except:
            auc_pr = y_true.mean()
        
        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Specificity and NPV
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return ClassificationMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            mcc=mcc,
            kappa=kappa,
            specificity=specificity,
            npv=npv,
            confusion_matrix=cm
        )
    
    def compute_calibration_metrics(self,
                                     y_true: np.ndarray,
                                     y_prob: np.ndarray
                                     ) -> CalibrationMetrics:
        """
        Compute calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Calibration metrics
        """
        # Brier score
        brier = np.mean((y_prob - y_true) ** 2)
        
        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=self.n_calibration_bins
            )
        except:
            prob_true = np.array([0.5])
            prob_pred = np.array([0.5])
        
        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, self.n_calibration_bins + 1)
        bin_counts = np.zeros(self.n_calibration_bins)
        bin_accuracy = np.zeros(self.n_calibration_bins)
        bin_confidence = np.zeros(self.n_calibration_bins)
        
        for i in range(self.n_calibration_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if i == self.n_calibration_bins - 1:
                in_bin = in_bin | (y_prob == 1)
            
            bin_counts[i] = in_bin.sum()
            if bin_counts[i] > 0:
                bin_accuracy[i] = y_true[in_bin].mean()
                bin_confidence[i] = y_prob[in_bin].mean()
        
        # ECE: weighted average of |accuracy - confidence|
        weights = bin_counts / bin_counts.sum()
        ece = np.sum(weights * np.abs(bin_accuracy - bin_confidence))
        
        # MCE: maximum |accuracy - confidence|
        mce = np.max(np.abs(bin_accuracy - bin_confidence))
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            reliability_diagram={
                'prob_true': prob_true,
                'prob_pred': prob_pred,
                'bin_counts': bin_counts
            }
        )
    
    def compute_fairness_metrics(self,
                                  y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  sensitive_attribute: np.ndarray,
                                  threshold: Optional[float] = None
                                  ) -> FairnessMetrics:
        """
        Compute fairness metrics across groups.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            sensitive_attribute: Group membership
            threshold: Classification threshold
            
        Returns:
            Fairness metrics
        """
        thresh = threshold or self.threshold
        y_pred = (y_prob >= thresh).astype(int)
        
        groups = np.unique(sensitive_attribute)
        group_metrics = {}
        
        # Compute per-group metrics
        for group in groups:
            mask = sensitive_attribute == group
            if mask.sum() == 0:
                continue
            
            group_metrics[str(group)] = {
                'size': int(mask.sum()),
                'positive_rate': float(y_pred[mask].mean()),
                'true_positive_rate': float(
                    recall_score(y_true[mask], y_pred[mask], zero_division=0)
                ),
                'false_positive_rate': float(
                    1 - y_pred[mask & (y_true == 0)].mean() if (mask & (y_true == 0)).sum() > 0 else 0
                ),
                'precision': float(
                    precision_score(y_true[mask], y_pred[mask], zero_division=0)
                )
            }
        
        # Demographic parity
        # Difference in positive prediction rates
        pos_rates = [m['positive_rate'] for m in group_metrics.values()]
        demographic_parity = max(pos_rates) - min(pos_rates) if pos_rates else 0
        
        # Equalized odds
        # Difference in TPR and FPR across groups
        tprs = [m['true_positive_rate'] for m in group_metrics.values()]
        fprs = [m['false_positive_rate'] for m in group_metrics.values()]
        equalized_odds = max(
            max(tprs) - min(tprs) if tprs else 0,
            max(fprs) - min(fprs) if fprs else 0
        )
        
        # Predictive parity
        # Difference in precision across groups
        precs = [m['precision'] for m in group_metrics.values()]
        predictive_parity = max(precs) - min(precs) if precs else 0
        
        # Individual fairness (simplified)
        # Variance in predictions for similar individuals
        individual_fairness = float(np.std(y_prob))
        
        return FairnessMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            predictive_parity=predictive_parity,
            individual_fairness=individual_fairness,
            group_metrics=group_metrics
        )
    
    def find_optimal_threshold(self,
                                y_true: np.ndarray,
                                y_prob: np.ndarray,
                                metric: str = 'f1'
                                ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'youden', 'precision', 'recall')
            
        Returns:
            (optimal_threshold, metric_value)
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_value = 0.0
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            if metric == 'f1':
                value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                value = sensitivity + specificity - 1
            elif metric == 'precision':
                value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                value = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if value > best_value:
                best_value = value
                best_threshold = thresh
        
        return best_threshold, best_value
    
    def compute_all_metrics(self,
                            y_true: np.ndarray,
                            y_prob: np.ndarray,
                            sensitive_attribute: Optional[np.ndarray] = None
                            ) -> Dict[str, Any]:
        """
        Compute all available metrics.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            sensitive_attribute: Optional group membership
            
        Returns:
            Dictionary of all metrics
        """
        results = {}
        
        # Classification metrics
        results['classification'] = self.compute_classification_metrics(
            y_true, y_prob
        ).to_dict()
        
        # Calibration metrics
        results['calibration'] = self.compute_calibration_metrics(
            y_true, y_prob
        ).to_dict()
        
        # Fairness metrics
        if sensitive_attribute is not None:
            results['fairness'] = self.compute_fairness_metrics(
                y_true, y_prob, sensitive_attribute
            ).to_dict()
        
        # Optimal thresholds
        for metric in ['f1', 'youden']:
            thresh, value = self.find_optimal_threshold(y_true, y_prob, metric)
            results[f'optimal_threshold_{metric}'] = {
                'threshold': thresh,
                'value': value
            }
        
        return results


class ClinicalMetrics:
    """
    Clinical relevance metrics for mental health prediction.
    """
    
    @staticmethod
    def compute_nnt(y_true: np.ndarray, 
                    y_prob: np.ndarray,
                    threshold: float = 0.5) -> float:
        """
        Compute Number Needed to Treat (NNT).
        
        How many patients need to be screened to identify one true case.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            NNT value
        """
        y_pred = (y_prob >= threshold).astype(int)
        
        # Positive predictive value
        tp = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positive = np.sum(y_pred == 1)
        
        if predicted_positive == 0:
            return float('inf')
        
        ppv = tp / predicted_positive
        
        if ppv == 0:
            return float('inf')
        
        return 1 / ppv
    
    @staticmethod
    def compute_clinical_utility_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_thresholds: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute clinical utility curve.
        
        Net benefit at different risk thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_thresholds: Number of thresholds
            
        Returns:
            Clinical utility data
        """
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        net_benefits = []
        treat_all_benefits = []
        
        prevalence = y_true.mean()
        n = len(y_true)
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            
            # Net benefit = (TP/n) - (FP/n) * (threshold / (1-threshold))
            if thresh < 1:
                net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            else:
                net_benefit = 0
            
            net_benefits.append(net_benefit)
            
            # Treat all strategy
            treat_all = prevalence - (1 - prevalence) * (thresh / (1 - thresh)) if thresh < 1 else 0
            treat_all_benefits.append(treat_all)
        
        return {
            'thresholds': thresholds,
            'net_benefit': np.array(net_benefits),
            'treat_all': np.array(treat_all_benefits),
            'treat_none': np.zeros(n_thresholds)
        }
    
    @staticmethod
    def compute_sensitivity_at_specificity(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_specificity: float = 0.90
    ) -> Tuple[float, float]:
        """
        Compute sensitivity at target specificity.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target_specificity: Target specificity
            
        Returns:
            (sensitivity, threshold)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        specificity = 1 - fpr
        
        # Find threshold closest to target specificity
        idx = np.argmin(np.abs(specificity - target_specificity))
        
        return tpr[idx], thresholds[idx]
    
    @staticmethod
    def compute_specificity_at_sensitivity(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_sensitivity: float = 0.90
    ) -> Tuple[float, float]:
        """
        Compute specificity at target sensitivity.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target_sensitivity: Target sensitivity
            
        Returns:
            (specificity, threshold)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find threshold closest to target sensitivity
        idx = np.argmin(np.abs(tpr - target_sensitivity))
        specificity = 1 - fpr[idx]
        
        return specificity, thresholds[idx]


class FederatedMetrics:
    """
    Metrics specific to federated learning evaluation.
    """
    
    @staticmethod
    def compute_client_metrics(
        client_predictions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Compute per-client and aggregate metrics.
        
        Args:
            client_predictions: Dict of client_id -> (y_true, y_prob)
            
        Returns:
            Client-level and aggregate metrics
        """
        calculator = MetricsCalculator()
        
        client_metrics = {}
        all_y_true = []
        all_y_prob = []
        
        for client_id, (y_true, y_prob) in client_predictions.items():
            metrics = calculator.compute_classification_metrics(y_true, y_prob)
            client_metrics[client_id] = metrics.to_dict()
            
            all_y_true.append(y_true)
            all_y_prob.append(y_prob)
        
        # Aggregate metrics
        combined_y_true = np.concatenate(all_y_true)
        combined_y_prob = np.concatenate(all_y_prob)
        
        aggregate = calculator.compute_classification_metrics(
            combined_y_true, combined_y_prob
        )
        
        # Compute variance across clients
        f1_scores = [m['f1'] for m in client_metrics.values()]
        auc_scores = [m['auc_roc'] for m in client_metrics.values()]
        
        return {
            'clients': client_metrics,
            'aggregate': aggregate.to_dict(),
            'variance': {
                'f1_std': float(np.std(f1_scores)),
                'auc_std': float(np.std(auc_scores)),
                'f1_range': float(max(f1_scores) - min(f1_scores)),
                'auc_range': float(max(auc_scores) - min(auc_scores))
            }
        }
    
    @staticmethod
    def compute_personalization_benefit(
        global_predictions: Tuple[np.ndarray, np.ndarray],
        personalized_predictions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Compute benefit of personalization over global model.
        
        Args:
            global_predictions: (y_true, y_prob) from global model
            personalized_predictions: Per-client personalized predictions
            
        Returns:
            Personalization benefit metrics
        """
        calculator = MetricsCalculator()
        
        global_metrics = calculator.compute_classification_metrics(
            global_predictions[0], global_predictions[1]
        )
        
        personalization_benefits = {}
        
        for client_id, (y_true, y_prob) in personalized_predictions.items():
            personalized = calculator.compute_classification_metrics(y_true, y_prob)
            
            personalization_benefits[client_id] = {
                'f1_improvement': personalized.f1 - global_metrics.f1,
                'auc_improvement': personalized.auc_roc - global_metrics.auc_roc,
                'accuracy_improvement': personalized.accuracy - global_metrics.accuracy
            }
        
        # Average improvement
        avg_f1_imp = np.mean([b['f1_improvement'] for b in personalization_benefits.values()])
        avg_auc_imp = np.mean([b['auc_improvement'] for b in personalization_benefits.values()])
        
        return {
            'global_metrics': global_metrics.to_dict(),
            'per_client': personalization_benefits,
            'average_improvement': {
                'f1': avg_f1_imp,
                'auc': avg_auc_imp
            }
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic predictions
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive
    y_prob = np.clip(y_true + np.random.randn(n_samples) * 0.3, 0, 1)
    sensitive = np.random.choice(['A', 'B', 'C'], n_samples)
    
    print("=" * 60)
    print("Comprehensive Metrics Demonstration")
    print("=" * 60)
    
    calculator = MetricsCalculator()
    
    # Classification metrics
    print("\nClassification Metrics:")
    class_metrics = calculator.compute_classification_metrics(y_true, y_prob)
    for key, value in class_metrics.to_dict().items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value:.4f}")
    
    # Calibration metrics
    print("\nCalibration Metrics:")
    calib_metrics = calculator.compute_calibration_metrics(y_true, y_prob)
    print(f"  ECE: {calib_metrics.ece:.4f}")
    print(f"  MCE: {calib_metrics.mce:.4f}")
    print(f"  Brier Score: {calib_metrics.brier_score:.4f}")
    
    # Fairness metrics
    print("\nFairness Metrics:")
    fair_metrics = calculator.compute_fairness_metrics(y_true, y_prob, sensitive)
    print(f"  Demographic Parity: {fair_metrics.demographic_parity:.4f}")
    print(f"  Equalized Odds: {fair_metrics.equalized_odds:.4f}")
    print(f"  Predictive Parity: {fair_metrics.predictive_parity:.4f}")
    
    # Clinical metrics
    print("\nClinical Metrics:")
    clinical = ClinicalMetrics()
    nnt = clinical.compute_nnt(y_true, y_prob)
    sens_at_spec, thresh = clinical.compute_sensitivity_at_specificity(y_true, y_prob, 0.90)
    print(f"  NNT: {nnt:.2f}")
    print(f"  Sensitivity @ 90% Specificity: {sens_at_spec:.4f}")
    
    # Optimal thresholds
    print("\nOptimal Thresholds:")
    for metric in ['f1', 'youden']:
        thresh, value = calculator.find_optimal_threshold(y_true, y_prob, metric)
        print(f"  {metric}: threshold={thresh:.3f}, value={value:.4f}")
