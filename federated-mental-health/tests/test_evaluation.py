"""
Tests for evaluation components.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    MetricsCalculator,
    ClassificationMetrics,
    CalibrationMetrics,
    FairnessMetrics,
    ClinicalMetrics
)
from evaluation.privacy_eval import (
    PrivacyEvaluator,
    PrivacyUtilityTradeoff,
    FederatedPrivacyEvaluator
)
from evaluation.comparative_analysis import (
    ComparativeAnalyzer,
    ExperimentResult,
    AblationStudy
)


class TestClassificationMetrics:
    """Tests for classification metrics."""
    
    @pytest.fixture
    def predictions(self):
        """Generate sample predictions."""
        np.random.seed(42)
        n_samples = 1000
        
        # True labels
        y_true = np.random.randint(0, 2, n_samples)
        
        # Predicted probabilities (correlated with true)
        y_pred = y_true * 0.7 + np.random.uniform(0, 0.3, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        
        return y_true, y_pred
    
    def test_accuracy(self, predictions):
        """Test accuracy calculation."""
        y_true, y_pred = predictions
        
        metrics = ClassificationMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
    
    def test_auc_roc(self, predictions):
        """Test AUC-ROC calculation."""
        y_true, y_pred = predictions
        
        metrics = ClassificationMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'auc_roc' in result
        assert 0 <= result['auc_roc'] <= 1
        assert result['auc_roc'] > 0.5  # Better than random
    
    def test_precision_recall_f1(self, predictions):
        """Test precision, recall, and F1."""
        y_true, y_pred = predictions
        
        metrics = ClassificationMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        
        # All should be in [0, 1]
        assert all(0 <= result[m] <= 1 for m in ['precision', 'recall', 'f1_score'])
    
    def test_sensitivity_specificity(self, predictions):
        """Test sensitivity and specificity."""
        y_true, y_pred = predictions
        
        metrics = ClassificationMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'sensitivity' in result
        assert 'specificity' in result
    
    def test_confusion_matrix(self, predictions):
        """Test confusion matrix."""
        y_true, y_pred = predictions
        
        metrics = ClassificationMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'confusion_matrix' in result
        cm = result['confusion_matrix']
        
        # Should be 2x2
        assert cm.shape == (2, 2)
        
        # Sum should equal total samples
        assert cm.sum() == len(y_true)


class TestCalibrationMetrics:
    """Tests for calibration metrics."""
    
    @pytest.fixture
    def calibrated_predictions(self):
        """Generate well-calibrated predictions."""
        np.random.seed(42)
        n_samples = 1000
        
        # True probabilities
        true_probs = np.random.uniform(0, 1, n_samples)
        y_true = (np.random.uniform(0, 1, n_samples) < true_probs).astype(int)
        
        # Predicted probabilities (close to true)
        y_pred = true_probs + np.random.normal(0, 0.1, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        
        return y_true, y_pred
    
    @pytest.fixture
    def miscalibrated_predictions(self):
        """Generate miscalibrated predictions."""
        np.random.seed(42)
        n_samples = 1000
        
        y_true = np.random.randint(0, 2, n_samples)
        
        # Overconfident predictions
        y_pred = np.where(y_true == 1, 0.95, 0.05)
        
        return y_true, y_pred
    
    def test_ece_calculation(self, calibrated_predictions):
        """Test Expected Calibration Error."""
        y_true, y_pred = calibrated_predictions
        
        metrics = CalibrationMetrics(n_bins=10)
        result = metrics.calculate(y_true, y_pred)
        
        assert 'ece' in result
        assert result['ece'] >= 0
    
    def test_mce_calculation(self, calibrated_predictions):
        """Test Maximum Calibration Error."""
        y_true, y_pred = calibrated_predictions
        
        metrics = CalibrationMetrics(n_bins=10)
        result = metrics.calculate(y_true, y_pred)
        
        assert 'mce' in result
        assert result['mce'] >= result['ece']  # MCE >= ECE
    
    def test_calibration_curve(self, calibrated_predictions):
        """Test calibration curve generation."""
        y_true, y_pred = calibrated_predictions
        
        metrics = CalibrationMetrics(n_bins=10)
        result = metrics.calculate(y_true, y_pred)
        
        assert 'calibration_curve' in result


class TestFairnessMetrics:
    """Tests for fairness metrics."""
    
    @pytest.fixture
    def biased_data(self):
        """Generate data with group disparities."""
        np.random.seed(42)
        n_samples = 1000
        
        # Protected attribute (e.g., group membership)
        protected = np.random.randint(0, 2, n_samples)
        
        # True labels
        y_true = np.random.randint(0, 2, n_samples)
        
        # Biased predictions (favor group 0)
        y_pred = np.where(protected == 0, 0.7, 0.3)
        y_pred += np.random.normal(0, 0.1, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        
        return y_true, y_pred, protected
    
    def test_demographic_parity(self, biased_data):
        """Test demographic parity calculation."""
        y_true, y_pred, protected = biased_data
        
        metrics = FairnessMetrics()
        result = metrics.calculate(y_true, y_pred, protected)
        
        assert 'demographic_parity_diff' in result
    
    def test_equalized_odds(self, biased_data):
        """Test equalized odds calculation."""
        y_true, y_pred, protected = biased_data
        
        metrics = FairnessMetrics()
        result = metrics.calculate(y_true, y_pred, protected)
        
        assert 'equalized_odds_diff' in result
    
    def test_fairness_ratio(self, biased_data):
        """Test disparate impact ratio."""
        y_true, y_pred, protected = biased_data
        
        metrics = FairnessMetrics()
        result = metrics.calculate(y_true, y_pred, protected)
        
        assert 'disparate_impact_ratio' in result
        
        # Ratio of 1.0 means perfect fairness
        assert 0 <= result['disparate_impact_ratio'] <= 2


class TestClinicalMetrics:
    """Tests for clinical metrics."""
    
    @pytest.fixture
    def clinical_predictions(self):
        """Generate clinical prediction data."""
        np.random.seed(42)
        n_samples = 500
        
        # Prevalence of ~10%
        y_true = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        # Predictions correlated with true labels
        y_pred = y_true * 0.5 + np.random.uniform(0, 0.5, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        
        return y_true, y_pred
    
    def test_nnt_calculation(self, clinical_predictions):
        """Test Number Needed to Treat."""
        y_true, y_pred = clinical_predictions
        
        metrics = ClinicalMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'nnt' in result
        assert result['nnt'] >= 1  # NNT is always >= 1
    
    def test_sensitivity_at_specificity(self, clinical_predictions):
        """Test sensitivity at fixed specificity."""
        y_true, y_pred = clinical_predictions
        
        metrics = ClinicalMetrics(target_specificity=0.9)
        result = metrics.calculate(y_true, y_pred)
        
        assert 'sensitivity_at_target_specificity' in result
    
    def test_ppv_npv(self, clinical_predictions):
        """Test positive and negative predictive values."""
        y_true, y_pred = clinical_predictions
        
        metrics = ClinicalMetrics()
        result = metrics.calculate(y_true, y_pred)
        
        assert 'ppv' in result  # Positive Predictive Value
        assert 'npv' in result  # Negative Predictive Value


class TestMetricsCalculator:
    """Tests for comprehensive metrics calculator."""
    
    @pytest.fixture
    def full_data(self):
        """Generate comprehensive data."""
        np.random.seed(42)
        n_samples = 500
        
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true * 0.6 + np.random.uniform(0, 0.4, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        protected = np.random.randint(0, 2, n_samples)
        
        return y_true, y_pred, protected
    
    def test_comprehensive_metrics(self, full_data):
        """Test all metrics together."""
        y_true, y_pred, protected = full_data
        
        calculator = MetricsCalculator()
        result = calculator.calculate_all(
            y_true=y_true,
            y_pred=y_pred,
            protected_attribute=protected
        )
        
        # Check all metric types present
        assert 'classification' in result
        assert 'calibration' in result
        assert 'fairness' in result
        assert 'clinical' in result
    
    def test_metrics_summary(self, full_data):
        """Test metrics summary generation."""
        y_true, y_pred, protected = full_data
        
        calculator = MetricsCalculator()
        result = calculator.calculate_all(y_true, y_pred, protected)
        
        summary = calculator.get_summary(result)
        
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestPrivacyEvaluator:
    """Tests for privacy evaluation."""
    
    def test_privacy_evaluation(self):
        """Test privacy evaluation."""
        evaluator = PrivacyEvaluator()
        
        # Mock privacy report
        privacy_report = {
            'epsilon': 1.0,
            'delta': 1e-5,
            'noise_multiplier': 1.1,
            'n_rounds': 100
        }
        
        result = evaluator.evaluate(privacy_report)
        
        assert 'privacy_level' in result
        assert 'risk_assessment' in result
    
    def test_privacy_levels(self):
        """Test privacy level classification."""
        evaluator = PrivacyEvaluator()
        
        # Strong privacy
        strong = evaluator.evaluate({'epsilon': 0.5, 'delta': 1e-6})
        
        # Moderate privacy
        moderate = evaluator.evaluate({'epsilon': 2.0, 'delta': 1e-5})
        
        # Weak privacy
        weak = evaluator.evaluate({'epsilon': 10.0, 'delta': 1e-3})
        
        # Privacy levels should differ
        assert strong['privacy_level'] != weak['privacy_level']


class TestPrivacyUtilityTradeoff:
    """Tests for privacy-utility tradeoff analysis."""
    
    def test_tradeoff_analysis(self):
        """Test privacy-utility tradeoff."""
        analyzer = PrivacyUtilityTradeoff()
        
        # Results at different epsilon values
        results = [
            {'epsilon': 0.1, 'accuracy': 0.65, 'auc': 0.70},
            {'epsilon': 0.5, 'accuracy': 0.72, 'auc': 0.78},
            {'epsilon': 1.0, 'accuracy': 0.78, 'auc': 0.84},
            {'epsilon': 5.0, 'accuracy': 0.82, 'auc': 0.88},
        ]
        
        tradeoff = analyzer.analyze(results)
        
        assert 'optimal_epsilon' in tradeoff
        assert 'pareto_frontier' in tradeoff
    
    def test_utility_at_epsilon(self):
        """Test utility at specific epsilon."""
        analyzer = PrivacyUtilityTradeoff()
        
        results = [
            {'epsilon': 0.5, 'accuracy': 0.70},
            {'epsilon': 1.0, 'accuracy': 0.75},
            {'epsilon': 2.0, 'accuracy': 0.80},
        ]
        
        analyzer.fit(results)
        
        # Interpolate utility at epsilon=1.5
        utility = analyzer.predict_utility(epsilon=1.5)
        
        assert 0.75 <= utility <= 0.80


class TestComparativeAnalyzer:
    """Tests for experiment comparison."""
    
    @pytest.fixture
    def experiments(self):
        """Create sample experiment results."""
        return [
            ExperimentResult(
                name='experiment_1',
                epsilon=1.0,
                accuracy=0.78,
                auc_roc=0.85,
                f1=0.76,
                training_time=1000
            ),
            ExperimentResult(
                name='experiment_2',
                epsilon=0.5,
                accuracy=0.72,
                auc_roc=0.80,
                f1=0.70,
                training_time=1200
            ),
            ExperimentResult(
                name='experiment_3',
                epsilon=2.0,
                accuracy=0.82,
                auc_roc=0.88,
                f1=0.80,
                training_time=900
            )
        ]
    
    def test_comparison_table(self, experiments):
        """Test comparison table generation."""
        analyzer = ComparativeAnalyzer()
        
        table = analyzer.create_comparison_table(experiments)
        
        assert len(table) == 3
    
    def test_best_experiment(self, experiments):
        """Test finding best experiment."""
        analyzer = ComparativeAnalyzer()
        
        best = analyzer.find_best(experiments, metric='accuracy')
        
        assert best.name == 'experiment_3'  # Highest accuracy
    
    def test_pareto_optimal(self, experiments):
        """Test Pareto frontier computation."""
        analyzer = ComparativeAnalyzer()
        
        pareto = analyzer.get_pareto_frontier(
            experiments,
            objectives=['accuracy', 'epsilon'],
            minimize=['epsilon']  # Lower epsilon is better
        )
        
        assert len(pareto) > 0


class TestAblationStudy:
    """Tests for ablation study analysis."""
    
    def test_ablation_analysis(self):
        """Test ablation study analysis."""
        study = AblationStudy()
        
        # Results with and without components
        results = {
            'full_model': {'accuracy': 0.80, 'auc': 0.86},
            'no_attention': {'accuracy': 0.75, 'auc': 0.82},
            'no_dp': {'accuracy': 0.85, 'auc': 0.90},
            'no_federated': {'accuracy': 0.82, 'auc': 0.87},
        }
        
        analysis = study.analyze(results, baseline='full_model')
        
        assert 'component_contributions' in analysis
        assert 'attention' in str(analysis)
    
    def test_component_importance(self):
        """Test component importance ranking."""
        study = AblationStudy()
        
        results = {
            'full': {'accuracy': 0.80},
            'without_A': {'accuracy': 0.70},  # A contributes 10%
            'without_B': {'accuracy': 0.75},  # B contributes 5%
            'without_C': {'accuracy': 0.78},  # C contributes 2%
        }
        
        analysis = study.analyze(results, baseline='full')
        
        # Component A should be ranked most important
        importance = analysis['component_contributions']
        assert importance['A'] > importance['B'] > importance['C']


class TestMetricsIntegration:
    """Integration tests for metrics."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate data
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = y_true * 0.6 + np.random.uniform(0, 0.4, n_samples)
        y_pred = np.clip(y_pred, 0, 1)
        protected = np.random.randint(0, 2, n_samples)
        
        # Calculate all metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all(y_true, y_pred, protected)
        
        # Privacy evaluation
        privacy_evaluator = PrivacyEvaluator()
        privacy = privacy_evaluator.evaluate({
            'epsilon': 1.0,
            'delta': 1e-5
        })
        
        # Combine results
        full_report = {
            'metrics': metrics,
            'privacy': privacy
        }
        
        assert 'classification' in full_report['metrics']
        assert 'privacy_level' in full_report['privacy']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
