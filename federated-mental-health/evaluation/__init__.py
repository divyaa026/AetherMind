"""
Evaluation package for federated mental health prediction.
"""

from .metrics import (
    MetricsCalculator,
    ClassificationMetrics,
    CalibrationMetrics,
    FairnessMetrics,
    ClinicalMetrics,
    FederatedMetrics
)

from .privacy_eval import (
    PrivacyEvaluator,
    PrivacyEvaluationResult,
    PrivacyUtilityTradeoff,
    FederatedPrivacyEvaluator,
    run_comprehensive_privacy_evaluation
)

from .comparative_analysis import (
    ComparativeAnalyzer,
    ExperimentResult,
    AblationStudy,
    CrossValidationAnalyzer
)

__all__ = [
    # Metrics
    'MetricsCalculator',
    'ClassificationMetrics',
    'CalibrationMetrics',
    'FairnessMetrics',
    'ClinicalMetrics',
    'FederatedMetrics',
    
    # Privacy Evaluation
    'PrivacyEvaluator',
    'PrivacyEvaluationResult',
    'PrivacyUtilityTradeoff',
    'FederatedPrivacyEvaluator',
    'run_comprehensive_privacy_evaluation',
    
    # Comparative Analysis
    'ComparativeAnalyzer',
    'ExperimentResult',
    'AblationStudy',
    'CrossValidationAnalyzer'
]
