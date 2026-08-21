"""
Privacy package for federated learning.
"""

from .dp_accounting import (
    RDPAccountant,
    CompositionAccountant,
    FederatedPrivacyAccountant,
    PrivacyBudget,
    MechanismParams,
    AccountingMethod
)

from .secure_aggregation import (
    SecureAggregator,
    ThresholdSecureAggregator,
    SecureAggConfig,
    SecureAggregationProtocol,
    SecretSharing,
    verify_secure_aggregation
)

from .attacks import (
    MembershipInferenceAttack,
    GradientLeakageAttack,
    AttributeInferenceAttack,
    DPDefenseEvaluator,
    AttackType,
    AttackResult,
    run_privacy_audit
)

__all__ = [
    # DP Accounting
    'RDPAccountant',
    'CompositionAccountant',
    'FederatedPrivacyAccountant',
    'PrivacyBudget',
    'MechanismParams',
    'AccountingMethod',
    
    # Secure Aggregation
    'SecureAggregator',
    'ThresholdSecureAggregator',
    'SecureAggConfig',
    'SecureAggregationProtocol',
    'SecretSharing',
    'verify_secure_aggregation',
    
    # Attacks
    'MembershipInferenceAttack',
    'GradientLeakageAttack',
    'AttributeInferenceAttack',
    'DPDefenseEvaluator',
    'AttackType',
    'AttackResult',
    'run_privacy_audit'
]
