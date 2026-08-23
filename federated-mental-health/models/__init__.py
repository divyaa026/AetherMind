"""
Models package for Federated Mental Health Learning
"""

from .architecture import (
    MentalHealthPredictor,
    SelfAttention,
    FocalLoss,
    WeightedBCELoss,
    create_model,
    count_parameters,
    model_summary
)

from .dp_optimizer import (
    DPConfig,
    DPOptimizer,
    LocalDPTrainer,
    calibrate_noise_multiplier
)

from .train_local import (
    TrainingConfig,
    LocalTrainer,
    train_local_model
)

__all__ = [
    # Architecture
    'MentalHealthPredictor',
    'SelfAttention',
    'FocalLoss',
    'WeightedBCELoss',
    'create_model',
    'count_parameters',
    'model_summary',
    # DP
    'DPConfig',
    'DPOptimizer',
    'LocalDPTrainer',
    'calibrate_noise_multiplier',
    # Training
    'TrainingConfig',
    'LocalTrainer',
    'train_local_model'
]
