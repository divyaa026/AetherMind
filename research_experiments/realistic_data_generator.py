"""
Realistic Synthetic Data Generator for Mental Health Crisis Prediction

This module generates synthetic data with realistic characteristics:
- Low positive class ratio (2-5% crisis days)
- Probabilistic relationships with label noise
- User-level temporal correlations
- Validated with logistic regression baseline
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger('ResearchExperiments.RealisticDataGen')


class RealisticDataGenerator:
    """Generate realistic mental health data with proper characteristics."""
    
    def __init__(self, seed: int = 42, positive_ratio: float = 0.03):
        """
        Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
            positive_ratio: Target ratio of positive (crisis) days (default 3%)
        """
        self.seed = seed
        self.positive_ratio = positive_ratio
        np.random.seed(seed)
        
    def generate_data(self, n_samples: int = 5000, n_users: int = 100) -> pd.DataFrame:
        """
        Generate realistic mental health data.
        
        Key features:
        - 2-5% positive class ratio (mental health crises are rare)
        - Probabilistic relationships (not deterministic)
        - Label noise added to prevent perfect predictions
        - User-level heterogeneity and temporal correlations
        
        Args:
            n_samples: Total number of samples
            n_users: Number of users
            
        Returns:
            DataFrame with realistic synthetic data
        """
        logger.info(f"Generating realistic data: {n_samples} samples, {n_users} users")
        logger.info(f"Target positive ratio: {self.positive_ratio:.1%}")
        
        days_per_user = n_samples // n_users
        data = []
        
        for user_id in range(n_users):
            # Each user has different baseline characteristics
            user_baseline = {
                'stress': np.random.uniform(0.25, 0.65),
                'sleep': np.random.uniform(5.5, 7.5),
                'risk_threshold': np.random.uniform(0.65, 0.85)  # Personal crisis threshold
            }
            
            for day in range(days_per_user):
                # Temporal correlation: today influenced by yesterday
                if day == 0:
                    stress = user_baseline['stress'] + np.random.normal(0, 0.15)
                    sleep = user_baseline['sleep'] + np.random.normal(0, 0.8)
                else:
                    # AR(1) process with user baseline
                    stress = (0.6 * data[-1]['stress_level'] + 
                             0.4 * user_baseline['stress'] + 
                             np.random.normal(0, 0.12))
                    sleep = (0.5 * data[-1]['sleep_hours'] + 
                            0.5 * user_baseline['sleep'] + 
                            np.random.normal(0, 0.7))
                
                # Clip to valid ranges
                stress = np.clip(stress, 0, 1)
                sleep = np.clip(sleep, 3, 11)
                
                # Other features moderately correlated with stress
                activity = np.clip(0.75 - 0.5*stress + np.random.normal(0, 0.25), 0, 1)
                social = np.clip(0.65 - 0.4*stress + np.random.normal(0, 0.2), 0, 1)
                mood = np.clip(0.7 - 0.45*stress + np.random.normal(0, 0.18), 0, 1)
                
                # Calculate probabilistic risk score (not deterministic!)
                # Create clear but noisy signal for learning
                sleep_deprivation = max(0, (7 - sleep) / 4)  # 0-1 scale, worse for <7hr
                
                # Combine stress and sleep as main risk factors (stronger weights)
                risk_score = (
                    0.60 * stress +  # Stress is primary indicator
                    0.25 * sleep_deprivation +  # Sleep matters significantly  
                    0.08 * (1 - activity) +
                    0.04 * (1 - social) +
                    0.03 * (1 - mood)
                )
                
                # Reduce noise for better learnability (but still probabilistic)
                risk_score = np.clip(risk_score + np.random.normal(0, 0.08), 0, 1)
                
                # Generate label based on probabilistic threshold
                # Not all high stress leads to crisis! And some low stress can.
                crisis_probability = self._calculate_crisis_probability(
                    risk_score, 
                    user_baseline['risk_threshold']
                )
                
                # Sample from probability to get binary label
                high_risk = int(np.random.random() < crisis_probability)
                
                # Add minimal label noise: 2% chance of flipping the label
                if np.random.random() < 0.02:
                    high_risk = 1 - high_risk
                
                data.append({
                    'user_id': user_id,
                    'day': day,
                    'sleep_hours': sleep,
                    'stress_level': stress,
                    'physical_activity': activity,
                    'social_interaction': social,
                    'mood_score': mood,
                    'risk_score': risk_score,  # Hidden variable for debugging
                    'high_risk_day': high_risk
                })
        
        df = pd.DataFrame(data)
        
        # Adjust to target positive ratio by sampling
        df = self._balance_to_target_ratio(df)
        
        actual_ratio = df['high_risk_day'].mean()
        logger.info(f"Generated data: {len(df)} samples")
        logger.info(f"Actual positive ratio: {actual_ratio:.2%}")
        logger.info(f"Number of crisis days: {df['high_risk_day'].sum()}")
        
        return df
    
    def _calculate_crisis_probability(self, risk_score: float, threshold: float) -> float:
        """
        Calculate probability of crisis given risk score and personal threshold.
        
        Creates learnable but probabilistic relationship.
        """
        # Create clear predictive signal while maintaining event rarity
        if risk_score > threshold:
            # Above personal threshold: significantly higher risk
            excess = (risk_score - threshold) / (1 - threshold)
            base_prob = 0.20 + 0.55 * (excess ** 1.2)  # 20-75% range for high risk
        else:
            # Below threshold: low but increasing risk
            ratio = risk_score / threshold  
            base_prob = 0.015 + 0.08 * (ratio ** 2)  # 1.5-9.5% range for moderate risk
        
        # Minimal noise to keep predictability
        noise = np.random.normal(0, 0.02)
        final_prob = np.clip(base_prob + noise, 0.01, 0.80)
        
        return final_prob
    
    def _balance_to_target_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust dataset to match target positive ratio by resampling.
        
        This is more realistic than synthetic generation that hits exact ratios.
        """
        current_ratio = df['high_risk_day'].mean()
        
        if abs(current_ratio - self.positive_ratio) < 0.01:
            return df  # Already close enough
        
        # If too many positives, randomly downsample some positives
        # If too few, this indicates the generation logic needs adjustment
        if current_ratio > self.positive_ratio * 1.5:
            positive_df = df[df['high_risk_day'] == 1]
            negative_df = df[df['high_risk_day'] == 0]
            
            target_positives = int(len(df) * self.positive_ratio)
            positive_df = positive_df.sample(n=target_positives, random_state=self.seed)
            
            df = pd.concat([positive_df, negative_df], ignore_index=True)
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df
    
    def validate_with_logistic_regression(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality using logistic regression baseline.
        
        A good synthetic dataset should have:
        - AUC-PR: 0.70-0.80 (challenging but learnable)
        - AUC-ROC: 0.75-0.85 (better than random, not perfect)
        
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating data with logistic regression...")
        
        # Prepare features and labels
        feature_cols = ['sleep_hours', 'stress_level', 'physical_activity', 
                       'social_interaction', 'mood_score']
        X = df[feature_cols].values
        y = df['high_risk_day'].values
        
        # Train-test split (80-20)
        n_train = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train logistic regression with polynomial features for better learning
        from sklearn.preprocessing import PolynomialFeatures
        
        # Create interaction terms to help logistic regression
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        lr = LogisticRegression(
            random_state=self.seed,
            max_iter=2000,
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs',
            C=0.1  # Add regularization
        )
        lr.fit(X_train_poly, y_train)
        
        # Predict probabilities
        y_pred_proba = lr.predict_proba(X_test_poly)[:, 1]
        
        # Calculate metrics
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.5
        
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auc_pr = auc(recall, precision)
        except:
            auc_pr = y_test.mean()  # Random baseline for imbalanced data
        
        # Check feature importance (from original features only)
        # Get coefficients for main effects (first 5 features after constant if bias=False)
        main_feature_importance = dict(zip(feature_cols, np.abs(lr.coef_[0][:len(feature_cols)])))
        top_feature = max(main_feature_importance, key=main_feature_importance.get)
        
        results = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'n_train': n_train,
            'n_test': len(test_idx),
            'positive_ratio_train': y_train.mean(),
            'positive_ratio_test': y_test.mean(),
            'feature_importance': main_feature_importance,
            'top_feature': top_feature,
            'random_baseline_auc_pr': y_test.mean()
        }
        
        # Calculate improvement over random baseline
        random_baseline = y_test.mean()
        improvement_factor = auc_pr / random_baseline if random_baseline > 0 else 0
        
        # Validation checks
        logger.info(f"Logistic Regression Validation Results:")
        logger.info(f"  AUC-ROC: {auc_roc:.4f}")
        logger.info(f"  AUC-PR: {auc_pr:.4f}")
        logger.info(f"  Random Baseline AUC-PR: {random_baseline:.4f}")
        logger.info(f"  Improvement Factor: {improvement_factor:.2f}x over random")
        logger.info(f"  Positive ratio (test): {y_test.mean():.2%}")
        logger.info(f"  Top predictive feature: {top_feature}")
        
        # Realistic expectations for imbalanced data (3-5% positive)
        # Good models should be 4-10x better than random guessing
        if improvement_factor >= 8:
            logger.info("SUCCESS: Strong predictive signal (8x+ improvement over random)")
            results['quality'] = 'excellent'
        elif improvement_factor >= 4:
            logger.info("SUCCESS: Reasonable predictive signal (4-8x improvement over random)")
            results['quality'] = 'good'
        elif improvement_factor >= 2:
            logger.warning("MARGINAL: Weak but learnable signal (2-4x improvement)")
            results['quality'] = 'marginal'
        else:
            logger.warning("WARNING: Signal may be too weak or noisy")
            results['quality'] = 'poor'
        
        results['is_realistic'] = improvement_factor >= 3
        results['improvement_factor'] = improvement_factor
        
        return results


def generate_realistic_sequential_data(
    n_samples: int,
    n_users: int,
    seq_length: int,
    positive_ratio: float = 0.03,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic sequential data for LSTM models.
    
    Returns:
        Tuple of (X, y, user_ids) where:
        - X: shape (n_samples, seq_length, n_features)
        - y: shape (n_samples, 1)
        - user_ids: shape (n_samples,)
    """
    generator = RealisticDataGenerator(seed=seed, positive_ratio=positive_ratio)
    
    # Generate more data to create sequences
    total_needed = n_samples * seq_length
    df = generator.generate_data(n_samples=total_needed, n_users=n_users)
    
    feature_cols = ['sleep_hours', 'stress_level', 'physical_activity', 
                   'social_interaction', 'mood_score']
    
    X_sequences = []
    y_labels = []
    user_ids = []
    
    # Create sequences per user to avoid data leakage
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('day')
        
        if len(user_data) < seq_length:
            continue
        
        features = user_data[feature_cols].values
        labels = user_data['high_risk_day'].values
        
        # Create sliding windows
        for i in range(len(features) - seq_length + 1):
            X_sequences.append(features[i:i+seq_length])
            y_labels.append(labels[i+seq_length-1])  # Predict last day in sequence
            user_ids.append(user_id)
    
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32).reshape(-1, 1)
    user_ids = np.array(user_ids)
    
    logger.info(f"Generated {len(X)} sequences of length {seq_length}")
    logger.info(f"Positive ratio: {y.mean():.2%}")
    
    return X, y, user_ids


if __name__ == "__main__":
    """Test data generation and validation."""
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("REALISTIC DATA GENERATOR TEST")
    print("="*80)
    
    # Generate data
    generator = RealisticDataGenerator(seed=42, positive_ratio=0.03)
    df = generator.generate_data(n_samples=5000, n_users=100)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Positive class ratio: {df['high_risk_day'].mean():.2%}")
    print(f"\nFeature statistics:")
    print(df[['stress_level', 'sleep_hours', 'mood_score']].describe())
    
    # Validate with logistic regression
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION VALIDATION")
    print("="*80)
    results = generator.validate_with_logistic_regression(df)
    
    if results['is_realistic']:
        print("\n[OK] Data generation passed validation!")
    else:
        print("\n[WARNING] Data may need adjustment!")
    
    print("\n" + "="*80)
