# Critical Fixes Applied to Research Experiment Suite

## Date: January 18, 2026

## Issues Identified and Fixed

### 1. **CRITICAL: Unrealistic Positive Class Ratio (FIXED)**
- **Problem**: Original data had 38% positive class ratio
- **Solution**: Reduced to realistic 3-4% (mental health crises are rare events)
- **Implementation**: New `realistic_data_generator.py` module

### 2. **CRITICAL: Deterministic Relationships (FIXED)**
- **Problem**: `high_risk = int(stress > 0.7 or sleep < 5.5)` created perfect predictions
- **Solution**: Probabilistic risk calculation with label noise (2% flip rate)
- **Key Change**: Uses risk scores + personal thresholds + sampling from probability distribution

### 3. **Data Validation Expectations (ADJUSTED)**
- **Problem**: Expected AUC-PR 0.70-0.80 unrealistic for 3% imbalanced data
- **Solution**: Adjusted to improvement factor over random baseline
  - Random baseline AUC-PR ≈ 0.03 (just guessing positive class)
  - Good model: 4-8x improvement (AUC-PR: 0.12-0.24)
  - Strong model: 8x+ improvement (AUC-PR: 0.24+)

### 4. **Realistic Performance Expectations (DOCUMENTED)**

#### New Expected Results:
- **Centralized Baseline**: AUC-PR ~0.40-0.60 (not 0.99!)
- **Federated (no DP)**: AUC-PR ~0.35-0.55 (5-15% drop from centralized)
- **FL + DP (ε=1.0)**: AUC-PR ~0.30-0.50 (15-25% drop with strong privacy)
- **Extreme Non-IID**: 15-25% performance degradation
- **Membership Inference**: AUC ≈ 0.50-0.55 (near-random = good privacy)

## Files Created

### realistic_data_generator.py (NEW)
**Purpose**: Generate realistic synthetic mental health data

**Key Features**:
- Low positive class ratio (3-4% crisis days)
- Probabilistic relationships (not deterministic)
- User-level heterogeneity and temporal correlations
- Label noise (2% random flips)
- Validation with logistic regression + polynomial features

**Key Functions**:
- `RealisticDataGenerator.generate_data()` - Main data generation
- `generate_realistic_sequential_data()` - For LSTM models
- `validate_with_logistic_regression()` - Quality check

## Next Steps Required

### Step 1: Update All Experiments to Use New Data Generator
Files to modify:
1. `experiment_data_quality.py` - Replace synthetic data generation
2. `experiment_convergence.py` - Use `generate_realistic_sequential_data()`
3. `experiment_privacy_tradeoff.py` - Use `generate_realistic_sequential_data()`
4. `experiment_robustness.py` - Use new generator + fix extreme non-IID

### Step 2: Fix Data Leakage in Train/Test Split
**Current Issue**: Random split across all samples may leak temporal information

**Solution**: Split by user_id BEFORE creating sequences
```python
# Split users into train/test
train_users, test_users = split_users(all_users, train_ratio=0.8)

# THEN create sequences per user set
train_sequences = create_sequences(data[data.user_id.isin(train_users)])
test_sequences = create_sequences(data[data.user_id.isin(test_users)])
```

### Step 3: Fix Extreme Non-IID Generation
**Current Issue**: Just sorts users by stress - not extreme enough

**Solution**: 
- Assign each client a specific stress range (e.g., Client 1: 0-0.3, Client 2: 0.3-0.6, Client 3: 0.6-1.0)
- This creates TRUE distribution shift (some clients never see high-stress users)

### Step 4: Verify Differential Privacy Implementation
**Add Test**:
```python
# Train two models: one with DP, one without
model_no_dp = train_model(epsilon=None)
model_with_dp = train_model(epsilon=1.0)

# Verify outputs differ
assert not np.allclose(model_no_dp.parameters, model_with_dp.parameters)
assert model_with_dp.privacy_accountant.epsilon <= 1.0
```

### Step 5: Delete Redundant/Obsolete Files
Files to remove:
1. Old result files from unrealistic runs
2. `SUCCESS_SUMMARY.md` (will be replaced after fixes)
3. Any cached synthetic data with 38% positive ratio

### Step 6: Update Documentation
Files to update:
1. `README.md` - Update expected performance ranges
2. `config.yaml` - Adjust epsilon values if needed
3. `IMPLEMENTATION_SUMMARY.md` - Document data generation changes

## Validation Checklist

Before re-running experiments, verify:

- [ ] Data generator produces 3-5% positive class ratio
- [ ] Logistic regression gets 4-8x improvement over random (AUC-PR: 0.12-0.30)
- [ ] Train/test split is user-level (no data leakage)
- [ ] Extreme non-IID creates true distribution shift
- [ ] DP implementation verified with test script
- [ ] Expected performance ranges documented

## Expected Timeline

1. Update experiments with new data generator: 30-45 minutes
2. Verify no data leakage: 15 minutes
3. Fix extreme non-IID: 10 minutes
4. Add DP verification test: 10 minutes
5. Clean up old files: 5 minutes
6. Re-run quick test: 10-15 minutes
7. Re-run full experiments: 60-90 minutes
8. Review and validate results: 15-20 minutes

**Total: ~2.5-3 hours for complete fix and re-run**

## Critical Success Criteria

Results should show:
1. ✅ Positive class ratio: 3-5%
2. ✅ Centralized AUC-PR: 0.40-0.60 (challenging but learnable)
3. ✅ FL overhead: 5-15% performance drop
4. ✅ DP overhead (ε=1.0): 15-25% additional drop
5. ✅ Extreme non-IID: 15-25% performance degradation
6. ✅ MIA AUC: 0.50-0.55 (good privacy protection)

## Notes

- The current results (98.5% accuracy, AUC-PR 0.998) indicate **perfect overfitting** due to:
  1. Deterministic labels
  2. Balanced classes (38% vs realistic 3%)
  3. Possible data leakage in temporal splits
  
- Real-world mental health prediction is HARD - models should struggle but still show learning

- Lower numbers are NOT bad - they're REALISTIC!
