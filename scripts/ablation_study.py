"""Efficient ablation study over text vs behavioral features on large data.

Optimizations:
- Stratified sampling for very large datasets
- Proper train/test split for unbiased F1
- LinearSVC for sparse text (fast) instead of RandomForest
- Sparse hstack to avoid dense conversions
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
import joblib


# Load prepared data
data = pd.read_csv('multimodal_dataset.csv')

# Downsample if extremely large (keep class balance)
MAX_SAMPLES = 100_000
if len(data) > MAX_SAMPLES:
    data, _ = train_test_split(
        data,
        train_size=MAX_SAMPLES,
        stratify=data['label'],
        random_state=42,
    )

# Split data into train/test
X_text = data['cleaned_text'].astype(str)
X_behavior = data[['screen_time', 'app_switches', 'night_activity']].astype(float)
y = data['label'].astype(int)

X_text_train, X_text_test, X_beh_train, X_beh_test, y_train, y_test = train_test_split(
    X_text, X_behavior, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorize text
text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_text_train_vec = text_vectorizer.fit_transform(X_text_train)
X_text_test_vec = text_vectorizer.transform(X_text_test)

# Scale behavioral features (no centering for sparse compatibility)
scaler = StandardScaler(with_mean=False)
X_beh_train_scaled = scaler.fit_transform(X_beh_train)
X_beh_test_scaled = scaler.transform(X_beh_test)

# Text-only model (fast linear SVM)
text_model = LinearSVC(random_state=42)
text_model.fit(X_text_train_vec, y_train)
text_preds = text_model.predict(X_text_test_vec)
text_f1 = f1_score(y_test, text_preds, average='weighted')

# Behavior-only model (RF is fine with 3 features; keep modest size)
behavior_model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
behavior_model.fit(X_beh_train, y_train)
behavior_preds = behavior_model.predict(X_beh_test)
behavior_f1 = f1_score(y_test, behavior_preds, average='weighted')

# Combined model: sparse hstack of text + behavior
X_train_combined = hstack([X_text_train_vec, csr_matrix(X_beh_train_scaled)])
X_test_combined = hstack([X_text_test_vec, csr_matrix(X_beh_test_scaled)])
combined_model = LinearSVC(random_state=42)
combined_model.fit(X_train_combined, y_train)
combined_preds = combined_model.predict(X_test_combined)
combined_f1 = f1_score(y_test, combined_preds, average='weighted')

# Save results
results = {
    "text_only_f1": float(text_f1),
    "behavior_only_f1": float(behavior_f1),
    "combined_f1": float(combined_f1),
    "improvement": float(combined_f1 - max(text_f1, behavior_f1))
}
joblib.dump(results, 'ablation_results.pkl')

print(f"Text-only F1: {text_f1:.4f}")
print(f"Behavior-only F1: {behavior_f1:.4f}")
print(f"Combined F1: {combined_f1:.4f}")
print(f"Improvement: {results['improvement']:.4f}")