"""
Feature 3 data pipeline: real text data (Suicide_Detection.csv /
depression_dataset_reddit_cleaned.csv), TF-IDF vectorized.

Replaces the synthetic behavioral data generator entirely (dropped per
decision - it failed its own documented signal-quality check, see the
Phase 2 report). Both real text datasets are roughly balanced (~50/50),
unlike the synthetic generator's ~3% positive rate, so none of the
class-imbalance machinery from Phase 2 is needed here.

Reuses the TF-IDF vectorization *approach* from
scripts/robust_validation_pipeline.py (TfidfVectorizer, ngram_range=(1,2),
min_df=2, max_df=0.95, same text cleaning: lowercase + strip punctuation)
rather than the literal models/text_vectorizer.pkl artifact, since that
pickle was fit only on the depression dataset's text as one half of a
deliberate cross-domain train/test split - fitting fresh on this
pipeline's own train split avoids vocabulary/IDF leakage into validation.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "suicide": REPO_ROOT / "datasets" / "Suicide_Detection.csv",
    "depression": REPO_ROOT / "datasets" / "depression_dataset_reddit_cleaned.csv",
}


def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)


def load_suicide_dataset(n_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATASETS["suicide"], usecols=["text", "class"])
    df["cleaned_text"] = _clean_text(df["text"])
    df["label"] = df["class"].map({"suicide": 1, "non-suicide": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if n_samples is not None and n_samples < len(df):
        df, _ = train_test_split(
            df, train_size=n_samples, stratify=df["label"], random_state=seed
        )
    return df[["cleaned_text", "label"]].reset_index(drop=True)


def load_depression_dataset(n_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATASETS["depression"], usecols=["clean_text", "is_depression"])
    df["cleaned_text"] = _clean_text(df["clean_text"])
    df["label"] = df["is_depression"].astype(int)

    if n_samples is not None and n_samples < len(df):
        df, _ = train_test_split(
            df, train_size=n_samples, stratify=df["label"], random_state=seed
        )
    return df[["cleaned_text", "label"]].reset_index(drop=True)


def prepare_tfidf_split(
    dataset: str = "suicide",
    n_samples: Optional[int] = 20000,
    max_features: int = 3000,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer]:
    """Load, split, and TF-IDF vectorize (vectorizer fit on train only).

    Returns (X_train, y_train, X_val, y_val, vectorizer) as dense float32
    arrays - fine at this n_samples/max_features scale (a few hundred MB at
    most); revisit if this pipeline is later scaled to the full 232k-row
    dataset.
    """
    loader = load_suicide_dataset if dataset == "suicide" else load_depression_dataset
    df = loader(n_samples=n_samples, seed=seed)

    train_df, val_df = train_test_split(
        df, test_size=val_frac, stratify=df["label"], random_state=seed
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(train_df["cleaned_text"]).toarray().astype(np.float32)
    X_val = vectorizer.transform(val_df["cleaned_text"]).toarray().astype(np.float32)

    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)

    return X_train, y_train, X_val, y_val, vectorizer


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, vec = prepare_tfidf_split()
    print(f"X_train {X_train.shape}  pos_ratio={y_train.mean():.3f}")
    print(f"X_val   {X_val.shape}  pos_ratio={y_val.mean():.3f}")
    print(f"vocab size {len(vec.vocabulary_)}")
