"""Class-imbalance helpers shared by train and optimize."""

from __future__ import annotations

import pandas as pd
from sklearn.utils import resample, shuffle


def random_oversample_binary(
    X: pd.DataFrame, y: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Balance binary labels by randomly oversampling the minority class (with replacement)."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    counts = y.value_counts()
    if len(counts) < 2:
        return X, y
    majority_n = counts.max()
    minority_n = counts.min()
    if minority_n >= majority_n:
        return X, y
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    mask_maj = y == majority_label
    mask_min = y == minority_label
    X_minor = X.loc[mask_min]
    y_minor = y.loc[mask_min]
    X_minor_up = resample(
        X_minor,
        replace=True,
        n_samples=majority_n,
        random_state=random_state,
    )
    y_minor_up = resample(
        y_minor,
        replace=True,
        n_samples=majority_n,
        random_state=random_state,
    )
    X_bal = pd.concat([X.loc[mask_maj], X_minor_up], ignore_index=True)
    y_bal = pd.concat([y.loc[mask_maj], y_minor_up], ignore_index=True)
    return shuffle(X_bal, y_bal, random_state=random_state)
