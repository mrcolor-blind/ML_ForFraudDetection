# utils/balancing.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _as_dataframe(X, columns):
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns)

def balance_train(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    method: str = "none",          # "none" | "ros" | "smote"
    random_state: int = 42,
    smote_k: int = 5,
    minority_label: int = 1,
):
    """
    Returns (X_bal, y_bal). Applies ONLY to TRAIN data.
    - none: no change
    - ros : RandomOverSampler to balance classes
    - smote: SMOTE (k_neighbors=smote_k)
    """
    method = (method or "none").lower()
    if method == "none":
        return X, y

    # Lazy imports so other stages don't need imblearn installed
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
    except Exception as e:
        raise ImportError(
            "imblearn is required for resampling. Install with: pip install imbalanced-learn"
        ) from e

    columns = list(X.columns) if isinstance(X, pd.DataFrame) else None

    if method == "ros":
        ros = RandomOverSampler(random_state=random_state)
        Xb, yb = ros.fit_resample(X, y)
    elif method == "smote":
        sm = SMOTE(
            sampling_strategy="auto",
            k_neighbors=smote_k,
            random_state=random_state,
        )
        Xb, yb = sm.fit_resample(X, y)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    # Preserve DataFrame structure if input was a DataFrame
    Xb = _as_dataframe(Xb, columns or [f"f{i}" for i in range(np.asarray(Xb).shape[1])])
    return Xb, yb
