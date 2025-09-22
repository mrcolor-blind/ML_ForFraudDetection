# models/random_forest_model.py
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)
from utils.balancing import balance_train

TARGET_COL = "fraud_bool"
MODEL_DIR = Path("models")

def _load_split(prepared_dir: Path, split_name: str):
    matches = list(prepared_dir.glob(f"*_{split_name}.pkl"))
    if not matches:
        raise FileNotFoundError(f"No '*_{split_name}.pkl' found in {prepared_dir}")
    matches.sort()
    payload = joblib.load(matches[0])
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Bad payload in {matches[0]}: expected dict with 'data' key.")
    return payload["variant"], payload["data"]

def _xy_from_df(df: pd.DataFrame, target_col: str = TARGET_COL):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in prepared data.")
    y = np.ravel(df[target_col].values)
    X = df.drop(columns=[target_col])
    return X, y

def evaluate(model, X, y, name="split"):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc_roc = roc_auc_score(y, proba)
    auc_pr  = average_precision_score(y, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, pred)
    print(f"\n=== {name} ===")
    print(f"AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("Confusion matrix [tn fp; fn tp]:")
    print(cm)


def run_rf(
    prepared_dir: Path,
    resampler: str = "none",   # "none" | "ros" | "smote"
    seed: int = 42,
    smote_k: int = 5,
):
    # Load splits (variant consistency warning like in XGB)
    v_tr, df_train = _load_split(prepared_dir, "train")
    v_va, df_val   = _load_split(prepared_dir, "val")
    v_te, df_test  = _load_split(prepared_dir, "test")
    if not (v_tr == v_va == v_te):
        print(f"[WARN] Split variants differ: train={v_tr}, val={v_va}, test={v_te}")

    # Build X/y
    X_train, y_train = _xy_from_df(df_train)
    X_val,   y_val   = _xy_from_df(df_val)
    X_test,  y_test  = _xy_from_df(df_test)

    # Optional resampling: TRAIN ONLY
    X_train, y_train = balance_train(
        X_train, y_train, method=resampler, random_state=seed, smote_k=smote_k
    )

    # Ensure numeric matrix (RF ignores categorical dtype; cast to float32 for safety)
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",  # still useful even after mild oversampling; okay to keep
    )
    model.fit(X_train, y_train)

    # Evaluate
    suffix = f"{v_tr} | resampler={resampler}"
    evaluate(model, X_train, y_train, f"TRAIN ({suffix})")
    evaluate(model, X_val,   y_val,   f"VAL ({v_va})")
    evaluate(model, X_test,  y_test,  f"TEST ({v_te})")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    variant_token = re.sub(r"\s+", "", (v_tr or 'Base'))
    out_path = MODEL_DIR / f"rf_{variant_token}.joblib"
    joblib.dump(model, out_path)
    print(f"\nModelo guardado en: {out_path}")