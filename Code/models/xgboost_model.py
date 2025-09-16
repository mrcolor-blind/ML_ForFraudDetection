# models/xgboost_model.py
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)
from xgboost import XGBClassifier
from utils.balancing import balance_train  # <-- NEW

TARGET_COL = "fraud_bool"
MODEL_DIR = Path("models")

def is_categorical_df(df: pd.DataFrame) -> bool:
    if not isinstance(df, pd.DataFrame):
        return False
    return any(pd.api.types.is_categorical_dtype(df[col]) or
               pd.api.types.is_object_dtype(df[col])
               for col in df.columns)

def compute_scale_pos_weight(y):
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))

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

def ensure_categorical_support(x_train: pd.DataFrame, params: dict):
    if is_categorical_df(x_train):
        params["enable_categorical"] = True
        params.setdefault("tree_method", "hist")

def _load_split(prepared_dir: Path, split_name: str) -> pd.DataFrame:
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

def run_xgb(
    prepared_dir: Path,
    resampler: str = "none",   # "none" | "ros" | "smote"
    seed: int = 42,
    smote_k: int = 5,
):
    # Load splits
    v_tr, df_train = _load_split(prepared_dir, "train")
    v_va, df_val   = _load_split(prepared_dir, "val")
    v_te, df_test  = _load_split(prepared_dir, "test")
    if not (v_tr == v_va == v_te):
        print(f"[WARN] Split variants differ: train={v_tr}, val={v_va}, test={v_te}")

    # Build X/y
    X_train, y_train = _xy_from_df(df_train)
    X_val,   y_val   = _xy_from_df(df_val)
    X_test,  y_test  = _xy_from_df(df_test)

    # --- Oversampling ONLY on train (optional) ---
    X_train, y_train = balance_train(
        X_train, y_train, method=resampler, random_state=seed, smote_k=smote_k
    )

    # XGBoost prefers numeric dtypes; ensure float32 for safety (esp. after SMOTE)
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    params = dict(
        n_estimators=5000,
        learning_rate=0.03,
        early_stopping_rounds=200,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    ensure_categorical_support(X_train, params)
    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Evaluate
    suffix = f"{v_tr} | resampler={resampler}"
    evaluate(model, X_train, y_train, f"TRAIN ({suffix})")
    evaluate(model, X_val,   y_val,   f"VAL ({v_va})")
    evaluate(model, X_test,  y_test,  f"TEST ({v_te})")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    variant_token = re.sub(r"\s+", "", (v_tr or 'Base'))
    out_path = MODEL_DIR / f"xgb_{variant_token}.joblib"
    joblib.dump(model, out_path)
    print(f"\nModelo guardado en: {out_path}")
