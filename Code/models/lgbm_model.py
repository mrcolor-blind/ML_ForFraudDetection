# models/lgbm_model.py

# ============================
# Imports
# ============================
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, recall_score, roc_curve, confusion_matrix

# ============================
# Helpers (prepared-splits I/O)
# ============================
TARGET_COL = "fraud_bool"

def _load_split(prepared_dir: Path, split_name: str):
    """
    Loads {'variant': <str>, 'data': <DataFrame>} for split in {train,val,test}
    from files matching '*_<split>.pkl' inside prepared_dir.
    """
    matches = list(Path(prepared_dir).glob(f"*_{split_name}.pkl"))
    if not matches:
        raise FileNotFoundError(f"No '*_{split_name}.pkl' found in {prepared_dir}")
    matches.sort()
    payload = joblib.load(matches[0])
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Bad payload in {matches[0]} (expected dict with 'data').")
    return payload["variant"], payload["data"]

def _xy_from_df(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])
    return X, y

def _ensure_results_dir() -> Path:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    return out

def _plot_learning_curve(evals_result: dict, name_map: dict, title: str, out_png: Path):
    """
    Plot AUC per iteration for each dataset in evals_result.
    name_map maps raw dataset names -> pretty labels (e.g., {'valid':'Val','train':'Train'})
    """
    # LightGBM stores metrics as: evals_result[dataset_name][metric_name] -> list
    plt.figure(figsize=(8,5))
    for raw, pretty in name_map.items():
        if raw in evals_result and "auc" in evals_result[raw]:
            plt.plot(evals_result[raw]["auc"], label=pretty)
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)

# ============================
# Core (logic preserved)
# ============================
def run_lgbm(prepared_dir: Path):
    """
    Reads prepared splits (train/val/test), trains base & alt LightGBM models
    (same params/logic as provided), selects threshold on VAL at 5% FPR,
    evaluates on TEST, saves results .txt and learning-curve PNGs.
    """
    # --- Load prepared splits ---
    v_tr, df_train = _load_split(prepared_dir, "train")
    v_va, df_val   = _load_split(prepared_dir, "val")
    v_te, df_test  = _load_split(prepared_dir, "test")

    X_train, y_train = _xy_from_df(df_train)
    X_valid, y_valid = _xy_from_df(df_val)
    X_test,  y_test  = _xy_from_df(df_test)

    # --- Datasets for LightGBM ---
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    # --- Params (unchanged) ---
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "is_unbalance": True,
        "verbosity": -1,
        "seed": 42,
        "num_threads": 0
    }

    # --- Train BASE model (early stopping on VALID) ---
    evals_result_base = {}
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid, dtrain],      # keep VALID first for same early-stopping behavior
        valid_names=["valid", "train"],   # so we can plot both curves
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.record_evaluation(evals_result_base)
        ]
    )

    # --- Threshold at 5% FPR chosen on VALID, evaluate on TEST (unchanged logic) ---
    p_v = model.predict(X_valid, num_iteration=model.best_iteration)
    fpr_v, tpr_v, thr_v = roc_curve(y_valid, p_v)
    idx = (fpr_v <= 0.05).nonzero()[0]
    thr_5fpr = thr_v[idx[-1]] if len(idx) else 1.0

    p_t  = model.predict(X_test,  num_iteration=model.best_iteration)
    auc  = roc_auc_score(y_test, p_t)
    yhat = (p_t >= thr_5fpr).astype(int)
    rec  = recall_score(y_test, yhat)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat, labels=[0,1]).ravel()
    fpr_real = fp / (fp + tn) if (fp + tn) else 0.0

    base_line = (
        f"[BASE] AUC={auc:.4f}  Recall@5%FPR={rec:.4f}  FPR_real={fpr_real:.4f}  "
        f"thr(valid)={thr_5fpr:.6f}\n"
        f"[BASE] best_iteration={model.best_iteration}\n"
    )

    # --- Train ALT model (unchanged) ---
    params_alt = params.copy()
    params_alt.update({"num_leaves": 63, "min_data_in_leaf": 80, "lambda_l2": 2.0})

    evals_result_alt = {}
    model_alt = lgb.train(
        params_alt,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid, dtrain],      # VALID first, TRAIN second (for curves only)
        valid_names=["valid", "train"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.record_evaluation(evals_result_alt)
        ]
    )

    p_v2 = model_alt.predict(X_valid, num_iteration=model_alt.best_iteration)
    fpr_v2, tpr_v2, thr_v2 = roc_curve(y_valid, p_v2)
    idx2 = (fpr_v2 <= 0.05).nonzero()[0]
    thr_5fpr_alt = thr_v2[idx2[-1]] if len(idx2) else 1.0

    p_t2  = model_alt.predict(X_test,  num_iteration=model_alt.best_iteration)
    auc2  = roc_auc_score(y_test, p_t2)
    rec2  = recall_score(y_test, (p_t2 >= thr_5fpr_alt).astype(int))

    alt_line = (
        f"[ALT ] AUC={auc2:.4f}  Recall@5%FPR={rec2:.4f}  "
        f"thr(valid)={thr_5fpr_alt:.6f}\n"
        f"[ALT ] best_iteration={model_alt.best_iteration}\n"
    )

    # --- Save results (.txt) ---
    results_dir = _ensure_results_dir()
    txt_path = results_dir / "lgbm_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(base_line)
        f.write(alt_line)
    print(base_line.strip())
    print(alt_line.strip())
    print(f"[OK] Saved: {txt_path}")

    # --- Save learning curves (Train vs Val AUC) for each model ---
    _plot_learning_curve(
        evals_result_base,
        name_map={"train": "Train AUC", "valid": "Val AUC"},
        title="LightGBM Learning Curve (BASE)",
        out_png=results_dir / "lgbm_learning_curve_base.png"
    )
    _plot_learning_curve(
        evals_result_alt,
        name_map={"train": "Train AUC", "valid": "Val AUC"},
        title="LightGBM Learning Curve (ALT)",
        out_png=results_dir / "lgbm_learning_curve_alt.png"
    )

    # Return artifacts if needed by callers
    return {
        "base": {"model": model, "evals": evals_result_base, "thr_valid_5fpr": thr_5fpr, "auc_test": auc, "recall_test": rec},
        "alt":  {"model": model_alt, "evals": evals_result_alt, "thr_valid_5fpr": thr_5fpr_alt, "auc_test": auc2, "recall_test": rec2},
        "results_path": str(txt_path)
    }

# ============================
# CLI entry (optional)
# ============================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LightGBM on prepared splits.")
    parser.add_argument("--preparedpath", type=str, required=True,
                        help="Path to prepared variant folder, e.g., prepared_data/prepared_base")
    args = parser.parse_args()
    prepared_dir = Path(args.preparedpath).expanduser().resolve()
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared path not found: {prepared_dir}")
    run_lgbm(prepared_dir)

if __name__ == "__main__":
    main()
