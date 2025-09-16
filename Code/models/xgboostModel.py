import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)
from xgboost import XGBClassifier


PREP_DIR = Path("prepared_data")

# ---------- Utils ----------
def load_pickle(name):
    return joblib.load(PREP_DIR / name)

def is_categorical_df(df: pd.DataFrame) -> bool:
    if not isinstance(df, pd.DataFrame):
        return False
    return any(pd.api.types.is_categorical_dtype(df[col]) or
               pd.api.types.is_object_dtype(df[col])
               for col in df.columns)

def compute_scale_pos_weight(y):
    # evita división por cero
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))

def evaluate(model, X, y, name="split"):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, proba)
    auc_pr  = average_precision_score(y, proba)  # AP = área PR
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, pred)

    print(f"\n=== {name} ===")
    print(f"AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("Confusion matrix [tn fp; fn tp]:")
    print(cm)
    # opcional, desglose por clase
    # print(classification_report(y, pred, digits=4))

def ensure_categorical_support(x_train, params):
    """
    Si el DataFrame trae columnas categóricas/objetos y XGBoost moderno puede manejarlas,
    activamos enable_categorical=True y tree_method='hist'.
    Si no, recomendamos convertirlas a 'category' antes del preprocesamiento.
    """
    if is_categorical_df(x_train):
        params["enable_categorical"] = True
        # hist es necesario para categóricas nativas
        params.setdefault("tree_method", "hist")

# ---------- Carga de datos ----------
X_train = load_pickle("X_train_base.pkl")
y_train = load_pickle("y_train_base.pkl")
X_val   = load_pickle("X_val_base.pkl")
y_val   = load_pickle("y_val_base.pkl")
X_test  = load_pickle("X_test_base.pkl")
y_test  = load_pickle("y_test_base.pkl")

# (opcionales) tests de otras variantes
X_test_v3 = load_pickle("X_test_v3.pkl")
y_test_v3 = load_pickle("y_test_v3.pkl")
X_test_v5 = load_pickle("X_test_v5.pkl")
y_test_v5 = load_pickle("y_test_v5.pkl")

# Asegurar que y sean vectores 1D
y_train = np.ravel(y_train)
y_val   = np.ravel(y_val)
y_test  = np.ravel(y_test)
y_test_v3 = np.ravel(y_test_v3)
y_test_v5 = np.ravel(y_test_v5)

# ---------- Parámetros base del modelo ----------
scale_pos = compute_scale_pos_weight(y_train)

model = XGBClassifier(
    n_estimators=5000,
    learning_rate=0.03,
    early_stopping_rounds=200,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos,
    objective="binary:logistic",
    eval_metric="aucpr",   # or ["aucpr","auc"]
    random_state=42,
    n_jobs=-1
)

# Pass early_stopping_rounds as an argument to fit()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# ---------- Evaluación ----------
evaluate(model, X_train, y_train, "TRAIN (base)")
evaluate(model, X_val,   y_val,   "VAL (base)")
evaluate(model, X_test,  y_test,  "TEST (base)")
evaluate(model, X_test_v3, y_test_v3, "TEST (Variant III)")
evaluate(model, X_test_v5, y_test_v5, "TEST (Variant V)")

# ---------- Guardar modelo ----------
Path("models").mkdir(exist_ok=True, parents=True)
joblib.dump(model, "models/xgb_base.joblib")
print("\nModelo guardado en: models/xgb_base.joblib")

# ---------- Importancias (opcional rápido por ganancia) ----------
try:
    importances = model.get_booster().get_score(importance_type="gain")
    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print("\nTop 20 features por 'gain':")
    for k, v in top:
        print(f"{k:30s} {v:.4f}")
except Exception as e:
    print(f"(No se pudieron extraer importancias: {e})")

