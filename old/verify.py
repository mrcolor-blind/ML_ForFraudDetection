import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PREP_DIR = Path("prepared_data")

def load(name):
    return joblib.load(PREP_DIR / name)

# --- 1) Normaliza y en binario 0/1 si hace falta ---
def to_binary_labels(y, positive_aliases=("1","true","fraud","fraude","yes","y","si","sí")):
    """
    Acepta y como array/Series con {0,1}, {False,True}, strings ('fraud', 'not_fraud'), etc.
    Devuelve np.ndarray con 0/1.
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.ravel(y)

    # Si ya es {0,1} devolvemos igual
    classes = np.unique(y)
    if set(classes).issubset({0,1}):
        return y.astype(int)

    # Si es booleana
    if y.dtype == bool or set(classes).issubset({True, False}):
        return y.astype(int)

    # Si es numérica distinta de {0,1} (p.ej. {0,2} o {-1,1})
    if np.issubdtype(y.dtype, np.number):
        # mapea al menor->0, mayor->1 (binary)
        uniq = np.unique(y)
        if len(uniq) == 2:
            return (y == uniq.max()).astype(int)

    # Si es string / object
    y_str = pd.Series(y).astype(str).str.strip().str.lower()
    pos = y_str.isin(positive_aliases)
    # Si no pudimos identificar, intenta heurística por palabra clave
    if pos.sum() == 0 and y_str.nunique() == 2:
        # Asigna la clase menos frecuente como positiva (último recurso)
        counts = y_str.value_counts()
        positive_label = counts.idxmin()
        pos = (y_str == positive_label)
    return pos.astype(int).values

# --- 2) Resumen bonito de clases ---
def summarize_labels(y, name="split"):
    y = np.ravel(y)
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    msg = [f"\n=== {name} ===  total={total}"]
    for c, ct in zip(classes, counts):
        pct = (ct/total*100) if total else 0
        msg.append(f"  clase {c}: {ct} ({pct:.2f}%)")
    # Prevalencia de fraude asumiendo 1=fraude
    prev = (y==1).sum()
    msg.append(f"  prevalencia (y==1): {prev} ({(prev/total*100 if total else 0):.3f}%)")
    if prev == 0:
        msg.append("  ⚠️ No hay positivos: ROC indefinido, PR≈0. Revisa tu split.")
    print("\n".join(msg))

# --- 3) Verificador mínimo de positivos ---
def require_min_positives(y, name="split", k_min=50):
    pos = (np.ravel(y) == 1).sum()
    if pos < k_min:
        print(f"⚠️ {name}: solo {pos} positivos (<{k_min}). Considera combinar meses o ajustar ventana temporal.")
    return pos

# === Ejemplo con tus pickles ===
X_train = load("X_train_base.pkl"); y_train = to_binary_labels(load("y_train_base.pkl"))
X_val   = load("X_val_base.pkl");   y_val   = to_binary_labels(load("y_val_base.pkl"))
X_test  = load("X_test_base.pkl");  y_test  = to_binary_labels(load("y_test_base.pkl"))

X_test_v3 = load("X_test_v3.pkl");  y_test_v3 = to_binary_labels(load("y_test_v3.pkl"))
X_test_v5 = load("X_test_v5.pkl");  y_test_v5 = to_binary_labels(load("y_test_v5.pkl"))

for name, y in [
    ("TRAIN base", y_train),
    ("VAL base",   y_val),
    ("TEST base",  y_test),
    ("TEST V3",    y_test_v3),
    ("TEST V5",    y_test_v5),
]:
    summarize_labels(y, name)
    require_min_positives(y, name, k_min=50)   # ajusta k_min a tu gusto
