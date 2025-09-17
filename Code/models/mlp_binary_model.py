# new mlp version with balancing
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import wandb

# NEW: reuse shared oversampling utility
from utils.balancing import balance_train


# ============================
# Pipeline integration helpers
# ============================
TARGET_COL = "fraud_bool"

def _load_split(prepared_dir: Path, split_name: str):
    """
    Loads a prepared pickle dict {'variant': <str>, 'data': <DataFrame>} for split_name in {train,val,test}.
    Accepts any filename in the folder that matches '*_<split>.pkl' (e.g., 'Base_train.pkl').
    """
    matches = list(Path(prepared_dir).glob(f"*_{split_name}.pkl"))
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
    y = np.ravel(df[target_col].values.astype(np.int64))
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    return X, y


# ============================
# Modelo
# ============================
class NetCSV(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.0):
        super(NetCSV, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 clases: fraude/no fraude
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # logits
        return x


# ============================
# Evaluación
# ============================
@torch.no_grad()
def evaluate(model, loader, mode="eval", device="cpu", criterion=None):
    if mode == "eval":
        model.eval()
    else:
        model.train()  # eval con dropout activo si se desea comparar (sin grad)

    total_loss = 0.0
    n_batches = 0

    y_true = []
    y_pred = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)

        if criterion is not None:
            loss = criterion(logits, yb)
            total_loss += float(loss.item())
            n_batches += 1

        preds = torch.argmax(logits, dim=1)

        y_true.extend(yb.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    rec_w = recall_score(y_true, y_pred, average="weighted")

    # Métrica específica para la clase positiva (fraude=1)
    try:
        rec_pos = recall_score(y_true, y_pred, pos_label=1)
        f1_pos = f1_score(y_true, y_pred, pos_label=1)
    except Exception:
        rec_pos, f1_pos = np.nan, np.nan

    loss_avg = (total_loss / n_batches) if n_batches > 0 else np.nan
    return loss_avg, acc, f1_w, rec_w, f1_pos, rec_pos


# ============================
# Gráficas (guardado en /results)
# ============================
def _ensure_results_dir():
    os.makedirs("results", exist_ok=True)
    return Path("results")

def plot_accuracies(train_acc_eval, val_acc, title, filename="mlp_accuracy.png"):
    out_path = _ensure_results_dir() / filename
    plt.figure(figsize=(8,5))
    plt.plot(train_acc_eval, label="Train (eval: sin dropout)")
    plt.plot(val_acc, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)

def plot_train_val_curves(
    train_loss, val_loss,
    train_acc, val_acc,
    train_f1w, val_f1w,
    filename="mlp_train_val_loss_acc_f1.png",
    title="Curvas Train vs Val (Loss / Accuracy / F1-weighted)"
):
    out_path = _ensure_results_dir() / filename
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12,8))

    # Loss
    plt.subplot(3,1,1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    # Accuracy
    plt.subplot(3,1,2)
    plt.plot(epochs, train_acc, label="Train acc")
    plt.plot(epochs, val_acc, label="Val acc")
    plt.ylabel("Accuracy")
    plt.legend()

    # F1 weighted
    plt.subplot(3,1,3)
    plt.plot(epochs, train_f1w, label="Train F1 (weighted)")
    plt.plot(epochs, val_f1w, label="Val F1 (weighted)")
    plt.xlabel("Época")
    plt.ylabel("F1 (weighted)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)

def plot_learning_curves_pos(
    train_f1_pos, val_f1_pos,
    train_rec_pos, val_rec_pos,
    filename="mlp_learning_curves_f1_recall_pos.png",
    title="Learning Curves (Clase Positiva: F1 / Recall)"
):
    out_path = _ensure_results_dir() / filename
    epochs = range(1, len(train_f1_pos) + 1)

    plt.figure(figsize=(12,6))

    # F1 (pos_label=1)
    plt.subplot(2,1,1)
    plt.plot(epochs, train_f1_pos, label="Train F1 (pos=1)")
    plt.plot(epochs, val_f1_pos, label="Val F1 (pos=1)")
    plt.ylabel("F1 (pos=1)")
    plt.title(title)
    plt.legend()

    # Recall (pos_label=1)
    plt.subplot(2,1,2)
    plt.plot(epochs, train_rec_pos, label="Train Recall (pos=1)")
    plt.plot(epochs, val_rec_pos, label="Val Recall (pos=1)")
    plt.xlabel("Época")
    plt.ylabel("Recall (pos=1)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


# ============================
# Helper: class weights
# ============================
def compute_class_weights(y_train, n_classes=2):
    """
    Retorna pesos inversamente proporcionales a la frecuencia de clase.
    Fórmula típica: weight_c = N_total / (n_clases * N_c)
    """
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    total = counts.sum()
    counts[counts == 0] = 1.0
    weights = total / (n_classes * counts)
    return weights


# ============================
# Entrenador principal (3 conjuntos)
# ============================
def clasificador_binario_tres_splits(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    project_name="fraude-tabular",
    run_name="baseline_classweights",
    dropout=0.2,
    lr=1e-3,
    batch_size=256,
    epochs=30,
    weight_decay=1e-4,
    use_class_weights=True,
    early_stopping_patience=5
):
    """
    Entrena un MLP binario con train/val/test.
    - Pondera clases en la función de pérdida (opcional).
    - Early stopping por F1 (weighted) de validación.
    - Devuelve el modelo (cargado con el mejor checkpoint) y métricas finales.
    """

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "use_class_weights": use_class_weights,
            "early_stopping_patience": early_stopping_patience
        },
        reinit=True,
    )
    config = wandb.config

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=1024, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=1024, shuffle=False)

    # Modelo
    input_dim = X_train.shape[1]
    model = NetCSV(input_dim, dropout_rate=config.dropout).to(device)

    # Pérdida con/ sin class weights
    if use_class_weights:
        weights_np = compute_class_weights(y_train, n_classes=2)
        class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Early stopping (por F1 de validación)
    best_val_f1 = -np.inf
    best_state = None
    patience = 0

    # Históricos para gráficas
    train_acc_eval_hist = []
    val_acc_hist = []

    # NUEVO: históricos extra
    train_loss_hist = []
    val_loss_hist = []
    train_f1w_hist = []
    val_f1w_hist = []
    train_f1_pos_hist = []
    val_f1_pos_hist = []
    train_rec_pos_hist = []
    val_rec_pos_hist = []

    for epoch in range(config.epochs):
        # Entrenamiento
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Evaluación (train en modo eval para curva estable)
        tr_loss_e, tr_acc_e, tr_f1_e, tr_rec_e, tr_f1_pos, tr_rec_pos = evaluate(
            model, train_loader, mode="eval", device=device, criterion=criterion
        )
        va_loss, va_acc, va_f1, va_rec, va_f1_pos, va_rec_pos = evaluate(
            model, val_loader, mode="eval", device=device, criterion=criterion
        )

        # Logs W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_eval": tr_loss_e,
            "train_acc_eval": tr_acc_e,
            "train_f1_eval_weighted": tr_f1_e,
            "train_recall_eval_weighted": tr_rec_e,
            "train_f1_eval_pos": tr_f1_pos,
            "train_recall_eval_pos": tr_rec_pos,

            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1_weighted": va_f1,
            "val_recall_weighted": va_rec,
            "val_f1_pos": va_f1_pos,
            "val_recall_pos": va_rec_pos,
        })

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train (eval) Acc={tr_acc_e:.4f} F1w={tr_f1_e:.4f} Rec_w={tr_rec_e:.4f} | "
            f"Val Acc={va_acc:.4f} F1w={va_f1:.4f} Rec_w={va_rec:.4f} F1_pos={va_f1_pos:.4f} Rec_pos={va_rec_pos:.4f}"
        )

        # Históricos existentes
        train_acc_eval_hist.append(tr_acc_e)
        val_acc_hist.append(va_acc)

        # NUEVO: agregar a históricos extra
        train_loss_hist.append(tr_loss_e)
        val_loss_hist.append(va_loss)
        train_f1w_hist.append(tr_f1_e)
        val_f1w_hist.append(va_f1)
        train_f1_pos_hist.append(tr_f1_pos)
        val_f1_pos_hist.append(va_f1_pos)
        train_rec_pos_hist.append(tr_rec_pos)
        val_rec_pos_hist.append(va_rec_pos)

        # Early stopping por F1_weighted de validación
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping en epoch {epoch+1} (mejor F1w val = {best_val_f1:.4f})")
                break

    # Cargar mejor estado y evaluar en TEST
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_f1w, te_recw, te_f1_pos, te_rec_pos = evaluate(
        model, test_loader, mode="eval", device=device, criterion=criterion
    )

    print(
        f"TEST | Acc={te_acc:.4f} F1w={te_f1w:.4f} Rec_w={te_recw:.4f} "
        f"F1_pos={te_f1_pos:.4f} Rec_pos={te_rec_pos:.4f}"
    )

    wandb.log({
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1_weighted": te_f1w,
        "test_recall_weighted": te_recw,
        "test_f1_pos": te_f1_pos,
        "test_recall_pos": te_rec_pos
    })
    wandb.finish()

    # === Guardado de resultados y gráficas ===
    # 1) Gráfica original de accuracy (train eval vs val)
    plot_accuracies(
        train_acc_eval_hist,
        val_acc_hist,
        title=f"Fraude (Tabular) — Dropout={config.dropout}, L2={config.weight_decay}",
        filename="mlp_accuracy.png"
    )

    # 2) Curvas Train vs Val: Loss / Accuracy / F1-weighted
    plot_train_val_curves(
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        train_acc=train_acc_eval_hist,
        val_acc=val_acc_hist,
        train_f1w=train_f1w_hist,
        val_f1w=val_f1w_hist,
        filename="mlp_train_val_loss_acc_f1.png",
        title="Curvas Train vs Val (Loss / Accuracy / F1-weighted)"
    )

    # 3) Learning Curves para clase positiva (F1 y Recall)
    plot_learning_curves_pos(
        train_f1_pos=train_f1_pos_hist,
        val_f1_pos=val_f1_pos_hist,
        train_rec_pos=train_rec_pos_hist,
        val_rec_pos=val_rec_pos_hist,
        filename="mlp_learning_curves_f1_recall_pos.png",
        title="Learning Curves (Clase Positiva: F1 / Recall)"
    )

    # 4) Guardar métricas de TEST a .txt
    results_dir = _ensure_results_dir()
    with open(results_dir / "mlp_test_metrics.txt", "w", encoding="utf-8") as f:
        f.write(
            "TEST RESULTS\n"
            f"Loss: {te_loss:.6f}\n"
            f"Accuracy: {te_acc:.6f}\n"
            f"F1 (weighted): {te_f1w:.6f}\n"
            f"Recall (weighted): {te_recw:.6f}\n"
            f"F1 (pos=1): {te_f1_pos:.6f}\n"
            f"Recall (pos=1): {te_rec_pos:.6f}\n"
        )

    return model, {
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1_weighted": te_f1w,
        "test_recall_weighted": te_recw,
        "test_f1_pos": te_f1_pos,
        "test_recall_pos": te_rec_pos
    }


# ============================
# Runner to integrate with run.py  (NOW WITH RESAMPLING)
# ============================
def run_mlp(prepared_dir: Path,
            project_name="fraude-mpl-tabular",
            run_name="mlp_baseline_classweights",
            dropout=0.2,
            lr=1e-3,
            batch_size=256,
            epochs=30,
            weight_decay=1e-4,
            use_class_weights=True,
            early_stopping_patience=5,
            # NEW: resampling controls (TRAIN only)
            resampler: str = "none",   # "none" | "ros" | "smote"
            seed: int = 42,
            smote_k: int = 5):
    """
    Loads prepared splits from `prepared_dir`, optionally oversamples TRAIN only,
    and runs the MLP. Validation and test remain untouched.
    """
    v_tr, df_train = _load_split(prepared_dir, "train")
    v_va, df_val   = _load_split(prepared_dir, "val")
    v_te, df_test  = _load_split(prepared_dir, "test")

    X_train, y_train = _xy_from_df(df_train)
    X_val,   y_val   = _xy_from_df(df_val)
    X_test,  y_test  = _xy_from_df(df_test)

    # === Apply oversampling on TRAIN only ===
    X_train_df = pd.DataFrame(X_train)
    y_train_s  = pd.Series(y_train)
    X_train_bal, y_train_bal = balance_train(
        X_train_df, y_train_s, method=resampler, random_state=seed, smote_k=smote_k
    )
    X_train = X_train_bal.values.astype(np.float32)
    y_train = y_train_bal.values.astype(np.int64)

    return clasificador_binario_tres_splits(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        project_name=project_name,
        run_name=run_name,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        use_class_weights=use_class_weights,
        early_stopping_patience=early_stopping_patience,
    )
