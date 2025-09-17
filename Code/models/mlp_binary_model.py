# models/mlp_binary_model.py
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score


# ============================
# Helpers to integrate prepared splits
# ============================
TARGET_COL = "fraud_bool"

def _load_split(prepared_dir: Path, split_name: str):
    """
    Loads a prepared pickle dict {'variant': <str>, 'data': <DataFrame>} for split {train,val,test}.
    Accepts any file matching '*_<split>.pkl'.
    """
    matches = list(Path(prepared_dir).glob(f"*_{split_name}.pkl"))
    if not matches:
        raise FileNotFoundError(f"No '*_{split_name}.pkl' found in {prepared_dir}")
    matches.sort()
    payload = joblib.load(matches[0])
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Bad payload in {matches[0]}: expected dict with 'data' key.")
    return payload["variant"], payload["data"]

def _xy_from_df(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in prepared data.")
    y = df[TARGET_COL].values.astype(np.int64)
    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    return X, y

def _ensure_results_dir():
    Path("results").mkdir(parents=True, exist_ok=True)


# ============================
# Original model & functions (logic preserved, no W&B)
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
        x = self.fc3(x)
        return x


# Evaluación (extendida para incluir métricas pos=1)
def evaluate(model, loader, mode="eval", device="cpu", criterion=nn.CrossEntropyLoss()):
    if mode == "eval":
        model.eval()
    else:
        model.train()

    correct = 0
    total = 0
    loss_total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = correct / total if total > 0 else float("nan")
    loss_avg = loss_total / len(loader) if len(loader) > 0 else float("nan")

    # Weighted metrics
    f1_w = f1_score(y_true, y_pred, average="weighted") if len(y_true) else float("nan")
    rec_w = recall_score(y_true, y_pred, average="weighted") if len(y_true) else float("nan")

    # Positive-class (fraude=1) metrics
    try:
        f1_pos = f1_score(y_true, y_pred, pos_label=1)
        rec_pos = recall_score(y_true, y_pred, pos_label=1)
    except Exception:
        f1_pos, rec_pos = float("nan"), float("nan")

    # Return both weighted + positive-class metrics
    return loss_avg, acc, f1_w, rec_w, f1_pos, rec_pos


# ============================
# Plotting helpers (save PNGs)
# ============================
def plot_accuracies(train_acc_dropout, train_acc_eval, val_acc, dropout, weight_decay):
    _ensure_results_dir()
    out_path = Path("results") / "mlp_accuracy.png"
    plt.figure(figsize=(8,5))
    plt.plot(train_acc_dropout, label="Train (con dropout)")
    plt.plot(train_acc_eval, label="Train (sin dropout)")
    plt.plot(val_acc, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title(f"Fraude Tabular - Dropout={dropout}, L2={weight_decay}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)

def plot_two_curves(train_vals, val_vals, ylabel, title, filename):
    _ensure_results_dir()
    out_path = Path("results") / filename
    plt.figure(figsize=(8,5))
    plt.plot(train_vals, label=f"Train {ylabel}")
    plt.plot(val_vals, label=f"Val {ylabel}")
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


# ============================
# Trainer (logic preserved; now tracks extra histories)
# ============================
def clasificador_multiclase(X_train, X_val, y_train, y_val,
                            dropout=0.0, lr=0.001, batch_size=64, epochs=3, weight_decay=0.0):
    # Tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_tensor   = torch.tensor(y_val,   dtype=torch.long)

    # Datasets y loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor,   y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1000, shuffle=False)

    # Modelo / dispositivo
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetCSV(input_dim, dropout_rate=dropout).to(device)

    # Pérdida / Optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Históricos
    train_acc_dropout = []
    train_f1_dropout = []
    train_recall_dropout = []

    train_acc_eval = []
    train_f1_eval = []
    train_recall_eval = []

    val_acc = []
    val_f1 = []
    val_recall = []

    # Extra: para curvas train-vs-val de loss/f1 (eval mode) y learning curves pos=1
    train_loss_eval_hist = []
    val_loss_hist = []
    train_f1_pos_hist = []
    val_f1_pos_hist = []
    train_rec_pos_hist = []
    val_rec_pos_hist = []

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Métricas por época
        tr_loss_d, tr_acc_d, tr_f1_d, tr_rec_d, tr_f1_pos_d, tr_rec_pos_d = evaluate(
            model, train_loader, mode="train", device=device, criterion=criterion
        )
        tr_loss_e, tr_acc_e, tr_f1_e, tr_rec_e, tr_f1_pos_e, tr_rec_pos_e = evaluate(
            model, train_loader, mode="eval", device=device, criterion=criterion
        )
        va_loss, va_acc_epoch, va_f1_epoch, va_rec_epoch, va_f1_pos, va_rec_pos = evaluate(
            model, val_loader, mode="eval", device=device, criterion=criterion
        )

        # Registros originales
        train_acc_dropout.append(tr_acc_d)
        train_f1_dropout.append(tr_f1_d)
        train_recall_dropout.append(tr_rec_d)

        train_acc_eval.append(tr_acc_e)
        train_f1_eval.append(tr_f1_e)
        train_recall_eval.append(tr_rec_e)

        val_acc.append(va_acc_epoch)
        val_f1.append(va_f1_epoch)
        val_recall.append(va_rec_epoch)

        # Registros extra para curvas
        train_loss_eval_hist.append(tr_loss_e)
        val_loss_hist.append(va_loss)
        train_f1_pos_hist.append(tr_f1_pos_e)
        val_f1_pos_hist.append(va_f1_pos)
        train_rec_pos_hist.append(tr_rec_pos_e)
        val_rec_pos_hist.append(va_rec_pos)

        print(f"Epoch {epoch+1}: "
              f"Train acc (dropout)={tr_acc_d:.4f}, f1={tr_f1_d:.4f}, rec={tr_rec_d:.4f} | "
              f"Train acc (eval)={tr_acc_e:.4f}, f1={tr_f1_e:.4f}, rec={tr_rec_e:.4f} | "
              f"Val acc={va_acc_epoch:.4f}, f1={va_f1_epoch:.4f}, rec={va_rec_epoch:.4f}")

    # Guardar las curvas:
    # 1) Accuracy (tres líneas: dropout/eval/val)
    plot_accuracies(train_acc_dropout, train_acc_eval, val_acc, dropout=dropout, weight_decay=weight_decay)

    # 2) Loss: train-eval vs val
    plot_two_curves(
        train_loss_eval_hist,
        val_loss_hist,
        ylabel="Loss",
        title="Curva de Loss (Train-eval vs Val)",
        filename="mlp_loss.png"
    )

    # 3) F1 (weighted): train-eval vs val
    plot_two_curves(
        train_f1_eval,
        val_f1,
        ylabel="F1 (weighted)",
        title="Curva de F1 (weighted) — Train-eval vs Val",
        filename="mlp_f1_weighted.png"
    )

    # 4) Learning curves positivos (pos=1): F1 y Recall — train vs val
    plot_two_curves(
        train_f1_pos_hist,
        val_f1_pos_hist,
        ylabel="F1 (pos=1)",
        title="Learning Curve — F1 (pos=1) Train vs Val",
        filename="mlp_learningcurve_f1_pos.png"
    )
    plot_two_curves(
        train_rec_pos_hist,
        val_rec_pos_hist,
        ylabel="Recall (pos=1)",
        title="Learning Curve — Recall (pos=1) Train vs Val",
        filename="mlp_learningcurve_recall_pos.png"
    )

    return model


# ============================
# Runner to be called from run.py --mode mlp
# ============================
def run_mlp(prepared_dir: Path,
            dropout=0.0,
            lr=0.001,
            batch_size=64,
            epochs=3,
            weight_decay=0.0):
    """
    Loads prepared splits from `prepared_dir`, uses VAL during training,
    saves accuracy + learning curves PNGs and a TXT with final TEST metrics in ./results/.
    """
    # Load prepared splits
    v_tr, df_train = _load_split(prepared_dir, "train")
    v_va, df_val   = _load_split(prepared_dir, "val")
    v_te, df_test  = _load_split(prepared_dir, "test")

    X_train, y_train = _xy_from_df(df_train)
    X_val,   y_val   = _xy_from_df(df_val)
    X_test,  y_test  = _xy_from_df(df_test)

    # Train with VAL used during epochs
    model = clasificador_multiclase(
        X_train, X_val, y_train, y_val,
        dropout=dropout, lr=lr, batch_size=batch_size, epochs=epochs, weight_decay=weight_decay
    )

    # Final TEST evaluation (and save TXT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    ), batch_size=1024, shuffle=False)

    te_loss, te_acc, te_f1_w, te_rec_w, te_f1_pos, te_rec_pos = evaluate(
        model, test_loader, mode="eval", device=device
    )

    _ensure_results_dir()
    txt_path = Path("results") / "mlp_test_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("==== MLP TEST RESULTS ====\n")
        f.write(f"Loss: {te_loss:.6f}\n")
        f.write(f"Accuracy: {te_acc:.6f}\n")
        f.write(f"F1 (weighted): {te_f1_w:.6f}\n")
        f.write(f"Recall (weighted): {te_rec_w:.6f}\n")
        f.write(f"F1 (pos=1): {te_f1_pos:.6f}\n")
        f.write(f"Recall (pos=1): {te_rec_pos:.6f}\n")

    print(f"[OK] Saved test results to: {txt_path}")
