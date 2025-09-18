# original clean_data.py
# -*- coding: utf-8 -*-
"""
(… your original module docstring …)
"""

# -------------------------
# Imports (unchanged, but grouped)
# -------------------------
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import PowerTransformer, OneHotEncoder

# -------------------------
# Functions (unchanged content)
# -------------------------
def outliers_plotting(col, data, outliers):
    # (your plotting code from earlier refactor)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=30, color="skyblue")
    plt.scatter(outliers, [0]*len(outliers), color="red", label="Outliers", zorder=5)
    plt.title(f"Distribución de {col}")
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data, color="lightgreen")
    plt.title(f"Boxplot de {col}")
    plt.tight_layout()
    plt.show()


def outliers_detection(df, numeric_vars, plot: bool = False):
    # (unchanged detection logic; only calls outliers_plotting if plot=True)
    cols_with_outliers = []
    for col in numeric_vars:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        if len(outliers) > 0:
            cols_with_outliers.append(col)
        if plot:
            outliers_plotting(col, data, outliers)
    return cols_with_outliers


def comparar_transformacion(nombre, original, transformada, metodo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # original
    sns.histplot(np.ravel(original), kde=True, bins=30, color="skyblue", ax=axes[0])
    axes[0].set_title(f"{nombre} (original)")

    # transformada
    sns.histplot(np.ravel(transformada), kde=True, bins=30, color="lightgreen", ax=axes[1])
    axes[1].set_title(f"{nombre} ({metodo})")

    plt.tight_layout()
    plt.show()
    return


def log_1p(nombre, val, plot: bool = False):
    v = np.ravel(val).astype(float)
    t = np.log1p(v)
    if plot:
        comparar_transformacion(nombre, v, t, "log1p")
    return t


def log10_transform(nombre, val, plot: bool = False):
    v = np.ravel(val).astype(float)
    v_safe = np.where(v <= 0, np.nan, v)
    t = np.log10(v_safe)
    if plot:
        comparar_transformacion(nombre, v, t, "Log10")
    return t


def sqrt_transform(nombre, val, plot: bool = False):
    v = np.ravel(val).astype(float)
    v_safe = np.where(v < 0, np.nan, v)
    t = np.sqrt(v_safe)
    if plot:
        comparar_transformacion(nombre, v, t, "Sqrt")
    return t


def yeo_johnson(nombre, val, plot: bool = False):
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(base_no_fraud[nombre].dropna().values.reshape(-1, 1).astype(float))
    t = pt.transform(val.astype(float)).ravel()
    if plot:
        comparar_transformacion(nombre, val, t, "Yeo-Johnson")
    return t


# -------------------------
# Public entry point used by run_all.py
# -------------------------
def run_clean(datacsv: str, output_csv: str):
    """
    Runs the original cleaning pipeline on the provided CSV and writes to output_csv.
    Only file I/O is changed; the pipeline logic remains identical.
    """
    # === I/O setup ===
    in_path = Path(datacsv).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # === BEGIN: original pipeline (unchanged logic; only the read/write lines were adapted) ===

    # Read (was: base = pd.read_csv("Base.csv"))
    base = pd.read_csv(str(in_path))
    base.head()
    base.shape
    print(base.dtypes)

    print("Duplicados:", base.duplicated().sum())
    base.info()

    cat_cols = [
        "fraud_bool","payment_type","employment_status","email_is_free","housing_status",
        "phone_home_valid","phone_mobile_valid","has_other_cards","foreign_request","source",
        "device_os","keep_alive_session","device_fraud_count"
    ]
    num_cols = [
        "income","name_email_similarity","prev_address_months_count","current_address_months_count",
        "customer_age","days_since_request","intended_balcon_amount","zip_count_4w","velocity_6h",
        "velocity_24h","velocity_4w","bank_branch_count_8w","date_of_birth_distinct_emails_4w",
        "credit_risk_score","bank_months_count","proposed_credit_limit","session_length_in_minutes",
        "device_distinct_emails_8w","month"
    ]

    # (… your null-handling, ranges dict, replacement with NaN, negatives count …)
    ranges = {
        "income": (0.1, 0.9),
        "name_email_similarity": (0, 1),
        "prev_address_months_count": (0, 380),
        "current_address_months_count": (0, 429),
        "customer_age": (10, 90),
        "days_since_request": (0, 79),
        "intended_balcon_amount": (0, 114),
        "zip_count_4w": (1, 6830),
        "velocity_6h": (0, 16818),
        "velocity_24h": (1297, 9586),
        "velocity_4w": (2825, 7020),
        "bank_branch_count_8w": (0, 2404),
        "date_of_birth_distinct_emails_4w": (0, 39),
        "credit_risk_score": (0, 389),
        "bank_months_count": (0, 32),
        "proposed_credit_limit": (200, 2000),
        "session_length_in_minutes": (0, 107),
        "device_distinct_emails_8w": (0, 2),
        "month": (0, 7)
    }
    for col, (low, high) in ranges.items():
        if col in base.columns:
            base.loc[(base[col] < low) | (base[col] > high), col] = np.nan
    print("Valores fuera de rango reemplazados por NaN")
    negativos = (base[num_cols] < 0).sum()
    print(negativos)

    # Outliers section (unchanged; uses global base_no_fraud)
    global base_no_fraud
    base_no_fraud = base[base['fraud_bool'] == 0].copy()
    num_cols_nf = [c for c in base_no_fraud.columns if is_numeric_dtype(base_no_fraud[c])]
    cols_to_check = []
    for c in num_cols_nf:
        vals = base_no_fraud[c].dropna().unique()
        if not set(vals).issubset({0, 1}):
            cols_to_check.append(c)
    print("Columnas NUMÉRICAS consideradas (no binarias, solo no-fraude):")
    print(cols_to_check)
    cols_outliers = outliers_detection(base_no_fraud, cols_to_check, plot=False)
    print("Columnas con outliers de no fraude:", cols_outliers)

    # Transformations (unchanged)
    sns.set(style="whitegrid")
    for col in [
        "prev_address_months_count","current_address_months_count","days_since_request",
        "intended_balcon_amount","zip_count_4w","velocity_6h","velocity_24h","bank_branch_count_8w",
        "credit_risk_score"
    ]:
        if col in base.columns:
            m = base[col].notna()
            base.loc[m, col] = yeo_johnson(col, base.loc[m, col].to_numpy().reshape(-1, 1))
    for col in ["date_of_birth_distinct_emails_4w","session_length_in_minutes"]:
        if col in base.columns:
            m = base[col].notna()
            base.loc[m, col] = log_1p(col, base.loc[m, col].to_numpy().reshape(-1, 1))

    # Imputation (unchanged)
    absence_real = [c for c in ["prev_address_months_count","bank_months_count","intended_balcon_amount"]
                    if c in base.columns]
    cols_with_nan = [c for c in num_cols if c in base.columns and base[c].isna().any()]
    for c in cols_with_nan:
        base[f"{c}_was_missing"] = base[c].isna().astype("int8")
    for c in absence_real:
        base[c] = base[c].fillna(0)
    median_cols = [c for c in cols_with_nan if c not in absence_real]
    for c in median_cols:
        base[c] = base[c].fillna(base[c].median())

    # Categorical encoding (unchanged)
    for col in cat_cols:
        if col in base.columns:
            print(base[col].value_counts())
            print('-' * 20)

    cat_cols_ohe = [c for c in ["payment_type","employment_status","housing_status","source","device_os"] if c in base.columns]
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.uint8)
    one_hot = enc.fit_transform(base[cat_cols_ohe]) if cat_cols_ohe else None
    if one_hot is not None:
        one_hot_df = pd.DataFrame(
            one_hot.toarray(),
            columns=enc.get_feature_names_out(cat_cols_ohe),
            index=base.index
        )
        base_clean = pd.concat([base.drop(columns=cat_cols_ohe), one_hot_df], axis=1)
    else:
        base_clean = base.copy()

    print("Dataset limpio listo:", base_clean.shape)

    # === WRITE (was: base_clean.to_csv("base_clean.csv", index=False)) ===
    base_clean.to_csv(str(out_path), index=False)  # change extension to .py if you truly need that

    # === END: original pipeline ===


# Optional: allow direct execution
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run data cleaning on a CSV.")
    p.add_argument("--datacsv", required=True, help="Path to input CSV")
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to ./cleandata/<basename>_clean.csv"
    )
    args = p.parse_args()

    in_path = Path(args.datacsv)
    if args.out is None:
        out_dir = Path("cleandata")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{in_path.stem}_clean.csv"
    else:
        out_path = Path(args.out)

    run_clean(datacsv=str(in_path), output_csv=str(out_path))
