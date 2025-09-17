# run.py
import argparse
from pathlib import Path
import re
import pickle
import pandas as pd
import clean_data

# Model runners
from models import xgboost_model
from models import mlp_binary_model
from models import lgbm_model  # NEW: LightGBM module


# ---------------------------
# Helpers
# ---------------------------
def infer_variant_name(path: Path) -> str:
    stem = path.stem
    token = re.sub(r"[^A-Za-z0-9]+", "", stem).lower()

    # combined/union datasets
    if "unido" in token or "combined" in token:
        return "Unido"

    if token.startswith("base"):
        return "Base"

    roman_map = {"i": "Variant I", "ii": "Variant II", "iii": "Variant III", "iv": "Variant IV", "v": "Variant V"}
    m = re.match(r"(?:variant|var)(i{1,3}|iv|v|1|2|3|4|5)", token)
    if m:
        grp = m.group(1)
        if grp.isdigit():
            return f"Variant {'I' * int(grp)}"
        return roman_map.get(grp.lower(), f"Variant {grp.upper()}")

    base = stem.split("_clean")[0] if "_clean" in stem else stem
    return base or "Base"


def variant_safe_name(variant: str) -> str:
    """'Variant III' -> 'VariantIII', 'Base' -> 'Base'."""
    return re.sub(r"\s+", "", variant)


def split_by_month(df: pd.DataFrame):
    if "month" not in df.columns:
        raise KeyError("Column 'month' not found in input data.")

    m = df["month"]
    uses_zero_based = m.min() == 0

    if uses_zero_based:
        train = df[m.between(0, 4)]
        val = df[m == 5]
        test = df[m.between(6, 7)]
    else:
        train = df[m.between(1, 5)]
        val = df[m == 6]
        test = df[m.between(7, 8)]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("[WARN] One or more splits are empty. Check 'month' values in your CSV.")

    return train, val, test


def save_split_pickle(df: pd.DataFrame, variant: str, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    vtok = variant_safe_name(variant)
    out_path = out_dir / f"{vtok}_{name}.pkl"
    payload = {"variant": variant, "data": df}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[OK] Saved {name} split -> {out_path}")


# ---------------------------
# Orchestrator
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Orchestrator: runs data cleaning, preparation and modeling."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["clean", "prepare", "xgboost", "mlp", "lgbm"],  # added lgbm
        help="Pipeline mode. 'clean' -> clean CSV; 'prepare' -> write train/val/test pickles; "
             "'xgboost' -> train XGBoost; 'mlp' -> train MLP; 'lgbm' -> train LightGBM."
    )
    parser.add_argument(
        "--datacsv",
        type=str,
        help="For --mode clean: raw CSV path. For --mode prepare: cleaned CSV path."
    )
    parser.add_argument(
        "--preparedpath",
        type=str,
        help="For --mode xgboost/mlp/lgbm: path to prepared variant folder, e.g. prepared_data/prepared_base"
    )
    parser.add_argument("--resampler", default="none", choices=["none", "ros", "smote"],
                        help="(xgboost/mlp) Resampling strategy applied to TRAIN only.")
    parser.add_argument("--seed", type=int, default=42,
                        help="(xgboost/mlp) Random seed for resampling and model.")
    parser.add_argument("--smote-k", type=int, default=5,
                        help="(xgboost/mlp) k_neighbors for SMOTE.")
    args = parser.parse_args()

    if args.mode in ("clean", "prepare"):
        if not args.datacsv:
            raise ValueError("--datacsv is required for modes 'clean' and 'prepare'")
        in_path = Path(args.datacsv).expanduser().resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {in_path}")

    if args.mode == "clean":
        out_dir = Path("cleandata")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{in_path.stem}_clean.csv"
        clean_data.run_clean(datacsv=str(in_path), output_csv=str(out_path))
        print(f"[OK] Cleaned file written to: {out_path}")

    elif args.mode == "prepare":
        df = pd.read_csv(in_path)
        variant = infer_variant_name(in_path)
        train, val, test = split_by_month(df)

        # Save into ./prepared_data/prepared_<variant-lower-no-spaces>/
        variant_token = re.sub(r"\s+", "", variant).lower()  # e.g., "base", "variantiii"
        variant_dir = Path("prepared_data") / f"prepared_{variant_token}"
        save_split_pickle(train, variant, "train", variant_dir)
        save_split_pickle(val, variant, "val", variant_dir)
        save_split_pickle(test, variant, "test", variant_dir)

    elif args.mode == "xgboost":
        if not args.preparedpath:
            raise ValueError("--preparedpath is required for mode 'xgboost'")
        prepared_dir = Path(args.preparedpath).expanduser().resolve()
        if not prepared_dir.exists():
            raise FileNotFoundError(f"Prepared path not found: {prepared_dir}")
        xgboost_model.run_xgb(
            prepared_dir,
            resampler=args.resampler,
            seed=args.seed,
            smote_k=args.smote_k,
        )

    elif args.mode == "mlp":
        if not args.preparedpath:
            raise ValueError("--preparedpath is required for mode 'mlp'")
        prepared_dir = Path(args.preparedpath).expanduser().resolve()
        if not prepared_dir.exists():
            raise FileNotFoundError(f"Prepared path not found: {prepared_dir}")

        mlp_binary_model.run_mlp(
            prepared_dir,
            # optional hyperparams; keep defaults if you prefer
            dropout=0.0,
            lr=0.001,
            batch_size=64,
            epochs=3,
            weight_decay=0.0,
            # resampling controls
            resampler=args.resampler,
            seed=args.seed,
            smote_k=args.smote_k,
        )

    elif args.mode == "lgbm":
        if not args.preparedpath:
            raise ValueError("--preparedpath is required for mode 'lgbm'")
        prepared_dir = Path(args.preparedpath).expanduser().resolve()
        if not prepared_dir.exists():
            raise FileNotFoundError(f"Prepared path not found: {prepared_dir}")

        lgbm_model.run_lgbm(prepared_dir)

if __name__ == "__main__":
    main()
