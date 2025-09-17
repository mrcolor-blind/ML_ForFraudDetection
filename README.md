# BAF Pipeline README

This repository implements a complete pipeline for preparing and modeling the **Bank Account Fraud (BAF)** dataset suite. The pipeline follows three main stages:

1. **Cleaning** raw CSVs
2. **Preparing** temporal splits for training, validation, and testing
3. **Training** machine learning models (currently XGBoost) with optional balancing

---

## 1. Cleaning Stage

The cleaning process loads a raw dataset (e.g., `Base.csv`), performs value normalization, outlier handling, transformations, imputations, and categorical encoding, then exports a cleaned version.

- Run cleaning:
```
python run.py --mode clean --datacsv OriginalData/"Base.csv"
```

- Output: 
  - cleandata/Base_clean.csv

---

## 2. Preparing Stage

The preparation step splits the cleaned dataset into train, validation, and test sets according to the `month` column:

- Train: months 1–5  
- Validation: month 6  
- Test: months 7–8  

Each split is saved as a `.pkl` file, containing both the variant name and the dataframe.

- Run preparation:
```
python run.py --mode prepare --datacsv cleandata/"Base_clean.csv"
```

- Output directory:
  - prepared_data/prepared_base/
    - Base_train.pkl
    - Base_val.pkl
    - Base_test.pkl

Other dataset variants (Variant I–V) are automatically handled and saved in corresponding prepared folders.

---

## 3. Modeling Stage (MLP)

This stage trains an MLP classifier using the prepared splits. It supports optional balancing of the train set with **none**, **random oversampling (ros)**, or **SMOTE (smote)**.

- Run training:
```
python run.py --mode xgboost --preparedpath prepared_data/prepared_base
```

## 3. Modeling Stage (XGBoost)

This stage trains an XGBoost classifier using the prepared splits. It supports optional balancing of the train set with **none**, **random oversampling (ros)**, or **SMOTE (smote)**.

- Run training:
```
python run.py --mode xgboost --preparedpath prepared_data/prepared_base
```

- With balancing options:
``` 
python run.py --mode xgboost --preparedpath prepared_data/prepared_base --resampler ros --seed 42
```
```
python run.py --mode xgboost --preparedpath prepared_data/prepared_base --resampler smote --smote-k 5
```

- Output:
  - models/xgb_Base.joblib (saved model)
  - Console evaluation metrics (AUC-ROC, AUC-PR, Precision, Recall, F1, Confusion Matrix)

---

## Arguments

- **--mode**  
  Selects pipeline stage. Options:  
  - clean  
  - prepare  
  - xgboost  

- **--datacsv**  
  Path to input CSV. Required for `clean` and `prepare`.

- **--preparedpath**  
  Path to folder with prepared splits (e.g., `prepared_data/prepared_base`). Required for `xgboost`.

- **--resampler** *(xgboost only)*  
  Strategy for balancing the train split. Options:  
  - none (default)  
  - ros (RandomOverSampler)  
  - smote (SMOTE)  

- **--seed** *(xgboost only)*  
  Random seed for reproducibility. Default: 42.

- **--smote-k** *(xgboost only)*  
  Number of neighbors for SMOTE. Default: 5.

---

## Workflow Summary

- Start with raw data (OriginalData/*.csv)  
- Clean it → cleandata/*.csv  
- Prepare splits → prepared_data/prepared_*  
- Train model → models/*.joblib  

This modular design allows adding future models or balancing strategies without changing the previous pipeline stages.
