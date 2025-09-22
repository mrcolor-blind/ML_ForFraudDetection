# Machine Learning For Fraud Detection (Using BAF dataset)
- Arah Rojas Blanco (A00834299)
- Valeria Aguilar Meza (A01741304)
- Miguel Angel Barrientos Ballesteros (A01637150)
- Mariela Quintanar de la Mora (A01642675)

Antes de la estructura general del README, incluimos en español la sección de Evaluables a continuación para facilitar la búsqueda de los profesores al evaluar el proyecto, así como el Resumen de Pasos para Reproducir la Solución Final.

## Evaluables

### Reporte y Análisis
- `/TeamDocs/`
  - **ReporteFinal_ML_FraudDetection.pdf**: documento con la descripción del reto, metodología, resultados y conclusiones.
  - **RetoIA_Fraude_Presentacion.pdf**: presentación del reto en equipo.

### Implementación
- `/Code/`
  - **clean_data.py**: script para limpieza y transformación de datos usando técnicas ETL.
  - **/prepared_data/**: carpetas por variante que contienen los splits limpios (la **Variant III** fue la seleccionada para la solución final del modelo).
  - **utils/balancing.py**: funciones de balanceo de datos (oversampling con RandomOverSampler y SMOTE).
  - **models/lgbm_model.py**: modelo final seleccionado para la solución del reto (LightGBM).
  - **results/lgbm[...]_alt.png**: gráficas generadas del desempeño del modelo final.
  - **results/lgbm_results.txt**: archivo con métricas de desempeño del modelo final.

## Resumen de Pasos para Reproducir la Solución Final

1. Descargar el dataset desde Kaggle.  
2. Colocar el archivo **"Variant III.csv"** dentro de la carpeta `/OriginalDataset/`.  
3. Abrir la terminal y situarse en la carpeta `/Code/` (ejecutar: `cd Code`).  
4. Limpiar el dataset:  
```
python run.py --mode clean --datacsv OriginalDataset/"Variant III.csv"
```
5. Preparar los conjuntos (train, val, test):  
```
python run.py --mode prepare --datacsv cleandata/"Variant III_clean.csv"
```
6. Entrenar el modelo LightGBM con **SMOTE**:  
```
python run.py --mode lgbm --preparedpath prepared_data/prepared_variantiii --resampler smote --smote-k 5 --seed 42
```

### Outputs esperados
- **results/lgbm_results.txt**: métricas principales del modelo (AUC, Recall@5%FPR, Accuracy, etc.).  
- **results/lgbm[...]_alt.png**: gráficas de curvas de aprendizaje y validación.  

**Nota**: El análisis e interpretación detallada de estos resultados se encuentra en el reporte final (`/TeamDocs/ReporteFinal_ML_FraudDetection.pdf`).  

## Repository

This repository implements an end-to-end machine learning pipeline for fraud detection using the **Bank Account Fraud (BAF)** dataset. The workflow covers data cleaning, preparation of temporal splits, and training of classification models with optional data balancing strategies.

### About the BAF Dataset
The **Bank Account Fraud (BAF)** dataset is a large-scale, privacy-preserving, synthetic dataset created from real-world bank account opening fraud scenarios. It was introduced at **NeurIPS 2022** as one of the first benchmark resources for evaluating fraud detection and fairness in machine learning.

- **Scale**: 1,000,000 applications, 30 features, spanning 8 months of data.  
- **Target**: `fraud_bool` column indicates whether an application is fraudulent.  
- **Challenge**: Highly imbalanced (fraud ≈ 1%), with temporal dynamics.  
- **Variants**: Includes one Base dataset and five biased variants (I–V), each designed to test models under different fairness and distribution-shift conditions.  

The dataset can be downloaded from Kaggle:  
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

### Pipeline
The pipeline follows three main stages:

1. **Cleaning** raw CSVs
2. **Preparing** temporal splits for training, validation, and testing
3. **Training** machine learning models with optional balancing
---

## Preparing Before Running the Code

Before using the pipeline, make sure the environment is properly set up:

- Navigate to the `/Code/` folder in your terminal. All commands provided in this repository must be executed from inside this directory. Running commands from another location may result in errors when accessing paths.

- Download the **Bank Account Fraud (BAF) dataset** from Kaggle:  
  https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

- Place the required `.csv` files into the `/OriginalDataset/` folder. Each dataset variant you want to process (e.g., "Base.csv", "Variant III.csv", etc.) must be located in this directory before starting the cleaning stage.


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

This stage trains an MLP classifier using the prepared splits. It supports optional balancing of the train set with **none**, **random oversampling (ros)**, or **SMOTE (smote)**. This specific model allows for W&B linking. After running, you may provide your API key or select dont visualize my results if you just want to run locally (Local outputs will still work).


- Run training:
```
python run.py --mode mlp --preparedpath prepared_data/prepared_base
```

- With balancing options:
``` 
python run.py --mode mlp --preparedpath prepared_data/prepared_base --resampler ros --seed 42
```
```
python run.py --mode mlp --preparedpath prepared_data/prepared_base --resampler smote --smote-k 5
```

- Output:  
  - results/mlp_test_results.txt (metrics file)  
  - results/mlp[...].png (learning and validation curves)


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

## 3. Modeling Stage (LightGBM)

This stage trains a LightGBM classifier using the prepared splits. It also supports optional balancing of the train set with **none**, **ros**, or **smote**.

- Run training:  
```
python run.py --mode lgbm --preparedpath prepared_data/prepared_base
```
- With balancing options:  
```
python run.py --mode lgbm --preparedpath prepared_data/prepared_base --resampler ros --seed 42  
```
```
python run.py --mode lgbm --preparedpath prepared_data/prepared_base --resampler smote --smote-k 5
```
- Output:  
  - models/lgbm_Base.joblib (saved model)  
  - results/lgbm_results.txt (metrics file)  
  - results/lgbm[...]_alt.png (learning and validation curves)

---

## 3. Modeling Stage (Random Forest)

This stage trains a Random Forest classifier using the prepared splits. It supports the same balancing options as the other models.

- Run training:  
```
python run.py --mode randomforest --preparedpath prepared_data/prepared_base
```

- With balancing options:  
```
python run.py --mode randomforest --preparedpath prepared_data/prepared_base --resampler ros --seed 42
```
```
python run.py --mode randomforest --preparedpath prepared_data/prepared_base --resampler smote --smote-k 5
```
- Output:  
  - models/rf_Base.joblib (saved model)  
  - Console evaluation metrics (AUC-ROC, Precision, Recall, F1, Confusion Matrix)


## Arguments

- **--mode**  
  Selects pipeline stage. Options:  
  - clean  
  - prepare  
  - mlp  
  - xgboost  
  - lgbm  
  - randomforest  

- **--datacsv**  
  Path to input CSV. Required for `clean` and `prepare`.

- **--preparedpath**  
  Path to folder with prepared splits (e.g., `prepared_data/prepared_base`). Required for all modeling stages (`mlp`, `xgboost`, `lgbm`, `randomforest`).

- **--resampler** *(modeling stages only)*  
  Strategy for balancing the train split. Options:  
  - none (default)  
  - ros (RandomOverSampler)  
  - smote (SMOTE)  

- **--seed** *(modeling stages only)*  
  Random seed for reproducibility. Default: 42.

- **--smote-k** *(when using `--resampler smote`)*  
  Number of neighbors for SMOTE. Default: 5.


## Workflow Summary

- Start with raw data (OriginalData/*.csv)  
- Clean it → cleandata/*.csv  
- Prepare splits → prepared_data/prepared_*  
- Train model → models/  

This modular design allows adding future models or balancing strategies without changing the previous pipeline stages.

