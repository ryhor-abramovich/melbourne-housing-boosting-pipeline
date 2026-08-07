# 🏠 Melbourne Housing Price Prediction: Engineering Pipeline Evolution

## 📌 Project Overview

This project addresses a classic regression task: predicting Melbourne residential property prices from historical housing data.

The project evolved through two stages:

- **Stage 1:** exploratory analysis, feature engineering, baseline regression models and iterative experimentation;
- **Stage 2:** a more structured, leakage-aware preprocessing and modeling pipeline.

The target variable is **`Price`**.

The original dataset contains **34,857 records and 21 columns**, with substantial missing data in several features.

---

## 🔄 Pipeline Evolution

### Stage 1 — Exploratory Modeling

The first stage focused on understanding the dataset and establishing baseline models.

The workflow included:

- data inspection and exploratory analysis;
- missing-value analysis;
- anomaly and outlier investigation;
- feature engineering;
- categorical encoding;
- Linear Regression;
- Random Forest;
- feature importance analysis;
- iterative refinement of the feature set.

The experiments showed that nonlinear tree-based models were substantially better suited to the data than a simple linear baseline.

### Stage 2 — Leakage-Aware Modeling Pipeline

The second stage focused on improving the reliability and reproducibility of the modeling workflow.

The main improvements were:

- anomaly filtering **before the train/test split**;
- explicit train/test separation;
- feature engineering performed separately on train and test data;
- train-only statistics for imputation;
- subgroup-based imputation using `Suburb + Type`;
- fallback to the global training-set median;
- `ColumnTransformer` and `Pipeline` for model-specific preprocessing;
- comparison of Ridge Regression, Random Forest and LightGBM;
- native categorical feature handling in LightGBM.

This produced a substantially more structured modeling pipeline than the initial exploratory version.

---

## 🧹 Data Cleaning

The original dataset contains 34,857 records.

Several data-quality issues were identified during EDA, including extreme values in `Landsize`, `BuildingArea`, and `YearBuilt`, as well as missing target values.

### Cleaning Rules

| Feature | Rule |
|---|---|
| **Price** | Remove records with missing target values |
| **Landsize** | Remove values above 5,000 |
| **BuildingArea** | Remove values above 600 |
| **YearBuilt** | Keep values between 1800 and 2026 |
| **Missing values** | Preserve where possible and handle during preprocessing |

After filtering, the dataset contained **27,111 records**.

---

## 📊 Train / Test Split

After cleaning:

- **Training set:** 21,688 records
- **Test set:** 5,423 records

The split was performed before model-specific imputation and preprocessing.

This prevents information from the test set from being used when calculating preprocessing statistics.

---

## 🛠️ Feature Engineering

Several features were engineered from the original data.

### Date Features

The original `Date` column was converted to datetime and decomposed into temporal features:

- `SaleYear`
- `SaleMonth`

The original `Date` column was then removed.

### Building Age

A new feature was created:

```text
BuildingAge = SaleYear - YearBuilt
```

### Landsize Imputation

Missing `Landsize` values were imputed using the median for the corresponding:

```text
Suburb + Type
```

combination.

The subgroup medians were calculated **only on the training set**.

For unseen `Suburb + Type` combinations, the global training-set median was used as a fallback.

### Other Numerical Features

Missing values in selected numerical features were filled using medians calculated from the training set.

### Categorical Features

Missing categorical values such as `CouncilArea` were handled using training-set statistics.

---

## 🧩 Feature Set

The final modeling data contained:

### Numerical Features — 13

```text
Rooms
Distance
Postcode
Bedroom2
Bathroom
Car
Landsize
BuildingArea
YearBuilt
Lattitude
Longtitude
Propertycount
BuildingAge
```

### Categorical Features — 6

```text
Suburb
Type
Method
SellerG
CouncilArea
Regionname
```

The `Address` column was removed as a non-predictive text identifier.

---

## 🤖 Models

Three regression models were compared.

### 1. Ridge Regression

The Ridge model used a preprocessing pipeline consisting of:

```text
Numerical:
median imputation → standard scaling

Categorical:
most-frequent imputation → one-hot encoding

                    ↓
                  Ridge
```

Configuration:

```python
Ridge(alpha=1.0)
```

### 2. Random Forest Regressor

Random Forest was trained using numerical median imputation and one-hot encoding for categorical variables.

Configuration:

```python
RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
```

### 3. LightGBM Regressor

LightGBM was used as the final gradient-boosting model.

Configuration:

```python
lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)
```

Categorical variables were converted to the `category` dtype and passed to LightGBM using its native categorical-feature handling.

---

## 📈 Model Performance

The models were evaluated on the held-out test set using **Mean Absolute Error (MAE)** and **R²**.

| Model | MAE | R² |
|---|---:|---:|
| Ridge Regression | 231,653.37 | 0.68 |
| Random Forest Regressor | 170,765.46 | 0.79 |
| **LightGBM Regressor** | **163,934.27** | **0.82** |

The final LightGBM model achieved the best result among the three tested models:

- **MAE: 163,934.27**
- **R²: 0.82**

---

## 🔬 What Changed Between the Two Stages?

The main difference between the two stages was not simply the choice of a more powerful model.

The second stage introduced a more disciplined ML workflow:

| Stage 1 | Stage 2 |
|---|---|
| Exploratory preprocessing | Structured preprocessing pipeline |
| Manual transformations | `Pipeline` / `ColumnTransformer` |
| Initial imputation experiments | Train-only imputation |
| Feature experimentation | Explicit feature groups |
| Baseline models | Ridge + Random Forest + LightGBM |
| General experimentation | Leakage-aware train/test workflow |
| One-hot encoding experiments | Model-specific preprocessing |

This evolution was an important part of the project: the goal was to improve not only predictive performance, but also the **reliability and reproducibility of the modeling process**.

---

## 💡 Key Takeaways

1. **Data quality matters.** Extreme and implausible values can have a substantial effect on regression models.
2. **Feature engineering matters.** Temporal features and `BuildingAge` provide additional information that is not directly available in the raw columns.
3. **Local imputation can be useful.** Using `Suburb + Type` groups for `Landsize` preserves local property characteristics better than immediately applying a global median.
4. **Preprocessing must respect the train/test boundary.** Imputation statistics for the test set were derived from the training data.
5. **Tree-based models outperform the linear baseline.** Both Random Forest and LightGBM substantially improved on Ridge Regression.
6. **Gradient boosting performed best.** LightGBM achieved an R² of **0.82** and the lowest MAE among the tested models.

---

## 🛠️ Technical Stack

- **Python 3**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **LightGBM**
- **Jupyter Notebook**

---

## 📁 Repository

The repository contains:

- `README.md`
- the notebook for **Stage 1**
- the notebook for **Stage 2**

The two notebooks document the progression from the initial exploratory modeling workflow to the refined preprocessing and modeling pipeline.

---

## 🎯 Project Summary

The project demonstrates a practical machine-learning workflow:

```text
Raw Data
   ↓
Data Quality Analysis
   ↓
Anomaly Filtering
   ↓
Train / Test Split
   ↓
Feature Engineering
   ↓
Leakage-Aware Imputation
   ↓
Model-Specific Preprocessing
   ↓
Ridge / Random Forest / LightGBM
   ↓
Model Comparison
```

The main focus is the **evolution of the modeling pipeline**: moving from exploratory experimentation toward a more systematic, reproducible and leakage-aware machine-learning workflow.
