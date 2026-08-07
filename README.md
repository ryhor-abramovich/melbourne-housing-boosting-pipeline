# 🏠 Melbourne Housing Price Prediction: Engineering Pipeline

A two-stage machine learning project for predicting Melbourne residential property prices.

The project focuses on data cleaning, feature engineering, handling missing values, categorical variables, and comparing several regression models.

---

## 📌 Project Overview

This project uses the Melbourne housing dataset containing **34,857 records and 21 columns**.

The work was developed in two stages:

1. **Exploratory / baseline stage** — initial analysis, feature engineering, and experiments with different regression approaches.
2. **Refined modeling stage** — stricter preprocessing, explicit train/test separation, leakage-safe imputation, and comparative evaluation of Ridge Regression, Random Forest, and LightGBM.

The target variable is **`Price`**.

---

## 🔍 Stage 1 — Initial Modeling

The first stage was used to explore the dataset and establish baseline performance.

The experiments included:

- Linear Regression
- Random Forest
- feature scaling
- categorical encoding
- feature engineering
- different combinations of predictors

The initial experiments showed that tree-based models substantially outperformed the linear baseline.

This stage was useful for identifying promising preprocessing and modeling directions before building the more structured final pipeline.

---

## 🧹 Stage 2 — Data Cleaning & Preprocessing

The second stage introduced a more disciplined preprocessing workflow.

### Dataset

The original dataset contains:

- **34,857 rows**
- **21 columns**
- numerical and categorical features
- substantial missing data in several variables

After removing rows with missing target values and filtering explicit physical/data errors, the dataset contained:

- **27,111 records**

The data was then split into:

- **21,688 training records**
- **5,423 test records**

The split was performed before model-specific preprocessing to avoid data leakage.

---

## 🚨 Anomaly Filtering

The following rules were applied before the train/test split:

| Feature | Filtering rule |
|---|---|
| `Price` | Rows with missing target values removed |
| `Landsize` | Values above 5,000 filtered; missing values preserved |
| `BuildingArea` | Values above 600 filtered; missing values preserved |
| `YearBuilt` | Values outside 1800–2026 filtered; missing values preserved |

The goal was not to remove statistically unusual houses simply because they were unusual, but to eliminate values that were considered implausible or indicative of data errors.

---

## 🛠️ Feature Engineering

### Date features

The original `Date` column was converted to datetime and decomposed into:

- `SaleYear`
- `SaleMonth`

The original `Date` column was then removed.

### Building age

A derived feature was created:

```text
BuildingAge = SaleYear - YearBuilt
