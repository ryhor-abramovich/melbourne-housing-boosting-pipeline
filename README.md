# 📈 Melbourne Housing Price Prediction: Engineering Pipeline Evolution

Comprehensive analysis of the Melbourne (Australia) housing market and development of predictive machine learning pipelines to estimate property values. 

This project addresses a classic **regression** task featuring distinct spatial dependencies and high-density missing data. Its core value lies in demonstrating the **evolution of an engineer's mindset** — shifting from a naive pipeline with data leakages to a robust, production-ready framework.

---

## 🚀 Key Results & Experiment Log

The project was developed in two distinct phases. The metrics below highlight how fixing structural issues and upgrading algorithms dramatically slashed the Mean Absolute Error (MAE) and boosted the $R^2$ score.

### Phase 2: Production-Ready Pipeline (Latest Version)
*Clean data splitting, proper cross-validation, target log-transformation, and advanced algorithms.*

| Model / Algorithm | MAE (AU$) | $R^2$ Score | Key Note |
| :--- | :---: | :---: | :--- |
| **Linear Regression (Ridge)** | 231,653.37 | 0.6800 | L2 regularization prevents overfitting on spatial flags |
| **Random Forest Regressor** | 170,765.46 | 0.7900 | Massive upgrade due to log-transformed target variable |
| **LightGBM Regressor (Final)** | **163,934.27** | **0.8200** | Top performance via native handling of high-cardinality suburbs |

### Phase 1: Legacy Pipeline (Archived Draft)
*Initial experiments featuring architectural flaws (data leakage during imputation) and un-transformed targets.*

| Model / Experiment | MAE (AU$) | $R^2$ Score | Experiment Takeaway |
| :--- | :---: | :---: | :--- |
| Baseline (Linear Regression) | 331,236.55 | 0.4342 | Naive model on raw numeric features |
| Linear Regression + Scaling + OHE | 318,274.77 | 0.4605 | Added `Inverse_Distance` and property type |
| Random Forest (Basic) | 249,702.25 | 0.5862 | Transitioned to tree-based ensembles |
| Random Forest + Spatial Context | 211,064.78 | 0.6758 | Added `Regionname` (dropped error by 38k AU$) |
| Random Forest + Feature Selection | 211,363.64 | 0.6758 | Dropped 4 low-importance columns with zero quality degradation |

---

## 🔄 Post-Code-Review Refactoring & Leakage Fix

The transition from Phase 1 to Phase 2 involved a deep architectural overhaul based on rigorous validation principles:

1. **Data Leakage Elimination:** In the legacy version, missing value imputations (group medians) were calculated across the entire dataset before splitting. In the production version, preprocessing parameters are strictly learned from the training folds and then mapped to validation/test folds, preventing information from bleeding backward.
2. **Target Log-Transformation:** Real estate prices are heavily right-skewed. Applying `np.log1p` to the target variable allowed models (especially Ridge and LightGBM) to learn exponentially better, avoiding massive errors on luxury properties.
3. **Model Simplification:** Proved that dropping redundant low-importance features based on `.feature_importances_` optimizes memory and speed without hurting accuracy.

---

## 🛠️ Feature Engineering Highlights

* **Distance Hyperbola:** EDA revealed a non-linear relationship between price and distance to the city center. An `Inverse_Distance = 1 / (Distance + 0.1)` feature was engineered to linearize this decay.
* **Premium Comfort Index:** Created a composite asset scale feature (`Total_Amenities = Rooms + Bathroom + Car`) to capture the overall capacity of the property.
* **Market Seasonality:** Extracted temporal trends (`Year` and `Month`) to account for cyclical real estate market demand.

## 📁 Repository Structure
* `Melbourne_housing_FULL.csv` — Raw dataset.
* `melbourne_ml_pipeline.ipynb` — Production notebook featuring clean validation, Ridge, RF, and LightGBM models.
* `legacy_v1_notebook.ipynb` — Initial project draft preserved to showcase pipeline optimization history.

## 🛠️ Technical Stack
`Python`, `Pandas`, `NumPy`, `LightGBM`, `Scikit-learn` (Ridge, RandomForestRegressor, StandardScaler), `Matplotlib`, `Seaborn`.
