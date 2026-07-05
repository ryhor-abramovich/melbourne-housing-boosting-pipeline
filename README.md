# melbourne-property-rf-analysis
End-to-end regression analysis of the Melbourne housing market using Scikit-Learn. Features iterative optimization from Linear Regression to Random Forest ($R^2$ = 0.676).
# Melbourne Housing Market: End-to-End Price Regression

Comprehensive analysis of the Melbourne (Australia) housing market and development of predictive machine learning models to estimate property values based on their physical and geographical characteristics.

## 📌 Project Overview
The goal of this project is to investigate the core factors driving real estate pricing in Melbourne, perform deep exploratory data analysis (EDA), handle domain-specific missing values, and build a robust machine learning pipeline for price prediction.

This project addresses a classic **regression** task featuring distinct spatial (geographical) dependencies and high-density missing data.

## 🛠 Technical Stack
* **Language:** Python
* **Data Analysis & Preprocessing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (LinearRegression, RandomForestRegressor, StandardScaler)

## 🧮 Feature Engineering & Data Cleansing
* **Distance Hyperbola:** EDA revealed a non-linear (hyperbolic) relationship between property price and distance to the city center. An `Inverse_Distance = 1 / (Distance + 0.1)` feature was engineered, significantly improving linear model performance.
* **Group-Based Imputation:** Missing values in `BuildingArea` and `Landsize` were imputed using a granular approach — extracting the median values grouped by specific `Suburb`, property `Type`, and number of `Rooms` via the `.transform('median')` method, rather than using a global dataset average.
* **Anomaly Filtering:** Integrated strict outlier removal constraints for building sizes (dropping records where `BuildingArea` < 10 or > 600 sq.m.) to eliminate data noise before training.

## 📈 Model Performance Evolution (Iterative Approach)

The feature architecture was developed iteratively, leading to a consistent improvement in model performance metrics (evaluated using Mean Absolute Error — MAE, and Coefficient of Determination — $R^2$):

| Iteration | Experiment Description | MAE (AU$) | $R^2$ |
| :--- | :--- | :--- | :--- |
| **Baseline** | Linear Regression on raw features (`Rooms`, `Distance`, `BuildingArea`) | 331,236 | 0.434 |
| **Feature Eng.** | Linear Regression + `Inverse_Distance` + One-Hot Encoding for property `Type` | 318,274 | 0.460 |
| **Algorithm Shift** | Transitioned to ensemble methods. Random Forest on the same feature set | 249,702 | 0.586 |
| **Spatial Context** | Random Forest + added geographical regions (`Regionname` via One-Hot Encoding) | 211,064 | **0.676** |
| **Optimization** | Random Forest + Dropped redundant low-importance features (`EV`, `NV`, `WV`, `SEM`) based on `.feature_importances_` | **211,363** | **0.676** |

**Key Takeaway:** Incorporating spatial context (`Regionname`) dropped the model's average error by nearly **38,000 AU$**. Subsequent feature selection based on importance metrics successfully simplified the model (removing 4 columns) with zero degradation in predictive power.

## 🚀 How to Run
1. Clone the repository.
2. Open the notebook in your Anaconda / Jupyter Notebook environment.
3. Check the execution of the final `predict_house_price` function designed for standalone, real-time value estimation.
