# Energy Consumption Prediction Using Machine Learning

## Project Overview
This project develops a regression model to predict daily energy consumption using environmental and economic indicators.  
The workflow includes exploratory data analysis, data cleaning, feature engineering, model training, hyperparameter tuning, and performance evaluation.

## Data Preparation
- Converted date fields to usable numerical components (year, month, day, day of week)
- Added lag features to capture short and long-term consumption trends:
  - energy_lag_1, energy_lag_7, energy_lag_30
- Added rolling window features to incorporate moving averages and volatility:
  - energy_roll_mean_7, energy_roll_mean_30
  - energy_roll_std_7, energy_roll_std_30
- Added Fourier seasonal encoding for weekly and yearly seasonality
- Encoded categorical variables using OneHotEncoder
- Imputed missing values and scaled numerical features

## Models Trained
| Model | Description |
|-------|-------------|
| Linear Regression | Baseline |
| Random Forest Regressor | Non-linear ensemble model |
| Gradient Boosting Regressor | Stage-wise additive tree model |
| XGBoost Regressor | Optimized gradient boosting with high performance |
| LightGBM Regressor | Gradient boosting based on leaf-wise splits (best performance) |

## Hyperparameter Tuning
- Performed RandomizedSearchCV with cross-validation
- Tuned Random Forest and Gradient Boosting models
- Evaluated XGBoost and LightGBM with optimized parameter sets

## Model Performance (Summary)
| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Random Forest (Tuned) | ~12.05M | ~2929 | ~0.12 |
| Gradient Boosting (Tuned) | ~12.11M | ~2941 | ~0.12 |
| XGBoost | Improved over RF/GB | Improved | Higher R² |
| **LightGBM (Best)** | **Lowest RMSE** | **Lowest MAE** | **Highest R²** |

**LightGBM selected as final model.**

## Key Findings
- Raw models performed poorly due to limited temporal information.
- Feature engineering (lag + rolling + seasonality) significantly improved predictive performance.
- LightGBM produced the most accurate and stable predictions.

## Final Output
- Trained LightGBM model saved as `.joblib` for deployment.
- The workflow can be extended for forecasting, dashboards, and automated reporting.

## Technologies
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- Matplotlib / Seaborn
