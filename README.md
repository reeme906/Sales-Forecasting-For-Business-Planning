# 📈 Sales Forecasting for Business Planning

## Overview

This project focuses on forecasting Walmart weekly sales using two time-series forecasting techniques:

- ARIMA (Statistical Model)
- LSTM (Deep Learning Model)

The objective is to compare both approaches and evaluate their ability to support business planning and decision-making through accurate sales forecasting.

# Business Problem

Retail companies rely on accurate sales forecasts to:

- Improve inventory management
- Reduce overstock and stockouts
- Support staffing decisions
- Optimize supply chain operations
- Improve financial planning

This project investigates which forecasting model provides better prediction performance.

# Dataset

- Walmart Weekly Sales Dataset
- Original dataset contains over 420,000 records.
- A subset of 1,000 aggregated weekly observations was used for this project.

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- TensorFlow / Keras
- Scikit-learn

# Methodology

1. Data Cleaning
2. Date Aggregation
3. Exploratory Data Analysis
4. ARIMA Model
5. LSTM Model
6. Model Evaluation

Performance metrics:

- MAE
- MSE
- RMSE

# Results

| Model | MAE | RMSE |
|-------|------:|------:|
| ARIMA | 17,531 | 20,179 |
| LSTM | 12,108 | 14,131 |

LSTM achieved better forecasting performance by capturing nonlinear sales patterns more effectively than ARIMA.

# Future Improvements

- Train using the complete Walmart dataset.
- Include external variables such as holidays and promotions.
- Hyperparameter tuning for LSTM.
- Deploy forecasting model as a web application.

