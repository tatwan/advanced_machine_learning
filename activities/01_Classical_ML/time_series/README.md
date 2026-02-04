# Time Series Forecasting

## 📚 Overview

This module covers **time series forecasting techniques** including Prophet, Theta method, and automated approaches with StatsForecast. Learn to build robust forecasting models that handle trends, seasonality, and special events.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Apply **Facebook Prophet** for trend and seasonality analysis
- ✅ Implement **Theta method** for exponential smoothing
- ✅ Use **StatsForecast** for automated forecasting (AutoARIMA, AutoTheta, AutoETS)
- ✅ Handle **holidays and special events** in forecasts
- ✅ Perform **multi-step ahead forecasting**
- ✅ Evaluate forecast accuracy with appropriate metrics

---

## 📂 Module Structure

```
time_series/
├── README.md (this file)
├── Forecasting.ipynb (Prophet, Theta, StatsForecast with AutoARIMA/AutoTheta/AutoETS)
└── datasets/
    ├── AEP_hourly.csv (Energy consumption)
    ├── air_passenger.csv (Classic time series)
    ├── closing_price.csv (Stock prices)
    ├── cpi.csv (Consumer Price Index)
    ├── daily_weather.csv
    ├── energy_consumption.csv
    ├── milk_production.csv
    └── weekly_sales.csv
```

---

## 🔄 Recommended Learning Path

### **Complete Forecasting Lab** (4-6 hours)

Work through `Forecasting.ipynb` which covers:

1. **Data Preparation**: Loading, visualization, stationarity checks
2. **Prophet**: Trend decomposition, seasonality, holiday effects
3. **Theta Method**: Exponential smoothing approach
4. **StatsForecast**: AutoARIMA, AutoTheta, AutoETS for automated model selection
5. **Evaluation**: Cross-validation, accuracy metrics (MAE, MAPE, RMSE)

---

## 🔍 Methods Covered

| Method | Description | Best For |
|--------|-------------|----------|
| **Prophet** | Additive model with trend, seasonality, holidays | Business forecasting, daily/weekly data |
| **Theta** | Exponential smoothing decomposition | Short-term forecasting |
| **AutoARIMA** | Automated ARIMA model selection | General time series |
| **AutoTheta** | Automated Theta method | Competitions, robust forecasts |
| **AutoETS** | Automated exponential smoothing | Trend and seasonal data |

---

## 📊 Datasets

Multiple real-world datasets for practice:
- **Energy**: Hourly/daily consumption patterns
- **Finance**: Stock prices, CPI trends
- **Retail**: Weekly sales data
- **Classic**: Air passengers, milk production

---

## 🛠️ Technical Requirements

```python
pandas, matplotlib, prophet, statsforecast, statsmodels
```

---

## 🔗 Related Modules

- **Prerequisites**: Basic statistics, data manipulation
- **Related**: [Anomaly Detection (Time Series)](../anomaly_detection_stats/)

---

*Module Difficulty: Intermediate*  
