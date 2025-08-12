# Portfolio Management & Forecasting Pipeline

## 📌 Overview
This project implements a **time series forecasting and portfolio optimization pipeline** for a basket of assets (e.g., TSLA, SPY, BND).  
It includes **data preparation, forecasting with ARIMA/SARIMA/LSTM, portfolio optimization, and backtesting** to evaluate strategy performance against a benchmark.

The goal is to:
1. Forecast asset returns using both traditional statistical models and deep learning.
2. Optimize portfolio weights based on predicted returns and risk.
3. Backtest the strategy to assess its viability.

---

## 📂 Project Structure

```
Week11_Portfolio_Management_AI_Timeseries
├─ data/
│  ├─ adj_close_prices.csv
│  ├─ returns.csv
├─ notebooks/
│  ├─ 01_data_fetch_and_eda.ipynb     
│  ├─ 02_arima_model.ipynb
│  ├─ 03_lstm_model.ipynb
│  ├─ 04_portfolio_optimization.ipynb
│  └─ 05_backtest.ipynb
├─ src/
│  ├─ fetch_data.py
│  ├─ preprocessing.py
│  ├─ models_arima.py
│  ├─ models_lstm.py
│  ├─ portfolio_optimization.py
│  ├─ forecast_arima.py
│  ├─ forecast_sarima.py
│  ├─ forecast_lstm.py
│  └─ backtesting.py
├── .gitignore
├── requirements.txt
└─ README.md
```


---

## 📜 Tasks Summary

### **Task 1: Git & Environment Setup**
- Initialized a **Git repository** and created a `.gitignore` file for Python projects.
- Installed dependencies listed in `requirements.txt`.
- Created a clean project structure with folders for data, source code, and notebooks.
- Tested the environment by running a sample Python script.

---

### **Task 2: Data Preparation**
- Loaded **adjusted closing prices** from `adj_close_prices.csv`.
- Processed data for missing values, aligned dates, and computed daily returns.
- Created utility functions for:
  - Loading CSV data.
  - Calculating log returns.
  - Splitting train/test datasets.

---

### **Task 3: Forecasting Models**
Implemented and compared three models:
1. **ARIMA** – AutoRegressive Integrated Moving Average.
2. **SARIMA** – Seasonal ARIMA.
3. **LSTM** – Long Short-Term Memory neural network.

For each model:
- Trained on historical data.
- Forecasted returns for a selected horizon.
- Calculated evaluation metrics: RMSE, MAE.

---

### **Task 4: Portfolio Optimization**
- Used forecasted returns to **compute optimal portfolio weights**.
- Implemented **Mean-Variance Optimization** (Markowitz) with:
  - Expected returns from forecasts.
  - Covariance matrix of historical returns.
- Generated:
  - Optimal weights.
  - Expected portfolio return.
  - Expected portfolio volatility.
  - Sharpe ratio.

---

### **Task 5: Backtesting**
- **Backtesting period:** Last year of the dataset (e.g., Aug 1 2024 – Jul 31 2025).
- **Benchmark:** 60% SPY / 40% BND static portfolio.
- Simulated:
  - Strategy portfolio using Task 4’s optimal weights.
  - Monthly holding period without rebalancing (simplified).
- **Analysis:**
  - Plotted cumulative returns of strategy vs benchmark.
  - Calculated Sharpe ratio & total return for both.
  - Compared results to assess model viability.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mewael-EME/Week11_Portfolio_Management_AI_Timeseries
cd portfolio-forecasting


## 🛠 Requirements

Install Python 3.9+ and the dependencies listed below:

```bash
pip install -r requirements.txt

requirements.txt should include:
pandas
numpy
matplotlib
seaborn
yfinance
scipy
statsmodels
```
🛠️ Dependencies

```
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- statsmodels, pmdarima
- scikit-learn
- tensorflow / keras
- scipy ```


🚀 How to Run the Pipeline

1. Clone the repository
```
git clone https://github.com/Mewael-EME/Week11_Portfolio_Management_AI_Timeseries
cd stock-data-analysis
```

2. Run the data fetch script
```python src/fetch_data.py```

3. Run the preprocessing script
```python src/preprocessing.py```

4. Check the generated file
```The processed dataset will be saved as: data/adj_close_prices.csv```

5. Open the notebook for EDA
```jupyter notebook notebooks/01_data_fetch_and_eda.ipynb```

Replace Task1_Data_Preparation.ipynb with any other task notebook.

Run Full Pipeline
```
python src/data_preparation.py
python src/forecast_arima.py
python src/forecast_sarima.py
python src/forecast_lstm.py
python src/portfolio_optimization.py
python src/backtesting.py

```

📊 Features

- Data Fetching — Downloads stock data for TSLA, BND, and SPY using yfinance.
- Preprocessing — Cleans data, handles missing values, and prepares adjusted close prices.
- Correlation & Covariance Analysis — Computes and visualizes correlations.
- Missing Data Visualization — Displays null values heatmap for documentation.
- Volatility Clustering — Shows rolling standard deviation trends.
- Cumulative Returns — Simulates growth from $1 investment.
- Stationarity Tests — Runs Augmented Dickey-Fuller tests on both prices and returns.

📈 Example Outputs

- Correlation heatmap between TSLA, BND, and SPY returns
- Rolling correlation plot (90-day window)
- Volatility clustering plot
- Cumulative returns plot
- ADF test results for returns

