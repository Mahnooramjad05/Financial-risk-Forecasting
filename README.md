💰 Financial Risk Forecasting Dashboard

This project is a Streamlit-based interactive dashboard for financial risk analysis and forecasting.
It enables users to upload financial datasets, preprocess data, visualize trends, and apply machine learning & time-series forecasting models to predict financial risks.

🚀 Features
📂 Data Upload & Exploration
Upload CSV datasets.
View data preview and summary statistics.
Interactive feature & target selection.

🧹 Preprocessing
Scaling methods: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer.
Missing value handling and encoding support.

📊 Visualization
Multiple chart types: Scatter, Line, Box, Histogram, Heatmap.
Correlation heatmap for numeric features.

🔮 Forecasting Models
Machine Learning & Time-Series Models:
XGBoost
LSTM
GRU
ARIMA
Prophet

Adjustable lookback window and forecast horizon.
Performance metrics: Mean Absolute Error (MAE) and R² Score.

📈 Financial Risk Analysis
Predict and visualize financial risk indicators.
Compare actual vs. predicted performance.

📦 Installation
Clone the repository:
git clone https://github.com/your-username/financial-risk-forecasting.git
cd financial-risk-forecasting

Install dependencies:
pip install -r requirements.txt


Run the app:
streamlit run FinancialRiskForecasting_StreamlitApp_fixed.py

⚙️ Requirements
Python 3.8+
Streamlit
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
XGBoost
Statsmodels (ARIMA)
Prophet
TensorFlow / Keras

Install all at once:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels prophet tensorflow

📂 Project Structure
📁 financial-risk-forecasting
 ┣ 📜 FinancialRiskForecasting_StreamlitApp_fixed.py   # Main app
 ┣ 📜 requirements.txt                                # Dependencies
 ┗ 📜 README.md                                       # Documentation

🎯 Usage Workflow
Upload a financial dataset (CSV).
Select target column (e.g., returns, risk index).
Choose features & apply preprocessing (scaling, encoding).
Visualize distributions, correlations, and risk patterns.
Train forecasting models (XGBoost, LSTM, GRU, ARIMA, Prophet).
Evaluate results with MAE and R².
Forecast future risk values.

🛠️ Future Enhancements
Portfolio risk & volatility analysis.
Integration with live financial APIs (Yahoo Finance, Alpha Vantage).
Advanced deep learning models for financial time-series.
Automated risk reporting.

👨‍💻 Author

Developed by Mahnoor Amjad ✨
