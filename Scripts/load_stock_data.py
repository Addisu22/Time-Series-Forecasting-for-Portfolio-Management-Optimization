import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from IPython.display import display


def download_data(tickers, start, end):
    """
    Download adjusted close prices for given tickers and date range.
    """
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)
        if data.isnull().values.any():
            print("Warning: Missing values detected in downloaded data.")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def clean_data(df):
    """
    Clean data by forward-fill and backward-fill missing values.
    """
    df_clean = df.copy()
    df_clean = df_clean.ffill().bfill()
    return df_clean

def basic_stats(df):
    """
    Print basic descriptive statistics.
    """
    print("Basic Statistics:")
    display(df.describe().T)

def plot_prices(df):
    """
    Plot adjusted close prices over time.
    """
    df.plot(figsize=(12,6), title='Adjusted Close Prices')
    plt.ylabel('Price ($)')
    plt.show()

def daily_returns(df):
    """
    Calculate daily percentage change.
    """
    return df.pct_change().dropna()

def plot_returns(returns):
    """
    Plot daily returns.
    """
    returns.plot(figsize=(12,6), title='Daily Returns')
    plt.ylabel('Daily Return')
    plt.show()

def rolling_stats(returns, window=20):
    """
    Calculate and plot rolling mean and std deviation.
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    plt.figure(figsize=(14,7))
    plt.subplot(2,1,1)
    rolling_mean.plot(title=f'{window}-Day Rolling Mean of Returns')
    plt.subplot(2,1,2)
    rolling_std.plot(title=f'{window}-Day Rolling Std Dev of Returns')
    plt.show()
    
    return rolling_mean, rolling_std

def detect_outliers(returns, z_thresh=3):
    """
    Detect outliers based on Z-score.
    """
    from scipy.stats import zscore
    z_scores = returns.apply(zscore)
    outliers = (np.abs(z_scores) > z_thresh)
    return outliers

def adf_test(series, name):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    print(f"\nADF Test for {name}:")
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value:.4f}")
    if result[1] < 0.05:
        print(f"Result: {name} is stationary (reject H0)")
    else:
        print(f"Result: {name} is non-stationary (fail to reject H0)")

def calculate_var(returns, confidence_level=0.05):
    """
    Calculate historical Value at Risk (VaR) at given confidence level.
    """
    var = returns.quantile(confidence_level)
    print(f"\nValue at Risk (VaR) at {int((1-confidence_level)*100)}% confidence level:")
    print(var)
    return var

def sharpe_ratio(returns, risk_free_rate=0.0, trading_days=252):
    """
    Calculate annualized Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate / trading_days
    sr = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)
    return sr