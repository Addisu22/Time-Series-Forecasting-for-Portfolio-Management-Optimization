import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller

def download_stock_data(tickers, start, end):
    """
    Downloads historical closing prices for a list of tickers
    using yfinance between the specified start and end dates.

    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols (e.g., ['TSLA', 'AAPL', 'SPY'])
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format

    Returns:
    --------
    pd.DataFrame
        DataFrame containing closing prices for each ticker
    """
    try:
        import yfinance as yf
    except ImportError:
        # pip install yfinance
        import yfinance as yf

    import pandas as pd

    df = pd.DataFrame()

    for tick in tickers:
        ydata = yf.download(tick, start=start, end=end)
        if 'Close' in ydata.columns:
            df[tick] = ydata['Close']
        else:
            print(f"Warning: 'Close' column not found for {tick}")

    df.index = pd.to_datetime(df.index)

    return df

def inspect_and_clean_df(df, method="linear"):
    """
    Inspect data types, missing values, and clean missing data.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to inspect and clean.
    method : str, optional (default="linear")
        The fill method to use for missing values:
        - "ffill": Forward fill
        - "linear": Linear interpolation

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with missing values handled.
    """
    import pandas as pd

    # Show column data types
    print("Data types of each column:")
    print(df.dtypes)

    # Show missing values
    print("\n Missing values per column:")
    print(df.isna().sum())

    # Handle missing values
    if method == "ffill":
        cleaned_df = df.fillna(method='ffill')
    elif method == "linear":
        cleaned_df = df.interpolate(method='linear')
    else:
        raise ValueError("Invalid method. Use 'ffill' or 'linear'.")

    print(f"\n Cleaned using '{method}' method.")
    return cleaned_df

import pandas as pd

def handle_missing_values(df, method="ffill", fill_value=None):
    """
    Handles missing values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): The method to handle missing values.
                      Options: 'ffill', 'zero', 'interpolate', 'drop'
        fill_value (any): Value to use if method='zero' or a custom fill.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    if method == "ffill":
        return df.fillna(method='ffill')
    elif method == "zero":
        return df.fillna(0 if fill_value is None else fill_value)
    elif method == "interpolate":
        return df.interpolate(method='linear')
    elif method == "drop":
        return df.dropna()
    else:
        raise ValueError("Invalid method. Choose from 'ffill', 'zero', 'interpolate', 'drop'.")
def clean_and_align(df):
    """
    Remove any rows with missing values from a DataFrame 
    and return the cleaned DataFrame.
    """
    df_clean = df.dropna(how='any')
    print("Downloaded Data shape:", df_clean.shape)
    display(df_clean.head())
    return df_clean
def compute_returns(df):
    """
    Compute simple daily returns and log returns from a price DataFrame.

    Parameters:
    df (pd.DataFrame): Price DataFrame with dates as index and tickers as columns.

    Returns:
    pd.DataFrame: DataFrame containing both simple and log returns.
    """
    # Simple returns
    simple_returns = df.pct_change().dropna()
    simple_returns.columns = [f"{col}_Simple_Returns" for col in simple_returns.columns]
    
    # Log returns
    log_returns = np.log(df / df.shift(1)).dropna()
    log_returns.columns = [f"{col}_Log_Returns" for col in log_returns.columns]
    
    # Combine
    returns_df = pd.concat([simple_returns, log_returns], axis=1)
    return returns_df

def calculate_annualized_stats(returns_df, trading_days=252):
    """
    Calculate annualized mean return and volatility from returns DataFrame.
    
    Parameters:
    ----------
    returns_df : pd.DataFrame
        DataFrame containing simple returns for assets.
    trading_days : int, optional (default=252)
        Number of trading days per year.
        
    Returns:
    -------
    pd.DataFrame
        Annualized mean return and volatility for each asset.
    """
    returns_mean = returns_df.mean() * trading_days
    returns_std = returns_df.std() * np.sqrt(trading_days)

    summary = pd.DataFrame({
        'Annualized Mean Return': returns_mean,
        'Annualized Volatility': returns_std
    })
    
    print("\nAnnualized return and volatility:")
    return summary

def plot_prices_and_returns(df, returns):
    """
    Plot adjusted close prices and daily simple returns.

    Parameters:
    df (pd.DataFrame): Adjusted close prices DataFrame.
    returns (pd.DataFrame): Simple returns DataFrame.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot adjusted close prices
    df.plot(ax=axes[0], title='Adjusted Close Prices')
    axes[0].set_ylabel('Price (USD)')
    
    # Plot simple returns
    returns.plot(ax=axes[1], title='Daily Simple Returns')
    axes[1].axhline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_rolling_volatility(log_returns, window=30):
    """
    Plot rolling annualized volatility for each asset in the log returns DataFrame.

    Parameters:
    - log_returns (pd.DataFrame): DataFrame of daily log returns for each asset.
    - window (int): Rolling window size in days (default is 30).
    """
    rolling_std = log_returns.rolling(window).std()

    plt.figure(figsize=(14, 6))
    for col in log_returns.columns:
        plt.plot(
            rolling_std[col] * np.sqrt(252),
            label=f"{col} rolling annualized vol ({window}d)"
        )
    plt.legend()
    plt.title(f"{window}-day Rolling Annualized Volatility")
    plt.ylabel("Volatility")
    plt.show()

def detect_outlier_days(log_returns, threshold=3):
    """
    Detect outlier days in log returns using z-scores.
    
    Parameters:
    log_returns (pd.DataFrame): DataFrame of log returns for assets
    threshold (float): z-score threshold to flag outliers (default=3)
    
    Returns:
    pd.DataFrame: DataFrame of outlier days
    """
    # Compute z-scores
    z_scores = log_returns.apply(zscore)
    
    # Filter rows where absolute z-score exceeds the threshold for any asset
    outlier_days = z_scores[(z_scores.abs() > threshold).any(axis=1)]
    
    print(f"\nOutlier days (|z|>{threshold}) found: {len(outlier_days)}")
    return outlier_days