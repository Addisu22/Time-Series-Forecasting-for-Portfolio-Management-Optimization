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

