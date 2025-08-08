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
