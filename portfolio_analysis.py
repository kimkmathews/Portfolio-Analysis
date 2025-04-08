# portfolio_analysis.py
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Load stock list from CSV
def load_stock_data(csv_path="EQUITY_L.csv"):
    df = pd.read_csv(csv_path)
    df['SYMBOL'] = df['SYMBOL'] + '.NS'  # Append .NS for NSE stocks
    return df[['SYMBOL', 'NAME OF COMPANY']]

# Fetch historical stock price data
def get_historical_prices(symbols, start_date, end_date):
    df = yf.download(symbols, start=start_date, end=end_date, progress=False)['Close']
    # Convert Series to DataFrame if single stock
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=[symbols[0] if isinstance(symbols, list) else symbols])
    return df.dropna()

# Calculate daily returns
def calc_daily_returns(closes):
    return np.log(closes / closes.shift(1)).dropna()

# Calculate annualized return
def calc_annualized_return(daily_returns):
    return daily_returns.mean() * 252

# Calculate CAGR
def calc_cagr(closes):
    n_days = (closes.index[-1] - closes.index[0]).days
    n_years = n_days / 365.25  # Account for leap years
    if len(closes) < 2 or n_years <= 0:
        return pd.Series(np.nan, index=closes.columns)
    return (closes.iloc[-1] / closes.iloc[0]) ** (1 / n_years) - 1

# Calculate overall return
def calc_overall_return(closes):
    if len(closes) < 2:
        return pd.Series(np.nan, index=closes.columns)
    return (closes.iloc[-1] / closes.iloc[0]) - 1

# Calculate Sharpe ratio
def calc_sharpe_ratio(daily_returns, risk_free_rate=0.01):
    annualized_return = calc_annualized_return(daily_returns)
    annualized_vol = daily_returns.std() * np.sqrt(252)
    return (annualized_return - risk_free_rate) / annualized_vol

# Calculate portfolio variance
def calc_portfolio_var(daily_returns, weights=None):
    if weights is None:
        weights = np.ones(daily_returns.columns.size) / daily_returns.columns.size
    cov_matrix = daily_returns.cov() * 252  # Annualized covariance
    var = weights.T @ cov_matrix @ weights
    return var

# Calculate Maximum Drawdown
def calc_max_drawdown(closes):
    cumulative = (closes / closes.iloc[0]).cummax()
    drawdown = (closes / cumulative) - 1
    return drawdown.min()

# Calculate Sortino Ratio
def calc_sortino_ratio(daily_returns, risk_free_rate=0.01):
    annualized_return = calc_annualized_return(daily_returns)
    downside_returns = daily_returns[daily_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino = (annualized_return - risk_free_rate) / downside_deviation
    sortino = sortino.where(downside_deviation != 0, np.nan)
    return sortino

# Calculate Calmar Ratio
def calc_calmar_ratio(closes, daily_returns):
    annualized_return = calc_annualized_return(daily_returns)
    max_drawdown = calc_max_drawdown(closes)
    # Element-wise division, replacing with NaN where max_drawdown is 0
    calmar = annualized_return / abs(max_drawdown)
    calmar = calmar.where(max_drawdown != 0, np.nan)
    return calmar

# Calculate Skewness of Returns
def calc_skewness(daily_returns):
    return daily_returns.skew()

# Calculate Kurtosis of Returns
def calc_kurtosis(daily_returns):
    return daily_returns.kurtosis()

# Main analysis function to return all metrics
def analyze_portfolio(symbols, start_date, end_date):
    if isinstance(symbols, str):
        symbols = [symbols]
    
    closes = get_historical_prices(symbols, start_date, end_date)
    if closes.empty:
        raise ValueError("No historical data available for the selected stocks.")
    
    daily_returns = calc_daily_returns(closes)
    
    # Calculate weights for portfolio variance (equal weights for individual stock analysis)
    weights = np.ones(len(symbols)) / len(symbols)
    
    metrics = {
        'Annualized Return': calc_annualized_return(daily_returns),
        'CAGR': calc_cagr(closes),
        'Overall Return': calc_overall_return(closes),
        'Volatility': daily_returns.std() * np.sqrt(252),
        'Variance': calc_portfolio_var(daily_returns, weights),
        'Sharpe Ratio': calc_sharpe_ratio(daily_returns),
        'Sortino Ratio': calc_sortino_ratio(daily_returns),
        'Max Drawdown': calc_max_drawdown(closes),
        'Calmar Ratio': calc_calmar_ratio(closes, daily_returns),
        'Skewness': calc_skewness(daily_returns),
        'Kurtosis': calc_kurtosis(daily_returns)
    }
    print(metrics)
    return closes, daily_returns, pd.DataFrame(metrics, index=closes.columns)

if __name__ == "__main__":
    stock_df = load_stock_data()
    symbols = stock_df['SYMBOL'].head(3).tolist()
    start_date = (datetime.now() - timedelta(days=7*365)).date()
    end_date = datetime.now().date()
    closes, daily_returns, metrics = analyze_portfolio(symbols, start_date, end_date)
    print(metrics)