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

# Calculate Value at Risk (VaR)
def calc_var(daily_returns, confidence_level=0.95):
    return daily_returns.quantile(1 - confidence_level)

# Calculate Conditional Value at Risk (CVaR)
def calc_cvar(daily_returns, confidence_level=0.95):
    var = calc_var(daily_returns, confidence_level)
    cvar = daily_returns[daily_returns <= var].mean()
    return cvar

# Calculate Beta (relative to NIFTY 50) - Fixed
def calc_beta_alpha(daily_returns, start_date, end_date,risk_free_rate = 0.05):
    market_df = yf.download('^NSEI',start_date, end_date )
    market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]
    market_returns = np.log(market_df['Close'] / market_df['Close'].shift(1)).dropna()
    
    # Initialize dictionaries to store results for each stock
    beta_values = []
    alpha_values = []
    expected_return_values = []
    stock_names = daily_returns.columns

    # Iterate over each stock's returns (each column in daily_returns)
    for stock in stock_names:
    
        stock_returns = daily_returns[stock].dropna()

        returns_df = pd.DataFrame({
            'stock_returns': stock_returns,
            'market_returns': market_returns
        }).dropna()

        covmat = np.cov(returns_df['stock_returns'], returns_df['market_returns'])
        beta = covmat[0,1] / covmat[1,1]

        annualized_return = stock_returns.mean() * 252
        market_annualized = market_returns.mean() * 252
        alpha = annualized_return - (risk_free_rate + beta * (market_annualized - risk_free_rate))
        expected_return = risk_free_rate + beta * (market_annualized - risk_free_rate)

        # Store results
        beta_values.append(beta)
        alpha_values.append(alpha)
        expected_return_values.append(expected_return)

    # Convert results to pandas Series with stock symbols as index
    beta_series = pd.Series(beta_values, index=stock_names, name='Beta')
    alpha_series = pd.Series(alpha_values, index=stock_names, name='Alpha')
    expected_return_series = pd.Series(expected_return_values, index=stock_names, name='Expected Return')

    return beta_series, alpha_series, expected_return_series


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

    beta, alpha, expected_return = calc_beta_alpha(daily_returns,start_date, end_date)
    
    metrics = {
        'Annualized Return': calc_annualized_return(daily_returns),
        'CAGR': calc_cagr(closes),
        'Overall Return': calc_overall_return(closes),
        'Volatility': daily_returns.std() * np.sqrt(252),
        'Variance': calc_portfolio_var(daily_returns, weights),
        'VaR (95%)': calc_var(daily_returns, confidence_level=0.95),
        'CVaR (95%)': calc_cvar(daily_returns, confidence_level=0.95),
        'Beta': beta,
        'Alpha': alpha,
        'Expected Return': expected_return,
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