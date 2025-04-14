# portfolio_optimization.py
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import plotly.graph_objects as go
import plotly.express as px

# Fetch historical stock price data
def get_historical_prices(symbols, start_date, end_date):
    df = yf.download(symbols, start=start_date, end=end_date, progress=False)['Close']
    return df.dropna()

# Calculate daily returns for portfolio
def calc_daily_returns(closes):
    return np.log(closes / closes.shift(1)).dropna()

# Calculate portfolio performance with unified metrics
def portfolio_performance(weights, closes, start_date, end_date, risk_free_rate=0.01):
    daily_returns = calc_daily_returns(closes)
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    
    # Annualized Return
    annualized_return = portfolio_returns.mean() * 252
    
    # CAGR
    n_days = (closes.index[-1] - closes.index[0]).days
    n_years = n_days / 365.25
    cagr = ((1 + portfolio_returns).cumprod().iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    
    # Overall Return
    overall_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    
    # Volatility and Variance
    cov_matrix = daily_returns.cov() * 252
    volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    variance = volatility ** 2
    
    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else np.nan
    
    # Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.nan
    
    # Maximum Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() * 100

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # Skewness and Kurtosis
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis()
    
    # VaR and CVaR
    var_95 = portfolio_returns.quantile(0.05)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    # Beta, Alpha, and Expected Return
    market_df = yf.download('^NSEI', start=start_date, end=end_date)
    market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]
    market_returns = np.log(market_df['Close'] / market_df['Close'].shift(1)).dropna()
    
    returns_df = pd.DataFrame({
        'portfolio_returns': portfolio_returns,
        'market_returns': market_returns
    }).dropna()

    covmat = np.cov(returns_df['portfolio_returns'], returns_df['market_returns'])
    beta = covmat[0, 1] / covmat[1, 1]

    annualized_portfolio_return = portfolio_returns.mean() * 252
    market_annualized = market_returns.mean() * 252
    alpha = annualized_portfolio_return - (risk_free_rate + beta * (market_annualized - risk_free_rate))
    expected_return = risk_free_rate + beta * (market_annualized - risk_free_rate)

    return {
        'Annualized Return': annualized_return,
        'CAGR': cagr,
        'Overall Return': overall_return,
        'Volatility': volatility,
        'Variance': variance,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Beta': beta,
        'Alpha': alpha,
        'Expected Return': expected_return,
        'Portfolio Returns': portfolio_returns,
        'Cumulative Returns': cumulative
    }

# Equal-weighted portfolio
def equal_weighted_portfolio(symbols, start_date, end_date):
    closes = get_historical_prices(symbols, start_date, end_date)
    weights = np.ones(len(symbols)) / len(symbols)
    performance = portfolio_performance(weights, closes, start_date, end_date)
    return weights, performance

# Optimized portfolio (Max Sharpe Ratio)
def optimized_portfolio(symbols, start_date, end_date):
    closes = get_historical_prices(symbols, start_date, end_date)
    mu = expected_returns.mean_historical_return(closes)
    S = risk_models.sample_cov(closes)
    
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe(risk_free_rate=0.01)
    cleaned_weights = ef.clean_weights()
    
    weights = np.array(list(cleaned_weights.values()))
    performance = portfolio_performance(weights, closes, start_date, end_date)
    return weights, performance

# Generate Efficient Frontier for visualization with debug
def generate_efficient_frontier(symbols, start_date, end_date):
    closes = get_historical_prices(symbols, start_date, end_date)
    mu = expected_returns.mean_historical_return(closes)
    S = risk_models.sample_cov(closes)
    
    print(f"Expected Returns (mu): {mu}")
    print(f"Maximum Expected Return: {mu.max()}")
    print(f"Minimum Expected Return: {mu.min()}")
    
    ef = EfficientFrontier(mu, S)
    
    # Calculate the maximum achievable return for safety
    ef_max = EfficientFrontier(mu, S)
    ef_max.max_sharpe(risk_free_rate=0.01)
    max_return = ef_max.portfolio_performance(verbose=True)[0]
    print(f"Maximum Achievable Return: {max_return}")
    
    # Generate target returns, ensuring the max is below the achievable maximum
    target_returns = np.linspace(mu.min(), max_return * 0.99, 50)
    print(f"Target Returns Range: {target_returns[0]} to {target_returns[-1]}")
    
    frontier_vols = []
    frontier_returns = []
    for target_return in target_returns:
        ef = EfficientFrontier(mu, S)
        try:
            ef.efficient_return(target_return)
            weights = np.array(list(ef.clean_weights().values()))
            ret, vol, _ = ef.portfolio_performance(risk_free_rate=0.01)
            frontier_returns.append(ret)
            frontier_vols.append(vol)
            print(f"Target: {target_return}, Return: {ret}, Volatility: {vol}")
        except ValueError as e:
            print(f"Error for target {target_return}: {e}")
            continue
    
    return frontier_vols, frontier_returns

# Main optimization function
def optimize_portfolio(symbols, start_date, end_date):
    # Equal-weighted
    eq_weights, eq_performance = equal_weighted_portfolio(symbols, start_date, end_date)
    
    # Optimized
    opt_weights, opt_performance = optimized_portfolio(symbols, start_date, end_date)
    
    # Portfolio performance metrics
    results = pd.DataFrame({
        'Portfolio': ['Equal-Weighted', 'Optimized'],
        'Annualized Return': [eq_performance['Annualized Return'], opt_performance['Annualized Return']],
        'CAGR': [eq_performance['CAGR'], opt_performance['CAGR']],
        'Overall Return': [eq_performance['Overall Return'], opt_performance['Overall Return']],
        'Volatility': [eq_performance['Volatility'], opt_performance['Volatility']],
        'Variance': [eq_performance['Variance'], opt_performance['Variance']],
        'VaR (95%)': [eq_performance['VaR (95%)'], opt_performance['VaR (95%)']],
        'CVaR (95%)': [eq_performance['CVaR (95%)'], opt_performance['CVaR (95%)']],
        'Beta': [eq_performance['Beta'], opt_performance['Beta']],
        'Alpha': [eq_performance['Alpha'], opt_performance['Alpha']],
        'Expected Return': [eq_performance['Expected Return'], opt_performance['Expected Return']],
        'Sharpe Ratio': [eq_performance['Sharpe Ratio'], opt_performance['Sharpe Ratio']],
        'Sortino Ratio': [eq_performance['Sortino Ratio'], opt_performance['Sortino Ratio']],
        'Max Drawdown': [eq_performance['Max Drawdown'], opt_performance['Max Drawdown']],
        'Calmar Ratio': [eq_performance['Calmar Ratio'], opt_performance['Calmar Ratio']],
        'Skewness': [eq_performance['Skewness'], opt_performance['Skewness']],
        'Kurtosis': [eq_performance['Kurtosis'], opt_performance['Kurtosis']]
        
    })
    
    # Portfolio weights
    weights_df = pd.DataFrame({
        'Stock': symbols,
        'Equal-Weighted': eq_weights,
        'Optimized': opt_weights
    })
    
    # Efficient frontier data for visualization
    frontier_vols, frontier_returns = generate_efficient_frontier(symbols, start_date, end_date)
    
    return results, weights_df, frontier_vols, frontier_returns, eq_performance['Portfolio Returns'], opt_performance['Portfolio Returns'], eq_performance['Cumulative Returns'], opt_performance['Cumulative Returns']

if __name__ == "__main__":
    from portfolio_analysis import load_stock_data
    stock_df = load_stock_data()
    symbols = stock_df['SYMBOL'].head(5).tolist()
    start_date = (datetime.now() - timedelta(days=7*365)).date()
    end_date = datetime.now().date()
    results, weights, frontier_vols, frontier_returns, eq_returns, opt_returns, eq_cumulative, opt_cumulative = optimize_portfolio(symbols, start_date, end_date)
    print("Portfolio Performance:\n", results)
    print("\nWeights:\n", weights)