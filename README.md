# Stock Portfolio Analysis and Optimization

A Streamlit application for analyzing and optimizing stock portfolios using historical data from the Indian Stock Market (NSE).

## Overview

This project provides a web-based tool to:
- Analyze the historical performance and risk of selected stocks.
- Optimize portfolios by comparing equal-weighted and Sharpe Ratio-optimized portfolios.
- Visualize key metrics through interactive charts like cumulative returns, drawdowns, and efficient frontiers.

The tool fetches stock data from Yahoo Finance and uses the NIFTY 50 index as a benchmark for risk metrics like Beta.

## Features

- **Portfolio Analysis**: Evaluate individual stocks with metrics like CAGR, volatility, and Beta.
- **Portfolio Optimization**: Compare equal-weighted and optimized portfolios (up to 20 stocks).
- **Interactive Visualizations**: Includes plots for returns, volatility, drawdowns, and more.
- **Custom Date Range**: Select a date range for analysis (default is the past 7 years).

## Live Demo

Check out the live application here: [Stock Portfolio Analysis Streamlit App](https://stock-portfolio-analysis.streamlit.app/)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
	```
2. Install the required dependencies
	```bash
   pip install -r requirements.txt
	```
3. Run the Streamlit app:
	```bash
    streamlit run app.py
	```
	
## Disclaimer
This tool is for educational purposes only and should not be used as financial advice. Always consult a qualified financial advisor before making investment decisions.