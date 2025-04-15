# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from portfolio_analysis import load_stock_data, analyze_portfolio, get_historical_prices, calc_daily_returns
from portfolio_optimization import optimize_portfolio
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit App
st.title("Portfolio Analysis & Optimization - Indian Stock Market")

# Load stock data
stock_df = load_stock_data()

# Sidebar Navigation with Radio Buttons
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Portfolio Analysis", "Portfolio Optimization"], index=None)

# Sidebar Date Picker
st.sidebar.header("Select Date Range")
default_end_date = datetime.now().date()
default_start_date = datetime(2018, 1, 1).date()  # Changed to Jan 1, 2018
start_date = st.sidebar.date_input("Start Date", value=default_start_date, min_value=datetime(2000, 1, 1).date(), max_value=default_end_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date, max_value=default_end_date)

# Sidebar Footnote with LinkedIn Link
st.sidebar.markdown(
    """
    <br><br>
    <footer style='text-align: center; font-size: 16px;'>
        Developed by <a href='https://www.linkedin.com/in/kim-kmathews/' target='_blank'>Kim Mathews</a>
    </footer>
    """,
    unsafe_allow_html=True
)
# Sidebar Stock Search
filtered_df = stock_df
stock_options = [f"{row['SYMBOL']} - {row['NAME OF COMPANY']}" for _, row in filtered_df.iterrows()]
stock_symbols = filtered_df['SYMBOL'].tolist()

# Custom CSS to increase table width and adjust column widths
st.markdown(
    """
    <style>
    .dataframe {
        width: 100% !important;
        max-width: 1200px !important;  /* Set a maximum width for the table */
        margin: 0 auto;  /* Center the table */
    }
    .dataframe th, .dataframe td {
        min-width: 150px !important;  /* Ensure columns are wide enough */
        padding: 8px !important;  /* Add padding for better spacing */
        text-align: center !important;  /* Center-align text */
    }
    .dataframe th:first-child {
        min-width: 200px !important;  /* Make the first column (metric names) wider */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Default Information Display if No Page is Selected
if page is None:
    st.header("Welcome to the Portfolio Analysis & Optimization Tool")
    st.write("""
    This application is designed to help you analyze and optimize investment portfolios using stocks from the Indian stock market (NSE). 
    It leverages historical data to provide insights into stock performance and portfolio optimization based on modern portfolio theory. 
    Whether you're a beginner exploring investment options or an experienced investor looking for data-driven insights, this tool offers a range of metrics and visualizations to assist in your decision-making process.

    ### Purpose of This Tool
    The primary goal of this tool is to provide educational insights into portfolio analysis and optimization. It allows you to:
    - **Analyze Individual Stocks:** Evaluate the historical performance and risk of selected stocks using a variety of financial metrics.
    - **Compare Portfolios:** Compare an equal-weighted portfolio with an optimized portfolio (maximized Sharpe Ratio) to understand the benefits of portfolio optimization.
    - **Visualize Trends:** Gain insights through interactive charts such as cumulative returns, drawdowns, rolling volatility, and more.

    ### Features
    - **Portfolio Analysis**: Evaluate the performance of selected stocks over a user-defined period (default is the past 7 years) with detailed metrics and visualizations.
    - **Portfolio Optimization**: Compare an equal-weighted portfolio with an optimized portfolio (maximized Sharpe Ratio) for up to 20 stocks, helping you understand how optimization can improve risk-adjusted returns.
    - **Stock Search**: Easily search for stocks by symbol or company name using the sidebar search bar.

    ### How to Use
    1. Use the radio buttons on the left to select either "Portfolio Analysis" or "Portfolio Optimization".
    2. Select the date range for analysis using the date pickers in the sidebar.
    3. Search for stocks in the sidebar and select them from the multi-select dropdown on the main page.
    4. View detailed performance metrics and visualizations tailored to your selections.

    ### Data Source
    - Stock data is fetched from Yahoo Finance using the `.NS` suffix for NSE-listed companies.
    - The stock list is sourced from the official NSE `EQUITY_L.csv` file.
    """)

    st.write("### Metrics Explained")
    st.write("This tool uses a variety of financial metrics to evaluate stock and portfolio performance. Below is a description of each metric to help you understand their significance:")
    st.markdown("""
    - **Annualized Return**: The average yearly return of an investment, assuming the returns are compounded annually. It helps you understand the expected yearly growth rate over the selected period.
    - **Compound Annual Growth Rate (CAGR)**: The annual growth rate of an investment over a specified period, assuming profits are reinvested. It provides a smoothed annual return, ignoring volatility, and is useful for comparing long-term performance.
    - **Overall Return**: The total percentage return of an investment over the entire period, calculated as $ ( (Final Value / Initial Value) - 1 ) \\times 100 $. It shows the total growth or loss without annualizing.
    - **Volatility**: A measure of the stock or portfolio’s price fluctuations, calculated as the annualized standard deviation of daily returns. Higher volatility indicates greater risk due to larger price swings.
    - **Variance**: The square of volatility, representing the dispersion of returns around the mean. It’s a statistical measure of risk, often used in portfolio optimization.
    - **VaR (95%) (Value at Risk)**: The maximum expected loss over a given period at a 95% confidence level, based on historical returns. For example, a VaR of -5% means there’s a 5% chance of losing more than 5% in a single day.
    - **CVaR (95%) (Conditional Value at Risk)**: The average loss expected in the worst 5% of cases, providing a measure of the tail risk beyond the VaR. It’s useful for understanding the severity of extreme losses.
    - **Beta**: A measure of the stock or portfolio’s sensitivity to market movements, calculated relative to the NIFTY 50 index. A beta of 1 means the stock moves with the market, >1 means it’s more volatile, and <1 means it’s less volatile.
    - **Alpha**: The excess return of the stock or portfolio relative to the expected return based on its beta, calculated using the Capital Asset Pricing Model (CAPM). Positive alpha indicates outperformance compared to the market.
    - **Expected Return**: The theoretical return of the stock or portfolio based on the CAPM, calculated as $ RiskFreeRate + Beta \\times (MarketReturn - RiskFreeRate) $. It represents the return expected given the stock’s market risk.
    - **Sharpe Ratio**: A risk-adjusted return metric, calculated as $ (Annualized Return - RiskFreeRate) / Volatility $. It measures the excess return per unit of risk, with higher values indicating better risk-adjusted performance (assumes a risk-free rate of 1%).
    - **Sortino Ratio**: Similar to the Sharpe Ratio but focuses on downside risk, calculated as $ (Annualized Return - RiskFreeRate) / DownsideDeviation $. It penalizes only negative volatility, making it useful for investors concerned about losses.
    - **Max Drawdown**: The largest peak-to-trough decline in the value of an investment, expressed as a percentage. It measures the worst-case loss during the period, helping you assess the potential downside risk.
    - **Calmar Ratio**: A risk-adjusted return metric, calculated as $ Annualized Return / |Max Drawdown| $. It measures return per unit of maximum drawdown, with higher values indicating better performance relative to the worst loss.
    - **Skewness**: A measure of the asymmetry of the return distribution. Positive skewness indicates more frequent small losses and occasional large gains, while negative skewness indicates more frequent small gains and occasional large losses.
    - **Kurtosis**: A measure of the "tailedness" of the return distribution. High kurtosis indicates a higher likelihood of extreme returns (both positive and negative), while low kurtosis suggests more moderate returns.
    """, unsafe_allow_html=True)

    st.write("""
    ### Disclaimer
    **For Educational Purposes Only:** This tool is intended solely for educational purposes to help users understand portfolio analysis and optimization concepts. It is not intended as financial advice. Any investment decisions should be made after consulting with a qualified financial advisor. Past performance is not a guarantee of future results, and investing in the stock market involves risks, including the potential loss of principal.

    Start by selecting a page from the sidebar to explore your portfolio options!
    """)
elif page == "Portfolio Analysis":
    st.header("Portfolio Analysis")
    selected_options = st.multiselect("Select stocks", stock_options, default=[stock_options[1857]])

    if selected_options:
        selected_indices = [stock_options.index(opt) for opt in selected_options]
        selected_stocks = [stock_symbols[i] for i in selected_indices]
        
        # Sort selected_stocks to ensure consistent ordering
        selected_stocks.sort()
        
        # Also sort selected_options to match the order of selected_stocks
        sorted_options = sorted(selected_options, key=lambda x: x.split(" - ")[0])
        
        try:
            closes, daily_returns, metrics = analyze_portfolio(selected_stocks, start_date=start_date, end_date=end_date)
            
            st.subheader(f"Performance Metrics ({start_date} to {end_date})")
            # Set the index using selected_stocks to ensure correct alignment
            metrics.index = selected_stocks  # Use the exact stock symbols passed to analyze_portfolio
            metrics_transposed = metrics.T  # Transpose the DataFrame
            # Round the values to 3 decimal places for better readability
            metrics_transposed = metrics_transposed.round(3)
            # Display the table with custom width
            st.dataframe(metrics_transposed, use_container_width=True)
            
            # Cumulative Returns Plot (Percentage from Initial Price)
            initial_price = closes.iloc[0]
            cumulative_returns = ((closes / initial_price) - 1) * 100  # Percentage change
            fig_cum = px.line(cumulative_returns, title="Cumulative Returns (%)")
            st.plotly_chart(fig_cum, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart displays the percentage growth of each stock's price over time, starting from 0% at the initial date. It helps you visualize how the value of each stock has changed relative to its starting price.
            """)

            # Price History Plot
            fig_price = px.line(closes, title="Adjusted Close Price History")
            st.plotly_chart(fig_price, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the adjusted closing price of each stock over time. It provides a view of the actual price movements, accounting for splits, dividends, and other adjustments.
            """)
            
            # Daily Returns Distribution
            fig_returns = px.histogram(daily_returns, nbins=50, title="Distribution of Daily Returns",
                                       marginal="box", barmode='overlay')
            st.plotly_chart(fig_returns, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This histogram displays the distribution of daily returns for each stock, with a box plot on top. It helps you understand the frequency and range of daily price changes, highlighting the variability and potential outliers in returns.
            """)
            
            # Drawdown Plot
            drawdowns = pd.DataFrame(index=closes.index)
            for stock in closes.columns:
                # Calculate the cumulative maximum directly on the price data
                cumulative_max = closes[stock].cummax()
                # Standard drawdown formula: (current - peak) / peak * 100
                drawdowns[stock] = ((closes[stock] - cumulative_max) / cumulative_max) * 100
            fig_drawdown = px.line(
                drawdowns,
                title="Drawdown Over Time (%)",
                labels={"value": "Drawdown (%)", "index": "Date"}
            )
            fig_drawdown.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                legend_title="Stock",
                hovermode="x unified"
            )
            for trace in fig_drawdown.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Drawdown: %{y:.2f}%<br>" +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_drawdown, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the percentage decline from the highest value (peak) for each stock over time. A value of -20% means the stock dropped 20% from its peak. The drawdown will always be between 0% (no decline) and -100% (complete loss).
            """)

            # Rolling Volatility Plot
            rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(252)
            fig_vol = px.line(
                rolling_vol,
                title="Rolling Volatility (30-Day Window)",
                labels={"value": "Annualized Volatility", "index": "Date"}
            )
            fig_vol.update_layout(
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                legend_title="Stock",
                hovermode="x unified"
            )
            for trace in fig_vol.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Volatility: %{y:.4f}<br>" +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_vol, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the rolling 30-day volatility (risk) of each stock over time. Higher values indicate periods of greater price fluctuations (higher risk), while lower values indicate more stability.
            """)

            # Common Metric Comparison (replacing Risk Metrics Comparison)
            st.subheader("Common Metric Comparison")
            available_metrics = metrics.columns.tolist()  # List of all available metrics
            selected_metric = st.selectbox("Select a metric to compare:", available_metrics, index=available_metrics.index('CAGR'))
            metric_data = pd.DataFrame({
                'Stock': selected_stocks,
                selected_metric: metrics[selected_metric]
            })
            fig_metric = px.bar(
                metric_data,
                x=selected_metric,
                y='Stock',
                title=f"Comparison of {selected_metric} Across Stocks",
                labels={selected_metric: selected_metric, 'Stock': 'Stock'},
                orientation='h'
            )
            fig_metric.update_layout(
                xaxis_title=selected_metric,
                yaxis_title="Stock",
                hovermode="y unified"
            )
            st.plotly_chart(fig_metric, use_container_width=True)
            st.write(f"""
            **What This Plot Shows:**  
            This chart compares the selected metric ({selected_metric}) across all selected stocks. Use the dropdown to choose different metrics for comparison.
            """)

            # Rolling Beta Plot
            market_df = yf.download('^NSEI', start=start_date, end=end_date)
            market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]
            market_returns = calc_daily_returns(market_df['Close'])
            rolling_beta = pd.DataFrame(index=daily_returns.index)
            for stock in daily_returns.columns:
                rolling_beta[stock] = daily_returns[stock].rolling(window=30).cov(market_returns) / market_returns.rolling(window=30).var()
            fig_beta = px.line(
                rolling_beta,
                title="Rolling Beta (30-Day Window) vs NIFTY 50",
                labels={"value": "Beta", "index": "Date"}
            )
            fig_beta.update_layout(
                xaxis_title="Date",
                yaxis_title="Beta",
                legend_title="Stock",
                hovermode="x unified"
            )
            for trace in fig_beta.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Beta: %{y:.2f}<br>" +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_beta, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the rolling 30-day beta of each stock compared to the NIFTY 50 index. A beta of 1 means the stock moves with the market, >1 means it’s more volatile, and <1 means it’s less volatile.
            """)

            # Annual Returns Heatmap
            annual_returns = daily_returns.resample('Y').sum() * 100  # Convert to percentage
            annual_returns.index = annual_returns.index.year
            fig_annual = px.imshow(
                annual_returns.T,
                labels=dict(x="Year", y="Stock", color="Annual Return (%)"),
                x=annual_returns.index,
                y=annual_returns.columns,
                color_continuous_scale='RdBu',
                title="Annual Returns by Year (%)",
                text_auto=True
            )
            fig_annual.update_layout(
                xaxis_title="Year",
                yaxis_title="Stock",
                coloraxis_colorbar_title="Return (%)",
                width=600,
                height=400
            )
            st.plotly_chart(fig_annual, use_container_width=True)
            st.write("""
            **What This Heatmap Shows:**  
            This heatmap shows the annual returns for each stock over the years. Red indicates negative returns, blue indicates positive returns, and the intensity shows the magnitude.
            """)
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Portfolio Optimization Page
elif page == "Portfolio Optimization":
    st.header("Portfolio Optimization")
    selected_options = st.multiselect("Select up to 20 stocks", stock_options, default=[stock_options[1857],
                                                                                        stock_options[1514],
                                                                                        stock_options[732]])
    
    if len(selected_options) > 20:
        st.error("Please select a maximum of 20 stocks.")
    elif selected_options:
        selected_indices = [stock_options.index(opt) for opt in selected_options]
        selected_stocks = [stock_symbols[i] for i in selected_indices]
        
        try:
            results, weights, frontier_vols, frontier_returns, eq_returns, opt_returns, eq_cumulative, opt_cumulative = optimize_portfolio(selected_stocks, start_date=start_date, end_date=end_date)
            
            # Portfolio Performance Metrics
            st.subheader(f"Portfolio Performance Metrics ({start_date} to {end_date})")
            # Transpose the results DataFrame: metrics as rows, portfolios as columns
            results_transposed = results.set_index('Portfolio').T  # Set 'Portfolio' as index and transpose
            # Round the values to 3 decimal places for better readability
            results_transposed = results_transposed.round(3)
            # Display the table with custom width
            st.dataframe(results_transposed, use_container_width=True)

            # Portfolio Weights
            st.subheader("Portfolio Weights")
            weights['Stock'] = [opt.split(" - ")[0] for opt in selected_options]
            st.write(weights)
            
            # Pie Charts for Weights - Fixed alignment
            st.subheader("Portfolio Allocation")
            col1, col2 = st.columns([1, 1])  # Ensure equal column widths
            with col1:
                fig_eq = px.pie(weights, values="Equal-Weighted", names="Stock", title="Equal-Weighted Portfolio Allocation")
                fig_eq.update_layout(
                    width=400,  # Set a fixed width
                    height=400,  # Set a fixed height
                    margin=dict(l=10, r=10, t=50, b=10)  # Adjust margins to prevent overlap
                )
                st.plotly_chart(fig_eq, use_container_width=True)
            with col2:
                fig_opt = px.pie(weights, values="Optimized", names="Stock", title="Optimized Portfolio Allocation")
                fig_opt.update_layout(
                    width=400,  # Set a fixed width
                    height=400,  # Set a fixed height
                    margin=dict(l=10, r=10, t=50, b=10)  # Adjust margins to prevent overlap
                )
                st.plotly_chart(fig_opt, use_container_width=True)
            st.write("""
            **What These Plots Show:**  
            The pie charts display the allocation of the equal-weighted and optimized portfolios. The equal-weighted portfolio assigns the same weight to each stock (e.g., 33.3% for 3 stocks), while the optimized portfolio adjusts weights to maximize the Sharpe Ratio, balancing risk and return.
            """)

            # Add Correlation Matrix Heatmap
            closes = get_historical_prices(selected_stocks, start_date, end_date)
            daily_returns = calc_daily_returns(closes)
            corr_matrix = daily_returns.corr()
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(x="Stocks", y="Stocks", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale='reds',
                zmin=0, zmax=1,
                title="Correlation Matrix of Selected Stocks",
                text_auto=True
            )
            fig_corr.update_layout(
                width=600,
                height=600,
                xaxis_title="Stocks",
                yaxis_title="Stocks",
                coloraxis_colorbar_title="Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.write("""
            **What This Heatmap Shows:**  
            This heatmap displays the correlation between the daily returns of selected stocks. Values range from 0 to 1, where 1 indicates perfect positive correlation (stocks move together), and 0 indicates no correlation. It helps identify diversification opportunities.
            """)

            # Efficient Frontier Plot
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=frontier_vols,
                y=frontier_returns,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue')
            ))
            fig_frontier.add_trace(go.Scatter(
                x=[results.loc[0, 'Volatility']],
                y=[results.loc[0, 'Annualized Return']],
                mode='markers',
                name='Equal-Weighted',
                marker=dict(size=10, color='red')
            ))
            fig_frontier.add_trace(go.Scatter(
                x=[results.loc[1, 'Volatility']],
                y=[results.loc[1, 'Annualized Return']],
                mode='markers',
                name='Optimized',
                marker=dict(size=10, color='green')
            ))
            fig_frontier.update_layout(
                xaxis_title="Volatility (Risk)",
                yaxis_title="Annualized Return",
                title="Efficient Frontier with Portfolios"
            )
            st.plotly_chart(fig_frontier, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This plot shows the Efficient Frontier, which represents the set of portfolios offering the highest expected return for a given level of risk (volatility). The red dot marks the equal-weighted portfolio, and the green dot marks the optimized portfolio with the highest Sharpe Ratio.
            """)
            
            # Cumulative Returns Plot
            cum_returns_df = pd.DataFrame({
                'Equal-Weighted': ((eq_cumulative - 1) * 100),
                'Optimized': ((opt_cumulative - 1) * 100)
            })
            fig_cum = px.line(cum_returns_df, title="Cumulative Returns Over Time")
            st.plotly_chart(fig_cum, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart displays the percentage growth of the equal-weighted and optimized portfolios over time, starting from 0% at the initial date. It helps compare the performance of the two portfolios.
            """)
            
            # Daily Returns Distribution
            returns_df = pd.DataFrame({
                'Equal-Weighted': eq_returns,
                'Optimized': opt_returns
            })
            fig_dist = px.histogram(returns_df, nbins=50, title="Distribution of Daily Returns", marginal="box",
                                    barmode='overlay')
            st.plotly_chart(fig_dist, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This histogram displays the distribution of daily returns for the equal-weighted and optimized portfolios, with a box plot on top. It helps compare the variability and potential outliers in daily returns between the two portfolios.
            """)

            # Drawdown Plot
            eq_cum_max = eq_cumulative.cummax()
            drawdown_eq = ((eq_cumulative - eq_cum_max) / eq_cum_max) * 100
            opt_cum_max = opt_cumulative.cummax()
            drawdown_opt = ((opt_cumulative - opt_cum_max) / opt_cum_max) * 100
            drawdown_df = pd.DataFrame({
                'Equal-Weighted': drawdown_eq,
                'Optimized': drawdown_opt
            })
            fig_drawdown = px.line(
                drawdown_df,
                title="Drawdown Over Time (%)",
                labels={"value": "Drawdown (%)", "index": "Date"}
            )
            fig_drawdown.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                legend_title="Portfolio",
                hovermode="x unified"
            )
            for trace in fig_drawdown.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Drawdown: %{y:.2f}%<br>" +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_drawdown, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the percentage decline from the highest value (peak) for each portfolio over time. A value of -20% means the portfolio dropped 20% from its peak. The drawdown will always be between 0% (no decline) and -100% (complete loss).
            """)

            # Rolling Volatility Plot
            rolling_vol_eq = eq_returns.rolling(window=30).std() * np.sqrt(252)
            rolling_vol_opt = opt_returns.rolling(window=30).std() * np.sqrt(252)
            vol_df = pd.DataFrame({
                'Equal-Weighted': rolling_vol_eq,
                'Optimized': rolling_vol_opt
            })
            fig_vol = px.line(
                vol_df,
                title="Rolling Volatility (30-Day Window)",
                labels={"value": "Annualized Volatility", "index": "Date"}
            )
            fig_vol.update_layout(
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                legend_title="Portfolio",
                hovermode="x unified"
            )
            for trace in fig_vol.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Volatility: %{y:.4f}<br>" +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_vol, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the rolling 30-day volatility (risk) of each portfolio over time. Higher values indicate periods of greater price fluctuations (higher risk), while lower values indicate more stability.
            """)

            # Common Metric Comparison (replacing Risk Metrics Comparison)
            st.subheader("Common Metric Comparison")
            available_metrics = results.columns.tolist()
            available_metrics.remove('Portfolio')  # Remove 'Portfolio' column from the list
            selected_metric = st.selectbox("Select a metric to compare:", available_metrics, index=available_metrics.index('CAGR'), key="portfolio_opt_metric")
            metric_data = pd.DataFrame({
                'Portfolio': ['Equal-Weighted', 'Optimized'],
                selected_metric: results[selected_metric]
            })
            fig_metric = px.bar(
                metric_data,
                x=selected_metric,
                y='Portfolio',
                title=f"Comparison of {selected_metric} Between Portfolios",
                labels={selected_metric: selected_metric, 'Portfolio': 'Portfolio'},
                orientation='h'
            )
            fig_metric.update_layout(
                xaxis_title=selected_metric,
                yaxis_title="Portfolio",
                hovermode="y unified"
            )
            st.plotly_chart(fig_metric, use_container_width=True)
            st.write(f"""
            **What This Plot Shows:**  
            This chart compares the selected metric ({selected_metric}) between the equal-weighted and optimized portfolios. Use the dropdown to choose different metrics for comparison.
            """)

            # Rolling Beta Plot
            market_df = yf.download('^NSEI', start=start_date, end=end_date)
            market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]
            market_returns = calc_daily_returns(market_df['Close'])
            rolling_beta_eq = pd.Series(index=eq_returns.index)
            rolling_beta_opt = pd.Series(index=opt_returns.index)
            for i in range(30, len(eq_returns)):
                cov_eq = eq_returns[i-30:i].cov(market_returns[i-30:i])
                var_market = market_returns[i-30:i].var()
                rolling_beta_eq.iloc[i] = cov_eq / var_market if var_market != 0 else np.nan
                cov_opt = opt_returns[i-30:i].cov(market_returns[i-30:i])
                rolling_beta_opt.iloc[i] = cov_opt / var_market if var_market != 0 else np.nan
            beta_df = pd.DataFrame({
                'Equal-Weighted': rolling_beta_eq,
                'Optimized': rolling_beta_opt
            })
            fig_beta = px.line(
                beta_df,
                title="Rolling Beta (30-Day Window) vs NIFTY 50",
                labels={"value": "Beta", "index": "Date"}
            )
            fig_beta.update_layout(
                xaxis_title="Date",
                yaxis_title="Beta",
                legend_title="Portfolio",
                hovermode="x unified"
            )
            for trace in fig_beta.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Date: %{x}<br>" +
                    "Beta: %{y:.2f}<br>" +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_beta, use_container_width=True)
            st.write("""
            **What This Plot Shows:**  
            This chart shows the rolling 30-day beta of each portfolio compared to the NIFTY 50 index. A beta of 1 means the portfolio moves with the market, >1 means it’s more volatile, and <1 means it’s less volatile.
            """)
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")