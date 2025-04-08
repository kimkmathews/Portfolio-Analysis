# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from portfolio_analysis import load_stock_data, analyze_portfolio,get_historical_prices,calc_daily_returns
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
default_start_date = (datetime.now() - timedelta(days=7*365)).date()
start_date = st.sidebar.date_input("Start Date", value=default_start_date, min_value=datetime(2000, 1, 1).date(), max_value=default_end_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date, max_value=default_end_date)

# Sidebar Stock Search
st.sidebar.header("Stock Search")
search_query = st.sidebar.text_input("Search by Symbol or Company Name", "")

# Filter stocks based on search query
if search_query:
    filtered_df = stock_df[
        stock_df['SYMBOL'].str.contains(search_query.upper(), case=False) |
        stock_df['NAME OF COMPANY'].str.contains(search_query, case=False)
    ]
else:
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
    It leverages historical data to provide insights and optimize your portfolio based on modern portfolio theory.

    ### Features:
    - **Portfolio Analysis**: Evaluate the performance of selected stocks over the past 7 years with metrics such as:
        - Annualized Return
        - Compound Annual Growth Rate (CAGR)
        - Overall Return
        - Sharpe Ratio
        - Volatility
    - **Portfolio Optimization**: Compare an equal-weighted portfolio with an optimized portfolio (maximized Sharpe Ratio) for up to 20 stocks.
    - **Stock Search**: Easily search for stocks by symbol or company name using the sidebar search bar.

    ### How to Use:
    1. Use the radio buttons on the left to select either "Portfolio Analysis" or "Portfolio Optimization".
    2. Select the date range for analysis using the date pickers in the sidebar.
    3. Search for stocks in the sidebar and select them from the multi-select dropdown on the main page.
    4. View detailed performance metrics and visualizations tailored to your selections.

    ### Data Source:
    - Stock data is fetched from Yahoo Finance using the `.NS` suffix for NSE-listed companies.
    - The stock list is sourced from the official NSE `EQUITY_L.csv` file.

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
            st.plotly_chart(fig_cum)

            # Price History Plot
            fig_price = px.line(closes, title="Adjusted Close Price History")
            st.plotly_chart(fig_price)
            
            # Daily Returns Distribution
            daily_returns_melted = daily_returns.melt(var_name="Ticker", value_name="Daily Return")
            fig_returns = px.histogram(daily_returns_melted, 
                                        x="Daily Return",
                                        color="Ticker",
                                        nbins=50, 
                                        title="Distribution of Daily Returns",
                                        opacity=0.6,
                                        histnorm='probability density',
                                        barmode='overlay')
            #st.plotly_chart(fig_returns)
            fig_returns.update_layout(
                xaxis_title="Daily Return",
                yaxis_title="Density" if 'probability density' in fig_returns.data[0].histnorm else "Count",
                legend_title="Stock",
                bargap=0.1,  # Add a small gap between bars for clarity
                hovermode="x unified"  # Show hover info for all stocks at the same x-value
            )
            # Customize hover template
            for trace in fig_returns.data:
                trace.hovertemplate = (
                    "<b>%{fullData.name}</b><br>" +
                    "Daily Return: %{x:.4f}<br>" +
                    ("Density: %{y:.4f}<br>" if 'probability density' in trace.histnorm else "Count: %{y}<br>") +
                    "<extra></extra>"
                )
            st.plotly_chart(fig_returns, use_container_width=True)

            fig_box = px.box(daily_returns_melted, x="Ticker", y="Daily Return", title="Box Plot of Daily Returns")
            st.plotly_chart(fig_box, use_container_width=True)

            fig_returns = px.histogram(daily_returns, nbins=50, title="Distribution of Daily Returns",
                                       marginal="box",
                                    barmode='overlay')
            st.plotly_chart(fig_returns)

            
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Portfolio Optimization Page
elif page == "Portfolio Optimization":
    st.header("Portfolio Optimization")
    selected_options = st.multiselect("Select up to 20 stocks", stock_options, default=[stock_options[1857]])
    
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

            # Add Correlation Matrix Heatmap
            st.subheader("Correlation Matrix of Stocks")
            # Fetch historical prices for the selected stocks
            closes = get_historical_prices(selected_stocks, start_date, end_date)
            # Calculate daily returns
            daily_returns = calc_daily_returns(closes)
            # Compute the correlation matrix
            corr_matrix = daily_returns.corr()
            # Create a heatmap
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(x="Stocks", y="Stocks", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale='reds',  # Diverging color scale for positive/negative correlations
                zmin=0, zmax=1,  # Correlation ranges from -1 to 1
                title="Correlation Matrix of Selected Stocks",
                text_auto=True  # Display correlation values on the heatmap
            )
            fig_corr.update_layout(
                width=600,
                height=600,
                xaxis_title="Stocks",
                yaxis_title="Stocks",
                coloraxis_colorbar_title="Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            # Efficient Frontier Plot
            st.subheader("Efficient Frontier")
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
            st.plotly_chart(fig_frontier)
            
            # Cumulative Returns Plot
            st.subheader("Cumulative Returns (%)")
            cum_returns_df = pd.DataFrame({
                'Equal-Weighted': ((eq_cumulative - 1) * 100),
                'Optimized': ((opt_cumulative - 1) * 100)
            })
            fig_cum = px.line(cum_returns_df, title="Cumulative Returns Over Time")
            st.plotly_chart(fig_cum)
            
            # Daily Returns Distribution
            st.subheader("Daily Returns Distribution")
            returns_df = pd.DataFrame({
                'Equal-Weighted': eq_returns,
                'Optimized': opt_returns
            })
            fig_dist = px.histogram(returns_df, nbins=50, title="Distribution of Daily Returns", marginal="box",
                                    barmode='overlay')
            st.plotly_chart(fig_dist)
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")