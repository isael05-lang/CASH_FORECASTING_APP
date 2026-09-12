import pandas as pd
import streamlit as st
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num

# Initialize session state to store data
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

# Function to load data from uploaded CSV
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Function to calculate historical metrics and ratios
def calculate_metrics(
    data,
    total_sales_col,
    total_expenses_col,
    cash_sale_col,
    credit_expenses_paid_col,
    date_col
):
    # Convert financial columns to numeric values
    numeric_cols = [
        total_sales_col,
        total_expenses_col,
        cash_sale_col,
        credit_expenses_paid_col,
        'Total Cash Expenses',
        'Total Credit Expenses'
    ]

    for col in numeric_cols:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Convert date column
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    # Calculate basic metrics
    data['cash_sale_percent'] = data[cash_sale_col] / data[total_sales_col]
    data['expense_ratio'] = data[total_expenses_col] / data[total_sales_col]

    # Calculate expense splits
    data['cash_expense_ratio'] = data['Total Cash Expenses'] / data[total_expenses_col]
    data['credit_expenses_paid_ratio'] = data[credit_expenses_paid_col] / data['Total Credit Expenses']

    data['credit_sales'] = data[total_sales_col] - data[cash_sale_col]
    data['credit_sales_days'] = (data['credit_sales'] / data[total_sales_col]) * 30  # Default

    # Calculate cash flow components
    data['cash_collected'] = data[cash_sale_col]  # Immediate cash sales
    data['cash_collected_credit'] = data['credit_sales'] / data['credit_sales_days']
    data['cash_flow'] = (
        data[cash_sale_col] + data['cash_collected_credit']
        - data['Total Cash Expenses']
        - (data['Total Credit Expenses'] * data['credit_expenses_paid_ratio'])
    )
    return data

# Function to forecast sales with Prophet
def forecast_sales(data, total_sales_col, date_col):
    data[date_col] = pd.to_datetime(data[date_col], format='%Y-%m')
    df = data[[total_sales_col, date_col]].rename(
        columns={date_col: 'ds', total_sales_col: 'y'}
    )
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq='MS')  # 12 months for 2025
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].rename(columns={'yhat': 'forecast_sales'})

# Function to apply assumptions and calculate cash flow
def apply_assumptions(
    forecast,
    data,
    credit_days,
    cash_percent,
    expense_ratio
):
    # Get historical averages
    avg_cash_exp_ratio = data['cash_expense_ratio'].mean()
    avg_credit_paid_ratio = data['credit_expenses_paid_ratio'].mean()

    # Calculate components from forecasted sales
    forecast['cash_sales'] = forecast['forecast_sales'] * cash_percent
    forecast['credit_sales'] = forecast['forecast_sales'] - forecast['cash_sales']
    forecast['total_expenses'] = forecast['forecast_sales'] * expense_ratio

    # Split expenses into cash/credit
    forecast['cash_expenses'] = forecast['total_expenses'] * avg_cash_exp_ratio
    forecast['credit_expenses'] = forecast['total_expenses'] - forecast['cash_expenses']

    # Calculate cash components
    forecast['cash_collected_credit'] = forecast['credit_sales'] / credit_days
    forecast['cash_paid_for_credit_expenses'] = (
        forecast['credit_expenses'] * avg_credit_paid_ratio
    )

    # Final cash flow calculation
    forecast['cash_flow'] = (
        forecast['cash_sales'] + forecast['cash_collected_credit']
        - forecast['cash_expenses']
        - forecast['cash_paid_for_credit_expenses']
    )
    return forecast

# Streamlit app layout
st.title('Cash Flow & Sales Forecasting App')

# File uploader and data loading
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
data = load_data(uploaded_file)

if data is not None:
    st.write("Data Preview:")
    st.write(data.head())
    st.write("Available Columns:")
    st.write(data.columns)

    # Column selection with correct default values
    columns = list(data.columns)

    def default_index(column_name):
        return columns.index(column_name) if column_name in columns else 0

    total_sales_col = st.selectbox(
        "Select Total Sales column",
        options=columns,
        index=default_index("Total Sales")
    )

    total_expenses_col = st.selectbox(
        "Select Total Expenses column",
        options=columns,
        index=default_index("Total Expenses")
    )

    cash_sale_col = st.selectbox(
        "Select Cash Sales column",
        options=columns,
        index=default_index("Total Cash Sales")
    )

    credit_expenses_paid_col = st.selectbox(
        "Select Credit Expenses Paid column",
        options=columns,
        index=default_index("Cash Paid for Credit Expenses")
    )

    date_col = st.selectbox(
        "Select Date column (YYYY-MM)",
        options=columns,
        index=default_index("Month")
    )
    # Validate date column
    try:
        data[date_col] = pd.to_datetime(data[date_col], format='%Y-%m', errors='coerce')
    except:
        st.error(f"Invalid date format in '{date_col}'. Ensure YYYY-MM format.")
        st.stop()
    if data[date_col].isnull().any():
        st.error(f"Some dates in '{date_col}' could not be parsed.")
        st.stop()

    # Check for null values
    if data.isnull().values.any():
        st.error("The data contains null values. Clean your CSV file.")
        st.stop()

    # Calculate historical metrics
    data = calculate_metrics(
        data,
        total_sales_col,
        total_expenses_col,
        cash_sale_col,
        credit_expenses_paid_col,
        date_col
    )
    st.write("Historical Metrics (First 5 Rows):")
    st.write(data[[date_col, total_sales_col, 'cash_flow']].head())

    # Forecast generation
    if st.button('Generate Base Forecast'):
        st.session_state.forecast = forecast_sales(data, total_sales_col, date_col)

    # Display base forecast if available
    if st.session_state.forecast is not None:
        st.write("2025 Sales Forecast:")
        st.write(st.session_state.forecast)

        # Historical vs Forecast Sales Plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(
            data[date_col].dt.to_pydatetime(),
            data[total_sales_col],
            label='Historical Sales (2023-2024)',
            marker='o'
        )
        ax1.plot(
            st.session_state.forecast['ds'].dt.to_pydatetime(),
            st.session_state.forecast['forecast_sales'],
            label='2025 Forecast',
            linestyle='--',
            color='red'
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        ax1.legend()
        st.pyplot(fig1)

        # Assumptions form
        st.header("Enter Assumptions for 2025")
        with st.form("assumptions"):
            credit_days = st.number_input(
                "Credit Sale Days (e.g., 30)",
                min_value=1,
                max_value=90,
                value=int(data['credit_sales_days'].mean())
            )
            cash_percent = st.slider(
                "Cash Sale % (of Total Sales)",
                0.0,
                100.0,
                float(data['cash_sale_percent'].mean() * 100),
                format="%.0f%%"
            )
            expense_ratio = st.slider(
                "Expense Ratio (of Total Sales)",
                0.0,
                100.0,
                float(data['expense_ratio'].mean() * 100),
                format="%.0f%%"
            )
            submitted = st.form_submit_button("Apply Assumptions")

            if submitted:
                # Convert percentages
                cash_percent /= 100.0
                expense_ratio /= 100.0

                # Apply assumptions to forecast
                adjusted_forecast = apply_assumptions(
                    st.session_state.forecast.copy(),
                    data,
                    credit_days,
                    cash_percent,
                    expense_ratio
                )

                # Display results
                st.write("Adjusted Forecast (2025):")
                st.write(adjusted_forecast[['ds', 'cash_flow']].head())

                # Plot historical vs adjusted cash flow
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(
                    data[date_col].dt.to_pydatetime(),
                    data['cash_flow'],
                    label='Historical Cash Flow (2023-2024)',
                    marker='o'
                )
                ax2.plot(
                    adjusted_forecast['ds'].dt.to_pydatetime(),
                    adjusted_forecast['cash_flow'],
                    label='2025 Forecast',
                    linestyle='--',
                    color='green'
                )
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator())
                plt.gcf().autofmt_xdate()
                ax2.legend()
                ax2.set_ylabel('Cash Flow')
                st.pyplot(fig2)
else:
    st.warning("Please upload a CSV file first.")
