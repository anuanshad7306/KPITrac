import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def load_and_preprocess(filepath=r'D:\Logic\KPI\Online Retail.xlsx'):
    # Load data
    df = pd.read_excel(filepath)
    print("=== Data Loaded ===")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print(df.head())

    # EDA: Missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    # EDA: Data types
    print("\n=== Data Types ===")
    print(df.dtypes)

    # EDA: Basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe(include='all'))

    # EDA: Unique values for categorical columns
    print("\n=== Unique Values (Sample) ===")
    for col in df.select_dtypes(include='object').columns:
        print(f"{col}: {df[col].nunique()} unique values. Sample: {df[col].unique()[:5]}")

    # Drop rows with missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    # Remove cancelled orders (InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    # Remove negative or zero quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    # Compute total price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # EDA: After cleaning
    print("\n=== After Cleaning ===")
    print(f"Shape: {df.shape}")
    print(df.head())

    # EDA: Outliers in TotalPrice
    print("\n=== Outlier Detection (TotalPrice) ===")
    q1 = df['TotalPrice'].quantile(0.25)
    q3 = df['TotalPrice'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df['TotalPrice'] < lower) | (df['TotalPrice'] > upper)]
    print(f"Number of outliers in TotalPrice: {len(outliers)}")

    # EDA: Sales over time
    print("\n=== Sales Over Time (First 5 Days) ===")
    sales_time = df.set_index('InvoiceDate').resample('D')['TotalPrice'].sum()
    print(sales_time.head())

    return df

def compute_kpis(df):
    # Grouping by InvoiceDate (daily)
    daily = df.set_index('InvoiceDate').resample('D').agg({
        'InvoiceNo': 'nunique',      # Number of transactions
        'CustomerID': 'nunique',     # Unique customers
        'TotalPrice': 'sum'          # Revenue
    }).rename(columns={'InvoiceNo': 'NumTransactions', 'CustomerID': 'NumCustomers', 'TotalPrice': 'Revenue'})
    daily['AvgOrderValue'] = daily['Revenue'] / daily['NumTransactions']
    return daily

def store_to_mongo(df, collection_name='daily_kpis'):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['kpi_tracker']
    collection = db[collection_name]
    # Clear previous data
    collection.delete_many({})
    # Insert new data
    collection.insert_many(df.reset_index().to_dict('records'))

def performance_metrics(original_df, kpi_df):
    """
    Print performance metrics comparing original and processed data.
    """
    print("=== Performance Metrics ===")
    # Revenue check
    orig_revenue = original_df['TotalPrice'].sum()
    kpi_revenue = kpi_df['Revenue'].sum()
    print(f"Original Revenue: {orig_revenue:.2f}")
    print(f"KPI Revenue:      {kpi_revenue:.2f}")
    print(f"Revenue Match:    {abs(orig_revenue - kpi_revenue) < 1e-6}")

    # Unique customers
    orig_customers = original_df['CustomerID'].nunique()
    kpi_customers = kpi_df['NumCustomers'].sum()
    print(f"Original Unique Customers: {orig_customers}")
    print(f"Sum of Daily Unique Customers: {kpi_customers}")

    # Transactions
    orig_transactions = original_df['InvoiceNo'].nunique()
    kpi_transactions = kpi_df['NumTransactions'].sum()
    print(f"Original Unique Transactions: {orig_transactions}")
    print(f"Sum of Daily Transactions:    {kpi_transactions}")

    # Avg Order Value
    orig_avg_order_value = orig_revenue / orig_transactions if orig_transactions else 0
    kpi_avg_order_value = kpi_df['AvgOrderValue'].mean()
    print(f"Original Avg Order Value: {orig_avg_order_value:.2f}")
    print(f"Mean Daily Avg Order Value: {kpi_avg_order_value:.2f}")

def regression_metrics(y_true, y_pred):
    """
    Print MAE, MSE, RMSE, and R2 score for regression predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("=== Regression Metrics ===")
    print(f"MAE:  {mae:.3f}")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

if __name__ == '__main__':
    df = load_and_preprocess()
    daily_kpis = compute_kpis(df)
    performance_metrics(df, daily_kpis)

    # --- Example: Add dummy predictions for demonstration ---
    # Let's pretend we have a model predicting daily revenue
    y_true = daily_kpis['Revenue'].values
    y_pred = y_true + np.random.normal(0, y_true.std() * 0.1, size=len(y_true))  # Dummy predictions

    regression_metrics(y_true, y_pred)
    store_to_mongo(daily_kpis)
