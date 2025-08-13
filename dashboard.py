import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
import hashlib
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import random

# === Load environment variables ===
load_dotenv()
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

# === MongoDB setup ===
@st.cache_resource
def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")
    return client

client = get_mongo_client()
db = client['kpi_tracker']
users = db['users']
reset_collection = db['password_resets']
daily_kpis_collection = db['daily_kpis']
realtime_kpis_collection = db['realtime_kpis']

# === Helper Functions ===
def hash_password(password):
    return hashlib.sha256(str(password).encode()).hexdigest()

def sanitize_input(value):
    import re
    if value is None:
        return ""
    cleaned_value = str(value).strip()
    return re.sub(r'[^\w.@-]', '', cleaned_value)

def authenticate_user(username, password):
    username = sanitize_input(username)
    user = users.find_one({"username": username})
    if user and user['password'] == hash_password(password):
        return user
    return None

def update_user_credentials(username, **kwargs):
    update_fields = {}
    if "new_username" in kwargs and kwargs["new_username"]:
        update_fields["username"] = sanitize_input(kwargs["new_username"])
    if "new_password" in kwargs and kwargs["new_password"]:
        update_fields["password"] = hash_password(kwargs["new_password"])
    if "new_email" in kwargs and kwargs["new_email"]:
        update_fields["email"] = sanitize_input(kwargs["new_email"])
    if update_fields:
        result = users.update_one({"username": username}, {"$set": update_fields})
        return result.modified_count > 0
    return False

def admin_add_analyst(analyst_username, analyst_password, analyst_email):
    analyst_username = sanitize_input(analyst_username)
    analyst_email = sanitize_input(analyst_email)
    analyst_password = sanitize_input(analyst_password)
    if not analyst_username or not analyst_password or not analyst_email:
        return False, "All fields are required."
    if users.find_one({"username": analyst_username}):
        return False, "Username already exists."
    users.insert_one({
        "username": analyst_username,
        "email": analyst_email,
        "password": hash_password(analyst_password),
        "role": "analyst"
    })
    return True, "Analyst added successfully."

def admin_reset_analyst_password(analyst_username, new_password):
    analyst_username = sanitize_input(analyst_username)
    if not analyst_username or not new_password:
        return False, "Username and new password are required."
    user = users.find_one({"username": analyst_username, "role": "analyst"})
    if not user:
        return False, "Analyst not found."
    users.update_one({"username": analyst_username}, {"$set": {"password": hash_password(new_password)}})
    return True, "Password reset successfully."

def admin_add_admin(admin_username, admin_password, admin_email):
    admin_username = sanitize_input(admin_username)
    admin_email = sanitize_input(admin_email)
    admin_password = sanitize_input(admin_password)
    if not admin_username or not admin_password or not admin_email:
        return False, "All fields are required."
    if users.find_one({"username": admin_username}):
        return False, "Username already exists."
    users.insert_one({
        "username": admin_username,
        "email": admin_email,
        "password": hash_password(admin_password),
        "role": "admin"
    })
    return True, "Admin added successfully."

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(email, otp):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        st.error("Email sending is not configured. Please set SENDER_EMAIL and SENDER_PASSWORD in your .env file.")
        return False
    try:
        msg = MIMEText(f"Your OTP for password reset is: {otp}\nThis OTP is valid for 10 minutes.")
        msg['Subject'] = "Password Reset OTP for KPITrac"
        msg['From'] = SENDER_EMAIL
        msg['To'] = email
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send OTP email: {e}")
        return False

def initiate_admin_password_reset(username):
    username = sanitize_input(username)
    user = users.find_one({"username": username, "role": "admin"})
    if not user or not user.get('email'):
        return False, "Admin not found or no email associated."
    otp = generate_otp()
    reset_token = hashlib.sha256((username + otp + str(datetime.now())).encode()).hexdigest()
    reset_collection.delete_many({"username": username})
    reset_collection.insert_one({
        "username": username,
        "otp": otp,
        "token": reset_token,
        "expires_at": datetime.now() + timedelta(minutes=10),
        "verified": False
    })
    if send_otp_email(user['email'], otp):
        return True, reset_token
    else:
        reset_collection.delete_one({"token": reset_token})
        return False, "Failed to send OTP email. Please check configuration."

def verify_otp(username, entered_otp):
    username = sanitize_input(username)
    reset = reset_collection.find_one({"username": username})
    if not reset:
        return False, "Invalid or expired OTP."
    if reset["expires_at"] < datetime.now():
        reset_collection.delete_one({"username": username})
        return False, "OTP expired."
    if reset["otp"] != entered_otp:
        return False, "Incorrect OTP."
    reset_collection.update_one({"username": username}, {"$set": {"verified": True}})
    return True, reset["token"]

def admin_reset_own_password(token, new_password):
    reset = reset_collection.find_one({"token": token, "verified": True})
    if not reset:
        return False, "Invalid or unverified reset request."
    if reset["expires_at"] < datetime.now():
        reset_collection.delete_one({"token": token})
        return False, "Reset request expired."
    update_user_credentials(reset["username"], new_password=new_password)
    reset_collection.delete_one({"token": token})
    return True, "Password updated successfully."

# Preprocessing for Excel
def preprocess_excel(file):
    df = pd.read_excel(file)
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    daily_kpis = df.set_index('InvoiceDate').resample('D').agg(
        InvoiceNo=('InvoiceNo', 'nunique'),
        CustomerID=('CustomerID', 'nunique'),
        TotalPrice=('TotalPrice', 'sum')
    ).rename(columns={
        'InvoiceNo': 'NumTransactions',
        'CustomerID': 'NumCustomers',
        'TotalPrice': 'Revenue'
    }).fillna(0)
    daily_kpis['AvgOrderValue'] = daily_kpis['Revenue'] / daily_kpis['NumTransactions']
    daily_kpis['AvgOrderValue'] = daily_kpis['AvgOrderValue'].replace([np.inf, -np.inf], 0).fillna(0)
    daily_kpis_collection.delete_many({})
    daily_kpis_collection.insert_many(daily_kpis.reset_index().to_dict('records'))
    df['Hour'] = df['InvoiceDate'].dt.hour
    db['transactions_hourly'].delete_many({})
    db['transactions_hourly'].insert_many(df.to_dict('records'))
    return daily_kpis

# Real-time ingestion normalization
def normalize_realtime_kpis():
    raw = list(db['raw_orders'].find())
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    daily = df.set_index('InvoiceDate').resample('D').agg({
        'Revenue': 'sum',
        'Quantity': 'sum',
        'CustomerID': 'nunique'
    }).rename(columns={'Quantity': 'NumTransactions', 'CustomerID': 'NumCustomers'}).fillna(0)
    daily['AvgOrderValue'] = daily['Revenue'] / daily['NumTransactions']
    daily['AvgOrderValue'] = daily['AvgOrderValue'].replace([np.inf, -np.inf], 0).fillna(0)
    realtime_kpis_collection.delete_many({})
    realtime_kpis_collection.insert_many(daily.reset_index().to_dict('records'))
    # For hourly analytics, store per-transaction data
    df['Hour'] = df['InvoiceDate'].dt.hour
    db['transactions_hourly_realtime'].delete_many({})
    db['transactions_hourly_realtime'].insert_many(df.to_dict('records'))
    return daily

@st.cache_data(ttl=600)
def fetch_kpis_data(collection_name='daily_kpis'):
    data = list(db[collection_name].find())
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate'])
        df = df.sort_values('InvoiceDate').reset_index(drop=True)
    return df

def fetch_hourly_transactions(selected_date, realtime=False):
    coll = db['transactions_hourly_realtime'] if realtime else db['transactions_hourly']
    docs = list(coll.find({
        "InvoiceDate": {
            "$gte": datetime.combine(selected_date, datetime.min.time()),
            "$lt": datetime.combine(selected_date, datetime.max.time()),
        }
    }))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if 'InvoiceDate' not in df.columns:
        return pd.DataFrame()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Hour'] = df['InvoiceDate'].dt.hour
    return df

# === Anomaly Detection ===
def detect_anomalies(df, revenue_column="Revenue"):
    df = df.copy()
    if df.empty or revenue_column not in df.columns:
        df['Anomaly'] = 0
        return df
    df[revenue_column] = pd.to_numeric(df[revenue_column], errors="coerce").fillna(0)
    if len(df) >= 2 and df[revenue_column].std() > 0:
        model = IsolationForest(contamination=0.15, random_state=42)
        preds = model.fit_predict(df[[revenue_column]])
        df['Anomaly'] = np.where(preds == -1, 1, 0)
    else:
        df['Anomaly'] = 0
    return df

# ==== Forecasting ==== #
def linear_regression_forecast(df, x_col, y_col, n_periods, freq="D"):
    """
    General-purpose linear regression forecast for daily/hourly/weekly/monthly.
    - For hourly: X = hour (0-23)
    - For daily/weekly/monthly: X = ordinal of date
    """
    df = df.copy().sort_values(x_col)
    df = df[[x_col, y_col]].dropna()
    df = df.reset_index(drop=True)
    results = []
    if freq == "H":
        if len(df) < 2:
            return []
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        model = LinearRegression().fit(X, y)
        future_X = np.arange(df[x_col].max() + 1, df[x_col].max() + 1 + n_periods).reshape(-1, 1)
        preds = model.predict(future_X)
        preds = np.maximum(preds, 0)  # Prevent negative predictions
        future_hours = future_X.flatten()
        return [(int(future_hours[i]), float(preds[i])) for i in range(n_periods)]
    else:
        df[x_col] = pd.to_datetime(df[x_col])
        df = df.sort_values(x_col)
        df['DateOrdinal'] = df[x_col].map(pd.Timestamp.toordinal)
        X = df[['DateOrdinal']]
        y = df[y_col]
        model = LinearRegression().fit(X, y)
        last_date = df[x_col].max()
        for i in range(1, n_periods + 1):
            if freq == "D":
                next_date = last_date + pd.Timedelta(days=i)
            elif freq == "W":
                next_date = last_date + pd.Timedelta(weeks=i)
            elif freq == "M":
                next_date = (last_date + pd.DateOffset(months=i)).replace(day=1)
            else:
                next_date = last_date + pd.Timedelta(days=i)
            pred = model.predict([[next_date.toordinal()]])[0]
            pred = max(pred, 0)  # Prevent negative predictions
            results.append((next_date, float(pred)))
        return results

def prepare_lstm_data(series, n_steps):
    scaler = MinMaxScaler()
    series = np.asarray(series).reshape(-1, 1)
    scaled = scaler.fit_transform(series)
    X, y = [], []
    for i in range(len(scaled) - n_steps):
        X.append(scaled[i:i + n_steps])
        y.append(scaled[i + n_steps])
    if not X:
        return None, None, None
    return np.array(X), np.array(y), scaler

def lstm_forecast(df, value_column, n_forecast=7, n_steps=14, freq="D", x_col=None):
    """
    LSTM forecast for various time frequencies.
    For hourly: X = hour, for other: X = date.
    """
    # Use only the value column for sequence prediction
    if freq == "H":
        df = df.copy().sort_values(x_col)
        series = df[value_column].values
        if len(series) < n_steps + 1:
            return []
        X, y, scaler = prepare_lstm_data(series, n_steps)
        if X is None or len(X) == 0:
            return []
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=100, verbose=0)
        forecast = []
        curr_seq = X[-1]
        for _ in range(n_forecast):
            pred = model.predict(curr_seq[np.newaxis, ...], verbose=0)
            forecast.append(pred[0, 0])
            curr_seq = np.vstack([curr_seq[1:], pred])
        forecast_values = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecast_values = np.maximum(forecast_values, 0)  # Prevent negative predictions
        last_hour = df[x_col].max()
        pred_hours = [int(last_hour + i + 1) for i in range(n_forecast)]
        return [(pred_hours[i], float(forecast_values[i])) for i in range(n_forecast)]
    else:
        df = df.copy().sort_values(x_col if x_col else "InvoiceDate")
        series = df[value_column].values
        if len(series) < n_steps + 1:
            return []
        X, y, scaler = prepare_lstm_data(series, n_steps)
        if X is None or len(X) == 0:
            return []
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=100, verbose=0)
        forecast = []
        curr_seq = X[-1]
        for _ in range(n_forecast):
            pred = model.predict(curr_seq[np.newaxis, ...], verbose=0)
            forecast.append(pred[0, 0])
            curr_seq = np.vstack([curr_seq[1:], pred])
        forecast_values = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        forecast_values = np.maximum(forecast_values, 0)  # Prevent negative predictions
        last_date = df[x_col if x_col else 'InvoiceDate'].max()
        preds = []
        for i, val in enumerate(forecast_values):
            if freq == "D":
                next_date = last_date + pd.Timedelta(days=i + 1)
            elif freq == "W":
                next_date = last_date + pd.Timedelta(weeks=i + 1)
            elif freq == "M":
                next_date = (last_date + pd.DateOffset(months=i + 1)).replace(day=1)
            else:
                next_date = last_date + pd.Timedelta(days=i + 1)
            preds.append((next_date, float(val)))
        return preds

# === UI Styling ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}
.kpitrac-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #8B0000;
    font-size: 60px;
    text-align: center;
    margin-top: 30px;
    margin-bottom: 0px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="kpitrac-title">KPITrac</div>', unsafe_allow_html=True)

# === Auth and Routing ===
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'

def rerun():
    st.session_state['page'] = st.session_state['page'] + ' '
    st.session_state['page'] = st.session_state['page'][:-1]

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state['user'] = user
                st.session_state['page'] = 'dashboard'
                rerun()
            else:
                st.error("❌ Invalid credentials.")
    with col2:
        if st.button("Forget Password"):
            st.session_state['page'] = 'reset'
            rerun()

def admin_panel():
    st.sidebar.header("Admin Menu")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Add Analyst", "Reset Analyst Password", "Add Admin", "Update Gmail"])
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.session_state['page'] = 'login'
        rerun()

    if page == "Dashboard":
        dashboard_page(admin=True)
    elif page == "Add Analyst":
        st.subheader("Add New Analyst")
        analyst_username = st.text_input("Analyst Username")
        analyst_email = st.text_input("Analyst Email")
        analyst_password = st.text_input("Analyst Password", type="password")
        if st.button("Create Analyst"):
            success, msg = admin_add_analyst(analyst_username, analyst_password, analyst_email)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    elif page == "Reset Analyst Password":
        st.subheader("Reset Analyst Password")
        analyst_username = st.text_input("Analyst Username for Password Reset")
        new_password = st.text_input("New Password for Analyst", type="password")
        confirm_password = st.text_input("Confirm New Password for Analyst", type="password")
        if st.button("Reset Analyst Password"):
            if new_password and new_password == confirm_password:
                success, msg = admin_reset_analyst_password(analyst_username, new_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.error("Passwords do not match or are empty.")
    elif page == "Add Admin":
        st.subheader("Add New Admin")
        admin_username = st.text_input("Admin Username")
        admin_email = st.text_input("Admin Email")
        admin_password = st.text_input("Admin Password", type="password")
        if st.button("Create Admin"):
            success, msg = admin_add_admin(admin_username, admin_password, admin_email)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    elif page == "Update Gmail":
        st.subheader("Update Your Admin Gmail")
        current_email = st.session_state['user'].get('email', '')
        new_gmail = st.text_input("Enter New Gmail", value=current_email)
        if st.button("Save Gmail"):
            if new_gmail and '@' in new_gmail and '.' in new_gmail:
                updated = update_user_credentials(st.session_state['user']['username'], new_email=new_gmail)
                if updated:
                    st.session_state['user']['email'] = new_gmail
                    st.success("Gmail updated successfully.")
                else:
                    st.error("Gmail update failed.")
            else:
                st.error("Invalid Gmail format. Please enter a valid email address.")

def analyst_panel():
    st.sidebar.header("Analyst Panel")
    page = st.sidebar.radio("Select Page", ["Dashboard"])
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.session_state['page'] = 'login'
        rerun()
    dashboard_page(admin=False)

def dashboard_page(admin=False):
    st.subheader("Dashboard")
    mode = st.radio("Select Data Source", ["Upload Dataset", "Real Time KPIs"])
    if mode == "Upload Dataset":
        uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
        if uploaded_file:
            preprocess_excel(uploaded_file)
            st.success("File processed and data saved.")
            st.cache_data.clear()
            rerun()
        kpi_collection = "daily_kpis"
        hourly_collection_realtime = False
        st.info("Showing KPIs from uploaded dataset.")
    else:
        with st.spinner("Aggregating real-time KPIs..."):
            normalize_realtime_kpis()
            st.cache_data.clear()
        kpi_collection = "realtime_kpis"
        hourly_collection_realtime = True
        st.info("Showing real-time KPIs (auto-updated by scheduled job).")

    df = fetch_kpis_data(kpi_collection)
    if df.empty:
        st.info("No data available. Please upload a dataset or wait for real-time KPIs.")
        return

    period = st.selectbox("Select View Period", ["Daily", "Weekly", "Monthly", "Yearly"])
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
    freq = freq_map[period]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    min_date = df['InvoiceDate'].min().date()
    max_date = df['InvoiceDate'].max().date()
    df_res = pd.DataFrame()
    if period == "Daily":
        selected_date = st.date_input("Select Day", min_value=min_date, max_value=max_date, value=max_date)
        df_hourly = fetch_hourly_transactions(selected_date, realtime=hourly_collection_realtime)
        if not df_hourly.empty:
            hourly_revenue = df_hourly.groupby('Hour').agg(
                Revenue=('TotalPrice', 'sum') if 'TotalPrice' in df_hourly.columns else ('Revenue', 'sum'),
                NumTransactions=('InvoiceNo', 'nunique') if 'InvoiceNo' in df_hourly.columns else ('Quantity', 'sum'),
                NumCustomers=('CustomerID', 'nunique')
            ).reset_index()
            hourly_revenue['AvgOrderValue'] = hourly_revenue['Revenue'] / hourly_revenue['NumTransactions']
            hourly_revenue = detect_anomalies(hourly_revenue, revenue_column="Revenue")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Revenue", f"${hourly_revenue['Revenue'].sum():,.2f}")
            kpi2.metric("Total Orders", int(hourly_revenue['NumTransactions'].sum()))
            kpi3.metric("Total Customers", int(hourly_revenue['NumCustomers'].sum()))
            kpi4.metric("Avg Order Value", f"${hourly_revenue['AvgOrderValue'].mean():,.2f}")

            st.write("### Hourly Revenue & Anomalies")
            fig = px.line(hourly_revenue, x='Hour', y='Revenue', markers=True, title="Hourly Revenue Trend")
            anomalies = hourly_revenue[hourly_revenue['Anomaly'] == 1]
            if not anomalies.empty:
                fig.add_scatter(
                    x=anomalies['Hour'], y=anomalies['Revenue'],
                    mode='markers', marker=dict(color='red', size=12, symbol='x'),
                    name='Anomaly'
                )
            fig.update_xaxes(title="Hour of Day (0-23)", tickmode='linear', tick0=0, dtick=1)
            fig.update_layout(xaxis=dict(range=[-0.5, 23.5]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected day.")
            return
    else:
        if period == "Weekly":
            weeks = pd.date_range(start=min_date, end=max_date, freq='W-MON')
            week_labels = [f"Week {d.isocalendar().week}, {d.year} (Starts: {d.strftime('%Y-%m-%d')})" for d in weeks]
            default_index = len(weeks) - 1 if len(weeks) > 0 else 0
            selected_week_label = st.selectbox("Select Week", week_labels, index=default_index)
            selected_week_start = weeks[week_labels.index(selected_week_label)]
            df_week = df[(df['InvoiceDate'] >= pd.Timestamp(selected_week_start)) & (df['InvoiceDate'] < pd.Timestamp(selected_week_start) + timedelta(days=7))].copy()
            df_res = df_week.set_index('InvoiceDate').resample('D').agg({
                'Revenue': 'sum',
                'NumTransactions': 'sum',
                'NumCustomers': 'sum',
                'AvgOrderValue': 'mean'
            }).fillna(0).reset_index()
        elif period == "Monthly":
            months = pd.date_range(start=min_date, end=max_date, freq='MS')
            month_labels = [d.strftime('%Y-%m') for d in months]
            default_index = len(months) - 1 if len(months) > 0 else 0
            selected_month = st.selectbox("Select Month", month_labels, index=default_index)
            selected_month_start = pd.to_datetime(selected_month + '-01')
            next_month = (selected_month_start + pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)).replace(day=1)
            df_month = df[(df['InvoiceDate'] >= selected_month_start) & (df['InvoiceDate'] < next_month)].copy()
            df_res = df_month.set_index('InvoiceDate').resample('D').agg({
                'Revenue': 'sum',
                'NumTransactions': 'sum',
                'NumCustomers': 'sum',
                'AvgOrderValue': 'mean'
            }).fillna(0).reset_index()
        elif period == "Yearly":
            years = list(range(min_date.year, max_date.year + 1))
            default_index = len(years) - 1 if len(years) > 0 else 0
            selected_year = st.selectbox("Select Year", years, index=default_index)
            df_year = df[df['InvoiceDate'].dt.year == selected_year].copy()
            df_res = df_year.set_index('InvoiceDate').resample('M').agg({
                'Revenue': 'sum',
                'NumTransactions': 'sum',
                'NumCustomers': 'sum',
                'AvgOrderValue': 'mean'
            }).fillna(0).reset_index()
        if df_res.empty:
            st.info("No data for selected period.")
            return
        df_res['AvgOrderValue'] = df_res['Revenue'] / df_res['NumTransactions']
        df_res['AvgOrderValue'] = df_res['AvgOrderValue'].replace([np.inf, -np.inf], 0).fillna(0)
        df_res = detect_anomalies(df_res, revenue_column="Revenue")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${df_res['Revenue'].sum():,.2f}")
        kpi2.metric("Total Orders", int(df_res['NumTransactions'].sum()))
        kpi3.metric("Total Customers", int(df_res['NumCustomers'].sum()))
        kpi4.metric("Avg Order Value", f"${df_res['AvgOrderValue'].mean():,.2f}")

        st.write(f"### {period} KPIs & Anomalies")
        fig = px.line(df_res, x='InvoiceDate', y='Revenue', title=f"{period} Revenue Trend", markers=True)
        anomalies = df_res[df_res['Anomaly'] == 1]
        if not anomalies.empty:
            fig.add_scatter(
                x=anomalies['InvoiceDate'], y=anomalies['Revenue'],
                mode='markers', marker=dict(color='red', size=12, symbol='x'),
                name='Anomaly'
            )
        st.plotly_chart(fig, use_container_width=True)

    # === Forecast Revenue feature: Add prediction frequency option ===
    st.subheader("Forecast Revenue")
    forecast_type = st.radio("Forecast Method", ["Linear Regression", "LSTM"], horizontal=True, key="forecast_type_"+period)
    forecast_freq = st.radio("Forecast Frequency", ["Daily", "Weekly", "Monthly"], horizontal=True, key="forecast_freq_"+period)
    freq_for_pred = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[forecast_freq]
    n_periods = st.number_input("Periods Ahead to Forecast", min_value=1, max_value=30, value=7, key="forecast_days_"+period)
    if st.button("Generate Forecast", key="forecast_btn_"+period):
        preds = []
        if period == "Daily" and not df_hourly.empty and forecast_freq == "Daily":
            # Correction: If forecast frequency is Daily, do day-wise forecast
            base_df = df[(df['InvoiceDate'].dt.date <= selected_date)].copy()
            if base_df.empty:
                st.warning("Not enough data for forecasting.")
                return
            base_df = base_df.set_index('InvoiceDate').resample('D').agg({
                'Revenue': 'sum'
            }).reset_index()
            if forecast_type == "Linear Regression":
                preds = linear_regression_forecast(base_df, "InvoiceDate", "Revenue", int(n_periods), freq="D")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            else:
                preds = lstm_forecast(base_df, "Revenue", n_forecast=int(n_periods), n_steps=min(7, len(base_df)-1), freq="D", x_col="InvoiceDate")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            st.write(f"**Forecast (Day-wise for next {n_periods} days):**")
        elif period == "Daily" and not df_hourly.empty and forecast_freq == "Weekly":
            # Weekly forecast for selected day context
            base_df = df[(df['InvoiceDate'].dt.date <= selected_date)].copy()
            if base_df.empty:
                st.warning("Not enough data for forecasting.")
                return
            base_df = base_df.set_index('InvoiceDate').resample('W').agg({
                'Revenue': 'sum'
            }).reset_index()
            if forecast_type == "Linear Regression":
                preds = linear_regression_forecast(base_df, "InvoiceDate", "Revenue", int(n_periods), freq="W")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            else:
                preds = lstm_forecast(base_df, "Revenue", n_forecast=int(n_periods), n_steps=min(7, len(base_df)-1), freq="W", x_col="InvoiceDate")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            st.write(f"**Forecast (Week-wise for next {n_periods} weeks):**")
        elif period == "Daily" and not df_hourly.empty and forecast_freq == "Monthly":
            # Monthly forecast for selected day context
            base_df = df[(df['InvoiceDate'].dt.date <= selected_date)].copy()
            if base_df.empty:
                st.warning("Not enough data for forecasting.")
                return
            base_df = base_df.set_index('InvoiceDate').resample('M').agg({
                'Revenue': 'sum'
            }).reset_index()
            if forecast_type == "Linear Regression":
                preds = linear_regression_forecast(base_df, "InvoiceDate", "Revenue", int(n_periods), freq="M")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            else:
                preds = lstm_forecast(base_df, "Revenue", n_forecast=int(n_periods), n_steps=min(7, len(base_df)-1), freq="M", x_col="InvoiceDate")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            st.write(f"**Forecast (Month-wise for next {n_periods} months):**")
        else:
            # For Weekly, Monthly, Yearly, or Daily (future days)
            base_df = df_res
            if base_df.empty:
                st.warning("Not enough data for forecasting.")
                return
            base_df = base_df.copy().sort_values("InvoiceDate")
            if forecast_type == "Linear Regression":
                preds = linear_regression_forecast(base_df, "InvoiceDate", "Revenue", int(n_periods), freq=freq_for_pred)
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            else:
                preds = lstm_forecast(base_df, "Revenue", n_forecast=int(n_periods), n_steps=min(7, len(base_df)-1), freq=freq_for_pred, x_col="InvoiceDate")
                pred_df = pd.DataFrame(preds, columns=["Date", "Predicted Revenue"])
                xaxis_label = "Date"
            st.write(f"**Forecast ({forecast_freq}):**")
        if preds:
            st.dataframe(pred_df)
            fig2 = px.line(pred_df, x=xaxis_label, y="Predicted Revenue", title="Revenue Forecast", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Not enough data for forecasting.")

def admin_password_reset_flow():
    st.subheader("Admin Password Reset")
    reset_username = st.text_input("Enter Admin Username for Reset")
    if st.button("Send OTP"):
        if reset_username:
            success, reset_token = initiate_admin_password_reset(reset_username)
            if success:
                st.session_state['reset_token'] = reset_token
                st.session_state['reset_username'] = reset_username
                st.session_state['otp_sent'] = True
                st.success("OTP sent to your admin email. Please check and enter below.")
            else:
                st.error(reset_token)
        else:
            st.error("Please enter a username.")
    if st.session_state.get('otp_sent'):
        otp = st.text_input("Enter OTP")
        if st.button("Verify OTP"):
            if otp:
                success, msg = verify_otp(st.session_state['reset_username'], otp)
                if success:
                    st.session_state['otp_verified'] = True
                    st.success("OTP verified. You can now set a new password.")
                else:
                    st.error(msg)
            else:
                st.error("Please enter the OTP.")
        if st.session_state.get('otp_verified'):
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            if new_password and confirm_password and new_password == confirm_password:
                if st.button("Set New Password"):
                    success, msg = admin_reset_own_password(st.session_state['reset_token'], new_password)
                    if success:
                        st.success("Password updated successfully. You can now log in with your new password.")
                        for key in ['reset_token', 'reset_username', 'otp_sent', 'otp_verified']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['page'] = 'login'
                        rerun()
                    else:
                        st.error(msg)
            elif new_password or confirm_password:
                st.warning("Passwords do not match or are empty.")

# === Routing ===
if st.session_state['page'] == 'login':
    login()
elif st.session_state['page'] == 'reset':
    admin_password_reset_flow()
elif 'user' in st.session_state:
    if st.session_state['user']['role'] == 'admin':
        admin_panel()
    elif st.session_state['user']['role'] == 'analyst':
        analyst_panel()
else:
    st.session_state['page'] = 'login'
    rerun()
