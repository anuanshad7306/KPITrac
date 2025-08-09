import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_anomalies(df, revenue_column="Revenue"):
    """
    Detect anomalies in a DataFrame's revenue column using Isolation Forest.
    Adds a column 'Anomaly' (1 for anomaly, 0 for normal).
    """
    df = df.copy()
    if df.empty or revenue_column not in df.columns:
        df['Anomaly'] = 0
        return df

    df[revenue_column] = pd.to_numeric(df[revenue_column], errors="coerce").fillna(0)

    if len(df) >= 2 and df[revenue_column].std() > 0:
        try:
            model = IsolationForest(contamination=0.15, random_state=42)
            preds = model.fit_predict(df[[revenue_column]])
            df['Anomaly'] = np.where(preds == -1, 1, 0)
            logger.info(f"IsolationForest: {df['Anomaly'].sum()} anomalies detected.")
        except Exception as e:
            logger.error(f"IsolationForest failed: {e}")
            mean = df[revenue_column].mean()
            std = df[revenue_column].std()
            df['Anomaly'] = ((df[revenue_column] < mean - 1.2*std) | (df[revenue_column] > mean + 1.2*std)).astype(int)
    else:
        df['Anomaly'] = 0

    return df