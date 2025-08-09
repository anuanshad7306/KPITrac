def get_recommendations(df_kpis, df_anomalies):
    """
    Generate business recommendations based on KPIs and anomalies.
    """
    suggestions = []
    if df_kpis.empty:
        suggestions.append("No KPI data available to generate recommendations.")
        return suggestions

    df_kpis = df_kpis.sort_values('InvoiceDate')
    if len(df_kpis) >= 8:
        rolling_avg = df_kpis['Revenue'].rolling(window=7, min_periods=1).mean()
        if rolling_avg.iloc[-1] < rolling_avg.iloc[-8]:
            suggestions.append("Revenue is trending down. Consider launching new promotions.")
        elif rolling_avg.iloc[-1] > rolling_avg.iloc[-8] * 1.1:
            suggestions.append("Revenue is improving! Analyze what worked.")

    if not df_anomalies.empty and df_anomalies['Anomaly'].iloc[-1] == 1:
        suggestions.append("Recent anomaly detected. Check for data or operational issues.")

    if 'AvgOrderValue' in df_kpis.columns and len(df_kpis) >= 2:
        if df_kpis['AvgOrderValue'].iloc[-1] < df_kpis['AvgOrderValue'].rolling(window=7, min_periods=1).mean().iloc[-1]:
            suggestions.append("Average order value is dropping. Consider incentives for higher spend.")

    if not suggestions:
        suggestions.append("All KPIs look normal. Keep monitoring.")
    return suggestions