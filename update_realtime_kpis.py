import logging
from realtime_ingest import insert_data
from dashboard import normalize_realtime_kpis

# Optional logging
logging.basicConfig(filename="update_realtime_kpis.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Step 1: Insert fake real-time KPI data
logging.info("=== Starting Real-time KPI Update ===")
insert_data(num_records=30)
logging.info("Inserted new KPI records into raw_orders.")

# Step 2: Normalize into realtime_kpis for dashboard
normalize_realtime_kpis()
logging.info("Normalized real-time KPIs and updated MongoDB.")

print("✅ Real-time KPI update completed successfully.")
