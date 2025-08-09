import logging
import random
from datetime import datetime, timedelta
from pymongo import MongoClient

# Setup logging for this script
logging.basicConfig(filename="realtime_ingest.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["kpi_tracker"]
collection = db["raw_orders"]

def generate_fake_kpi():
    """
    Generates a single fake KPI record for real-time ingestion.
    Data is generated for a random datetime between July 1, 2023, and now.
    """
    start_date = datetime(2023, 7, 1)
    now = datetime.now()
    total_seconds = int((now - start_date).total_seconds())
    random_seconds = random.randint(0, total_seconds)
    invoice_datetime = start_date + timedelta(seconds=random_seconds)
    return {
        "InvoiceDate": invoice_datetime,  # This now includes both date and time
        "Revenue": round(random.uniform(10000, 100000), 2),
        "Quantity": random.randint(1, 50),
        "CustomerID": random.randint(10000, 99999)
    }

def insert_data(num_records=30):
    logger.info(f"Clearing existing data from 'raw_orders' collection...")
    collection.delete_many({})
    logger.info(f"Inserting {num_records} new records into 'raw_orders' collection...")
    for _ in range(num_records):
        doc = generate_fake_kpi()
        try:
            collection.insert_one(doc)
            logger.info(f"Inserted: {doc}")
        except Exception as e:
            logger.error(f"Error inserting document {doc}: {e}")
    logger.info(f"Finished inserting {num_records} records.")

if __name__ == "__main__":
    insert_data(num_records=30)