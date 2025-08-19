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

def generate_fake_customer():
    """Generate a random customer profile."""
    first_names = ["John", "Jane", "Alex", "Priya", "Chen", "Maria", "Ahmed", "Sara", "Chris", "Emma", "Ravi", "Olga"]
    last_names = ["Smith", "Patel", "Wang", "Fernandez", "Kumar", "Ivanov", "Ali", "Kim", "Brown", "Singh", "Garcia", "Lee"]
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "company.com", "outlook.com"]

    first = random.choice(first_names)
    last = random.choice(last_names)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}{random.randint(10,99)}@{random.choice(domains)}"
    city = random.choice(["New York", "London", "Mumbai", "Beijing", "Sao Paulo", "Berlin", "Dubai", "Lagos"])
    return {
        "customer_id": random.randint(10000, 99999),
        "name": name,
        "email": email,
        "city": city
    }

def generate_fake_kpi():
    """
    Generates a single fake KPI record for real-time ingestion.
    Data is generated for a random datetime between July 1, 2023, and now.
    Adds time of purchase and random customer details.
    """
    start_date = datetime(2023, 7, 1)
    now = datetime.now()
    total_seconds = int((now - start_date).total_seconds())
    random_seconds = random.randint(0, total_seconds)
    invoice_datetime = start_date + timedelta(seconds=random_seconds)
    # Add random hour/minute/second for realism
    invoice_datetime = invoice_datetime.replace(
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
        microsecond=0
    )
    time_of_purchase = invoice_datetime.strftime("%H:%M:%S")
    customer = generate_fake_customer()
    return {
        "InvoiceDate": invoice_datetime,
        "TimeOfPurchase": time_of_purchase,
        "Revenue": round(random.uniform(10000, 100000), 2),
        "Quantity": random.randint(1, 50),
        "CustomerID": customer["customer_id"],
        "CustomerInfo": customer
    }

def insert_data(num_records=30):
    logger.info(f"Inserting {num_records} new records into 'raw_orders' collection (no deletion)...")
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
