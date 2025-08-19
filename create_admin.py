from pymongo import MongoClient
import hashlib

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['kpi_tracker']
users = db['users']

def hash_password(password):
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

# Insert the default admin user
# This script is meant to be run once to initialize the admin user.
# You can modify the username, email, and password here.
admin_username = "AK"
admin_email = "abhinandabhi1001.com"
admin_password = "Hello123" # IMPORTANT: Change this password after first login!

# Check if the admin user already exists to prevent duplicates
if users.find_one({"username": admin_username}) is None:
    users.insert_one({
        "username": admin_username,
        "email": admin_email,
        "password": hash_password(admin_password),
        "role": "admin"
    })
    print(f"Admin user '{admin_username}' created successfully.")
else:
    print(f"Admin user '{admin_username}' already exists. No new user created.")
