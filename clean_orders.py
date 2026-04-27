
import os
from pymongo import MongoClient
try:
    import mongomock  # type: ignore
except Exception:
    mongomock = None

def clean_data():
    mongo_url = os.environ.get("MONGO_URL", "").strip()
    if mongo_url:
        client = MongoClient(mongo_url)
    else:
        client = mongomock.MongoClient() if mongomock is not None else MongoClient("mongodb://localhost:27017")
    db = client["tejas_kitchen"]
    db["orders"].delete_many({})
    db["counters"].delete_one({"_id": "orders"})
    print("All orders and related data cleaned.")

if __name__ == "__main__":
    clean_data()
