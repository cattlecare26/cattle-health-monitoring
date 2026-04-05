"""
Async MongoDB connection using Motor.
Handles connection lifecycle, time series collection creation, and index setup.
"""

from pymongo.errors import CollectionInvalid
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None

SENSOR_COLLECTION = "cattle_sensor_data_ts"
LOGS_COLLECTION = "logs"


async def connect_db() -> None:
    """Connect to MongoDB, create time series collections and indexes."""
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.DATABASE_NAME]

    # Create sensor time series collection if it doesn't exist
    try:
        await db.create_collection(
            SENSOR_COLLECTION,
            timeseries={
                "timeField": "timestamp_iso",
                "metaField": "cid",
                "granularity": "seconds",
            },
        )
        print(f"📊 Created time series collection: {SENSOR_COLLECTION}")
    except CollectionInvalid:
        pass

    # Create logs time series collection if it doesn't exist
    try:
        await db.create_collection(
            LOGS_COLLECTION,
            timeseries={
                "timeField": "timestamp",
                "metaField": "service",
                "granularity": "seconds",
            },
        )
        print(f"📊 Created time series collection: {LOGS_COLLECTION}")
    except CollectionInvalid:
        pass

    # Create indexes (ignore conflicts if indexes already exist)
    try:
        await db.cattle.create_index("cid", unique=True)
    except Exception:
        pass
    try:
        await db[SENSOR_COLLECTION].create_index(
            [("cid", 1), ("timestamp_iso", -1)]
        )
    except Exception:
        pass
    try:
        await db.cattle_health_events.create_index("cid")
    except Exception:
        pass
    try:
        await db.cattle_health_events.create_index([("timestamp", -1)])
    except Exception:
        pass
    try:
        await db[LOGS_COLLECTION].create_index([("service", 1), ("timestamp", -1)])
    except Exception:
        pass
    try:
        await db[LOGS_COLLECTION].create_index("cid")
    except Exception:
        pass

    # User collection indexes
    try:
        await db.users.create_index("username", unique=True)
    except Exception:
        pass
    try:
        await db.users.create_index("email", unique=True)
    except Exception:
        pass

    # ML predictions indexes
    try:
        await db.ml_predictions.create_index([("cid", 1), ("timestamp", -1)])
    except Exception:
        pass
    try:
        await db.ml_predictions.create_index("cid")
    except Exception:
        pass

    # Health alert indexes
    try:
        await db.health_alerts.create_index("cid")
    except Exception:
        pass
    try:
        await db.health_alerts.create_index([("timestamp", -1)])
    except Exception:
        pass
    try:
        await db.health_alerts.create_index([("cid", 1), ("timestamp", -1)])
    except Exception:
        pass
    try:
        await db.alert_counters.create_index("cid", unique=True)
    except Exception:
        pass

    # Cattle doctor/owner reference indexes
    try:
        await db.cattle.create_index("doctor_id")
    except Exception:
        pass
    try:
        await db.cattle.create_index("owner_id")
    except Exception:
        pass

    print(f"✅ Connected to MongoDB: {settings.DATABASE_NAME}")


async def close_db() -> None:
    """Close the MongoDB connection."""
    global client
    if client:
        client.close()
        print("🔌 MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    """Return the database instance."""
    if db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return db
