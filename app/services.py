"""
Service layer: data transformation and database operations.
Converts raw ESP32 flat data into structured MongoDB documents.
Provides query functions for data retrieval APIs.
"""

from datetime import datetime, timedelta
from typing import Optional

from app.models import SensorRow, SensorDocument, AccelData, GyroData, HeartData, CattleCreate, CattleUpdate
from app.database import get_db, SENSOR_COLLECTION
from app.logger import log_event

# Fields to exclude from query results
_PROJECTION = {"_id": 0}


def transform_sensor_row(cid: int, row: SensorRow) -> dict:
    """
    Transform a raw flat sensor row into a structured MongoDB document.

    Raw format:  { temp_c, ax, ay, az, gx, gy, gz, signal, peak, down, bpm }
    DB format:   { temperature, accel: {ax,ay,az}, gyro: {gx,gy,gz}, heart: {signal,peak,down,bpm} }
    """
    return {
        "cid": cid,
        "timestamp_iso": datetime.fromisoformat(row.timestamp_iso),
        "timestamp_ms": row.timestamp_ms,
        "temperature": row.temp_c,
        "accel": {
            "ax": row.ax,
            "ay": row.ay,
            "az": row.az,
        },
        "gyro": {
            "gx": row.gx,
            "gy": row.gy,
            "gz": row.gz,
        },
        "heart": {
            "signal": row.signal,
            "peak": row.peak,
            "down": row.down,
            "bpm": row.bpm,
        },
        "created_at": datetime.utcnow(),
    }


def transform_sensor_rows(cid: int, rows: list[SensorRow]) -> list[dict]:
    """Transform a batch of raw sensor rows into structured documents."""
    return [transform_sensor_row(cid, row) for row in rows]


async def bulk_insert_sensor_data(cid: int, rows: list[SensorRow]) -> int:
    """
    Validate cattle exists, then transform and insert sensor data.
    Automatically triggers health evaluation after ingestion.
    Returns the number of documents inserted.
    Raises ValueError if cattle is not registered.
    """
    db = get_db()

    # Validate cattle exists before inserting sensor data
    cattle = await db.cattle.find_one({"cid": cid})
    if not cattle:
        await log_event(
            service="sensor_api", level="WARNING", action="invalid_cattle_id",
            collection=SENSOR_COLLECTION, cid=cid,
            message=f"Sensor data rejected — cattle with CID {cid} is not registered",
        )
        raise ValueError(f"Cattle with CID {cid} is not registered")

    documents = transform_sensor_rows(cid, rows)
    result = await db[SENSOR_COLLECTION].insert_many(documents)
    inserted = len(result.inserted_ids)

    await log_event(
        service="sensor_api", level="INFO", action="bulk_insert",
        collection=SENSOR_COLLECTION, cid=cid, records_count=inserted,
        message=f"Sensor batch inserted successfully for CID {cid}",
    )

    # Trigger automatic health evaluation after sensor ingestion
    try:
        from app.alert_services import evaluate_cattle_health
        await evaluate_cattle_health(cid)
    except Exception:
        pass  # Alert evaluation should never block sensor ingestion

    return inserted


# ── Data Retrieval Services ──


async def get_cattle_metadata(cid: int) -> Optional[dict]:
    """Fetch cattle metadata from the cattle collection."""
    db = get_db()
    return await db.cattle.find_one({"cid": cid}, _PROJECTION)


async def get_latest_sensor_data(cid: int) -> Optional[dict]:
    """Fetch the most recent sensor record for a cattle."""
    db = get_db()
    cursor = db[SENSOR_COLLECTION].find(
        {"cid": cid}, _PROJECTION
    ).sort("timestamp_iso", -1).limit(1)
    docs = await cursor.to_list(length=1)
    return docs[0] if docs else None


async def get_recent_records(cid: int, limit: int = 100) -> list[dict]:
    """Fetch the last N sensor records for a cattle, newest first."""
    db = get_db()
    cursor = db[SENSOR_COLLECTION].find(
        {"cid": cid}, _PROJECTION
    ).sort("timestamp_iso", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def get_last_hour_data(cid: int) -> list[dict]:
    """Fetch all sensor readings from the last 1 hour for a cattle."""
    db = get_db()
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    cursor = db[SENSOR_COLLECTION].find(
        {"cid": cid, "timestamp_iso": {"$gte": one_hour_ago}},
        _PROJECTION,
    ).sort("timestamp_iso", 1)
    return await cursor.to_list(length=5000)


async def get_range_data(cid: int, start: datetime, end: datetime) -> list[dict]:
    """Fetch sensor readings within a time range for a cattle."""
    db = get_db()
    cursor = db[SENSOR_COLLECTION].find(
        {
            "cid": cid,
            "timestamp_iso": {"$gte": start, "$lte": end},
        },
        _PROJECTION,
    ).sort("timestamp_iso", 1)
    return await cursor.to_list(length=10000)


async def get_all_cattle_latest() -> list[dict]:
    """Fetch the latest sensor reading for every cattle (dashboard view)."""
    db = get_db()
    pipeline = [
        {"$sort": {"timestamp_iso": -1}},
        {
            "$group": {
                "_id": "$cid",
                "cid": {"$first": "$cid"},
                "timestamp_iso": {"$first": "$timestamp_iso"},
                "temperature": {"$first": "$temperature"},
                "accel": {"$first": "$accel"},
                "gyro": {"$first": "$gyro"},
                "heart": {"$first": "$heart"},
            }
        },
        {"$project": {"_id": 0}},
        {"$sort": {"cid": 1}},
    ]
    return await db[SENSOR_COLLECTION].aggregate(pipeline).to_list(length=1000)


# ── Cattle Management Services ──


async def create_cattle(data: CattleCreate) -> dict:
    """Create a new cattle record. Raises ValueError if cid already exists."""
    db = get_db()
    existing = await db.cattle.find_one({"cid": data.cid})
    if existing:
        raise ValueError(f"Cattle with cid {data.cid} already exists")
    doc = data.model_dump()
    doc["created_at"] = datetime.utcnow()
    await db.cattle.insert_one(doc)

    await log_event(
        service="cattle_api", level="INFO", action="create_cattle",
        collection="cattle", cid=data.cid,
        message=f"New cattle registered: CID {data.cid}",
    )
    return doc


async def update_cattle(cid: int, data: CattleUpdate) -> Optional[dict]:
    """Update cattle metadata. Returns updated document or None if not found."""
    db = get_db()
    update_fields = {k: v for k, v in data.model_dump().items() if v is not None}
    if not update_fields:
        return await db.cattle.find_one({"cid": cid}, _PROJECTION)
    result = await db.cattle.update_one({"cid": cid}, {"$set": update_fields})
    if result.matched_count == 0:
        return None

    await log_event(
        service="cattle_api", level="INFO", action="update_cattle",
        collection="cattle", cid=cid,
        message=f"Cattle updated: CID {cid}, fields: {list(update_fields.keys())}",
    )
    return await db.cattle.find_one({"cid": cid}, _PROJECTION)


async def get_all_cattle() -> list[dict]:
    """Fetch all registered cattle."""
    db = get_db()
    cursor = db.cattle.find({}, _PROJECTION).sort("cid", 1)
    return await cursor.to_list(length=1000)


# ── Health Event Services ──


async def get_cattle_health_events(cid: int, limit: int = 50) -> list[dict]:
    """Fetch health events for a specific cattle, newest first."""
    db = get_db()
    cursor = db.cattle_health_events.find(
        {"cid": cid}, _PROJECTION
    ).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def get_recent_health_events(limit: int = 50) -> list[dict]:
    """Fetch recent health events across all cattle, newest first."""
    db = get_db()
    cursor = db.cattle_health_events.find(
        {}, _PROJECTION
    ).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)
