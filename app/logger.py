"""
Async logging module.
Inserts structured log documents into the `logs` Time Series collection.
"""

from datetime import datetime
from typing import Optional

from app.database import get_db, LOGS_COLLECTION


async def log_event(
    service: str,
    level: str,
    action: str,
    collection: str,
    message: str,
    cid: Optional[int] = None,
    reference_id: Optional[str] = None,
    records_count: Optional[int] = None,
    prediction: Optional[str] = None,
    prediction_status: Optional[str] = None,
) -> None:
    """Insert a structured log document into the logs collection."""
    db = get_db()
    doc: dict = {
        "timestamp": datetime.utcnow(),
        "service": service,
        "level": level,
        "action": action,
        "collection": collection,
        "message": message,
    }
    if cid is not None:
        doc["cid"] = cid
    if reference_id is not None:
        doc["reference_id"] = reference_id
    if records_count is not None:
        doc["records_count"] = records_count
    if prediction is not None:
        doc["prediction"] = prediction
    if prediction_status is not None:
        doc["prediction_status"] = prediction_status

    try:
        await db[LOGS_COLLECTION].insert_one(doc)
    except Exception:
        # Logging should never crash the application
        pass
