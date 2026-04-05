"""
Alert orchestration service.
Tracks consecutive bad readings, determines alert levels,
generates graphs, sends notifications, and logs alerts.
Combines rule-based threshold checks with ML behavior predictions.
"""

from datetime import datetime
from typing import Optional

from app.config import settings
from app.database import get_db, SENSOR_COLLECTION
from app.health_evaluator import evaluate_reading, evaluate_readings, determine_overall_status
from app.graph_service import fetch_graph_data, generate_health_graph
from app.email_service import send_health_alert_email, is_email_configured
from app.user_services import get_admins_by_farm_id, get_user_by_id
from app.logger import log_event
from app.ml_model import is_model_loaded, predict_from_db_docs, derive_health_status

ALERT_COUNTERS = "alert_counters"
HEALTH_ALERTS = "health_alerts"

_PROJECTION = {"_id": 0}


# ── Counter Management ──


async def get_counter(cid: int) -> dict:
    """Get or create the alert counter for a cattle."""
    db = get_db()
    counter = await db[ALERT_COUNTERS].find_one({"cid": cid}, _PROJECTION)
    if not counter:
        counter = {
            "cid": cid,
            "consecutive_bad_count": 0,
            "last_status": "healthy",
            "last_checked": datetime.utcnow(),
        }
        await db[ALERT_COUNTERS].insert_one(counter)
        counter.pop("_id", None)
    return counter


async def update_counter(cid: int, new_status: str) -> dict:
    """
    Update the consecutive bad reading counter.
    Increments on 'bad'/'warning', resets on 'healthy'.
    """
    db = get_db()
    now = datetime.utcnow()

    if new_status in ("bad", "warning"):
        result = await db[ALERT_COUNTERS].find_one_and_update(
            {"cid": cid},
            {
                "$inc": {"consecutive_bad_count": 1},
                "$set": {"last_status": new_status, "last_checked": now},
            },
            upsert=True,
            return_document=True,
        )
    else:
        # Healthy reading — reset counter
        result = await db[ALERT_COUNTERS].find_one_and_update(
            {"cid": cid},
            {"$set": {"consecutive_bad_count": 0, "last_status": "healthy", "last_checked": now}},
            upsert=True,
            return_document=True,
        )

    result.pop("_id", None)
    return result


def determine_alert_level(consecutive_count: int) -> Optional[str]:
    """
    Determine alert level based on consecutive bad reading count.
    Returns 'critical' at threshold, 'warning' below, None if healthy.
    """
    if consecutive_count >= settings.ALERT_THRESHOLD:
        return "critical"
    elif consecutive_count > 0:
        return "warning"
    return None


# ── Core Evaluation Pipeline ──


async def evaluate_cattle_health(cid: int) -> dict:
    """
    Full health evaluation pipeline for a single cattle:
    1. Fetch recent sensor readings
    2. Evaluate against thresholds (rule-based)
    3. Run ML behavior prediction
    4. Combine both signals into overall status
    5. Update consecutive counter
    6. If alert triggered → generate graph → send email → log alert
    7. Return evaluation result

    This function is idempotent — safe to call multiple times.
    """
    db = get_db()

    # 1. Fetch the latest sensor readings (last 5 for threshold evaluation)
    cursor = db[SENSOR_COLLECTION].find(
        {"cid": cid}, {"_id": 0}
    ).sort("timestamp_iso", -1).limit(5)
    recent_readings = await cursor.to_list(length=5)

    if not recent_readings:
        return {
            "cid": cid,
            "status": "no_data",
            "consecutive_bad_count": 0,
            "alert_level": None,
            "alert_triggered": False,
            "email_sent": False,
            "conditions": [],
            "ml_prediction": None,
            "message": f"No sensor data available for cattle {cid}",
        }

    # 2. Rule-based threshold evaluation
    evaluations = evaluate_readings(recent_readings)
    rule_status = determine_overall_status(evaluations)

    # 3. ML behavior prediction
    ml_behavior = None
    ml_status = None
    if is_model_loaded():
        try:
            # Fetch more data for ML (needs 10-second windows)
            ml_cursor = db[SENSOR_COLLECTION].find(
                {"cid": cid}, {"_id": 0}
            ).sort("timestamp_iso", -1).limit(150)
            ml_readings = await ml_cursor.to_list(length=150)
            ml_readings.reverse()

            ml_result = predict_from_db_docs(ml_readings, cid)
            ml_behavior = ml_result.get("prediction")

            # Get latest vitals for combined status derivation
            latest_temp = recent_readings[0].get("temperature")
            latest_bpm = recent_readings[0].get("heart", {}).get("bpm")
            ml_status = derive_health_status(
                ml_behavior, latest_temp, latest_bpm if latest_bpm else None
            )
        except Exception:
            pass  # ML should never block alert evaluation

    # 4. Combine rule-based + ML status
    overall_status = _combine_statuses(rule_status, ml_status)

    # 5. Update counter
    counter = await update_counter(cid, overall_status)
    consecutive_count = counter["consecutive_bad_count"]

    # 6. Determine alert level
    alert_level = determine_alert_level(consecutive_count)
    # Only trigger full alert (email + graph) at critical level (threshold met)
    alert_triggered = alert_level == "critical" and overall_status != "healthy"

    email_sent = False
    graph_generated = False

    if alert_triggered:
        # Get cattle info for doctor lookup
        cattle = await db.cattle.find_one({"cid": cid}, {"_id": 0})
        farm_id = cattle.get("farm_id", "") if cattle else ""
        doctor_id = cattle.get("doctor_id") if cattle else None

        # Build health summary including ML prediction
        all_reasons = []
        for e in evaluations:
            all_reasons.extend(e.get("reasons", []))
        if ml_behavior and ml_status in ("warning", "anomaly"):
            all_reasons.append(f"ML detected behavior: {ml_behavior} (status: {ml_status})")
        health_summary = "<br>".join(set(all_reasons)) if all_reasons else "Multiple abnormal readings detected."

        # Generate graph
        graph_data = await fetch_graph_data(cid)
        graph_png = generate_health_graph(cid, graph_data)
        graph_generated = True

        # Resolve email recipient: cattle's doctor → fallback to default
        recipient_email = settings.DEFAULT_DOCTOR_EMAIL
        recipient_name = "Veterinary Doctor"

        if doctor_id:
            doctor = await get_user_by_id(doctor_id)
            if doctor and doctor.get("email"):
                recipient_email = doctor["email"]
                recipient_name = doctor.get("full_name", doctor.get("username", "Doctor"))

        # Send alert email to the doctor
        sent = await send_health_alert_email(
            to_email=recipient_email,
            recipient_name=recipient_name,
            cid=cid,
            alert_status=alert_level,
            consecutive_count=consecutive_count,
            health_summary=health_summary,
            graph_png=graph_png,
        )
        if sent:
            email_sent = True

        # Log the alert
        alert_doc = {
            "cid": cid,
            "doctor_id": doctor_id or "default",
            "doctor_email": recipient_email,
            "doctor_name": recipient_name,
            "status": alert_level,
            "consecutive_count": consecutive_count,
            "email_sent": email_sent,
            "health_details": {
                "overall_status": overall_status,
                "rule_status": rule_status,
                "ml_behavior": ml_behavior,
                "ml_status": ml_status,
                "reasons": list(set(all_reasons)),
                "latest_temperature": evaluations[0].get("temperature") if evaluations else None,
                "latest_bpm": evaluations[0].get("bpm") if evaluations else None,
            },
            "graph_generated": graph_generated,
            "timestamp": datetime.utcnow(),
        }
        await db[HEALTH_ALERTS].insert_one(alert_doc)

        await log_event(
            service="alert_system",
            level="ERROR",
            action="critical_alert",
            collection=HEALTH_ALERTS,
            cid=cid,
            prediction=ml_behavior,
            prediction_status=ml_status,
            message=f"CRITICAL alert for CID {cid}: {consecutive_count} consecutive bad readings, "
                    f"email sent to {recipient_email}"
                    + (f", ML behavior: {ml_behavior}" if ml_behavior else ""),
        )

    return {
        "cid": cid,
        "status": overall_status,
        "consecutive_bad_count": consecutive_count,
        "alert_level": alert_level,
        "alert_triggered": alert_triggered,
        "email_sent": email_sent,
        "conditions": evaluations,
        "ml_prediction": {"behavior": ml_behavior, "status": ml_status} if ml_behavior else None,
        "message": _build_message(cid, overall_status, consecutive_count, alert_level, email_sent),
    }


def _combine_statuses(rule_status: str, ml_status: Optional[str]) -> str:
    """
    Combine rule-based and ML health statuses.
    Either system can escalate the overall status.

    Priority: bad/anomaly > warning > healthy/normal
    """
    if ml_status is None:
        return rule_status

    # Map ML status to rule-based equivalents
    ml_mapped = {"anomaly": "bad", "warning": "warning", "normal": "healthy", "unknown": "healthy"}
    ml_equivalent = ml_mapped.get(ml_status, "healthy")

    severity = {"healthy": 0, "warning": 1, "bad": 2}
    rule_sev = severity.get(rule_status, 0)
    ml_sev = severity.get(ml_equivalent, 0)

    # Take the worse of the two
    if ml_sev > rule_sev:
        return ml_equivalent
    return rule_status


async def evaluate_all_cattle() -> dict:
    """
    Evaluate health for all registered cattle.
    Returns summary with individual results.
    """
    db = get_db()
    cattle_list = await db.cattle.find({}, {"cid": 1, "_id": 0}).sort("cid", 1).to_list(length=1000)

    results = []
    alerts_triggered = 0
    emails_sent = 0

    for cattle in cattle_list:
        result = await evaluate_cattle_health(cattle["cid"])
        results.append(result)
        if result.get("alert_triggered"):
            alerts_triggered += 1
        if result.get("email_sent"):
            emails_sent += 1

    return {
        "total_evaluated": len(results),
        "alerts_triggered": alerts_triggered,
        "emails_sent": emails_sent,
        "results": results,
    }


# ── Alert Query Functions ──


async def get_alerts_for_cattle(cid: int, limit: int = 50) -> list[dict]:
    """Fetch alert history for a specific cattle, newest first."""
    db = get_db()
    cursor = db[HEALTH_ALERTS].find(
        {"cid": cid}, _PROJECTION
    ).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def get_recent_alerts(limit: int = 50) -> list[dict]:
    """Fetch recent alerts across all cattle, newest first."""
    db = get_db()
    cursor = db[HEALTH_ALERTS].find(
        {}, _PROJECTION
    ).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)


async def get_alert_counter(cid: int) -> dict:
    """Get the current alert counter for a cattle."""
    return await get_counter(cid)


# ── Helpers ──


def _build_message(cid: int, status: str, count: int, level: Optional[str], emailed: bool) -> str:
    if status == "healthy":
        return f"Cattle {cid} is healthy. Counter reset."
    parts = [f"Cattle {cid} status: {status} ({count} consecutive bad readings)."]
    if level:
        parts.append(f"Alert level: {level.upper()}.")
    if emailed:
        parts.append("Email notification sent.")
    elif level:
        parts.append("Email not sent (SMTP not configured or no admin found).")
    return " ".join(parts)
