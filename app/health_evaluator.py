"""
Health evaluation engine.
Analyzes sensor readings against configurable thresholds to determine cattle health status.
"""

import math
from typing import Optional

from app.config import settings


def compute_activity_magnitude(accel: dict) -> float:
    """Compute the magnitude of accelerometer vector as a proxy for activity level."""
    ax = accel.get("ax", 0)
    ay = accel.get("ay", 0)
    az = accel.get("az", 0)
    return math.sqrt(ax**2 + ay**2 + az**2)


def evaluate_reading(sensor_doc: dict) -> dict:
    """
    Evaluate a single sensor reading against health thresholds.

    Returns:
        dict with keys: status ('healthy', 'warning', 'bad'), reasons (list[str]),
        temperature, bpm, activity_magnitude
    """
    reasons = []
    severity = "healthy"
    temperature = sensor_doc.get("temperature", 0.0)
    heart = sensor_doc.get("heart", {})
    accel = sensor_doc.get("accel", {})
    bpm = heart.get("bpm", 0.0)
    activity = compute_activity_magnitude(accel)

    # Temperature checks
    if temperature > settings.TEMP_HIGH:
        reasons.append(f"High temperature: {temperature:.1f}°C (threshold: {settings.TEMP_HIGH}°C)")
        severity = "bad"
    elif temperature < settings.TEMP_LOW and temperature > 0:
        reasons.append(f"Low temperature: {temperature:.1f}°C (threshold: {settings.TEMP_LOW}°C)")
        severity = "bad" if severity != "bad" else severity

    # Heart rate checks (only when BPM is reported)
    if bpm > 0:
        if bpm > settings.BPM_HIGH:
            reasons.append(f"High heart rate: {bpm:.0f} BPM (threshold: {settings.BPM_HIGH})")
            severity = "bad"
        elif bpm < settings.BPM_LOW:
            reasons.append(f"Low heart rate: {bpm:.0f} BPM (threshold: {settings.BPM_LOW})")
            severity = "bad" if severity != "bad" else severity

    # Activity check
    if activity < settings.ACTIVITY_LOW and activity > 0:
        reasons.append(f"Low activity: {activity:.0f} (threshold: {settings.ACTIVITY_LOW})")
        if severity == "healthy":
            severity = "warning"

    return {
        "status": severity,
        "reasons": reasons,
        "temperature": temperature,
        "bpm": bpm,
        "activity_magnitude": round(activity, 2),
    }


def evaluate_readings(readings: list[dict]) -> list[dict]:
    """Evaluate a batch of sensor readings. Returns list of evaluation results."""
    return [evaluate_reading(r) for r in readings]


def determine_overall_status(evaluations: list[dict]) -> str:
    """
    Determine the overall health status from multiple evaluations.
    If any reading is 'bad', overall is 'bad'.
    If any is 'warning' (and none 'bad'), overall is 'warning'.
    Otherwise 'healthy'.
    """
    if not evaluations:
        return "healthy"
    statuses = [e["status"] for e in evaluations]
    if "bad" in statuses:
        return "bad"
    if "warning" in statuses:
        return "warning"
    return "healthy"
