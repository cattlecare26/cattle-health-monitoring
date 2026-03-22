"""
Graph generation service.
Creates time-series charts of cattle health data using matplotlib.
Returns PNG image bytes for email embedding.
"""

import io
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from app.config import settings
from app.database import get_db, SENSOR_COLLECTION


async def fetch_graph_data(cid: int, hours: int = None) -> list[dict]:
    """Fetch sensor data for the given time window."""
    if hours is None:
        hours = settings.GRAPH_TIME_WINDOW
    db = get_db()
    since = datetime.utcnow() - timedelta(hours=hours)
    cursor = db[SENSOR_COLLECTION].find(
        {"cid": cid, "timestamp_iso": {"$gte": since}},
        {"_id": 0},
    ).sort("timestamp_iso", 1)
    return await cursor.to_list(length=50000)


def generate_health_graph(cid: int, readings: list[dict]) -> bytes:
    """
    Generate a multi-panel time-series graph from sensor readings.
    Panels: Temperature, Heart Rate (BPM), Activity (accel magnitude).
    Returns PNG image as bytes.
    """
    if not readings:
        return _generate_no_data_graph(cid)

    timestamps = []
    temperatures = []
    bpms = []
    activities = []

    for r in readings:
        ts = r.get("timestamp_iso")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        timestamps.append(ts)
        temperatures.append(r.get("temperature", 0))

        heart = r.get("heart", {})
        bpms.append(heart.get("bpm", 0))

        accel = r.get("accel", {})
        ax, ay, az = accel.get("ax", 0), accel.get("ay", 0), accel.get("az", 0)
        activities.append((ax**2 + ay**2 + az**2) ** 0.5)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Cattle {cid} — Health Data (Last {settings.GRAPH_TIME_WINDOW}h)", fontsize=14, fontweight="bold")

    # Temperature
    axes[0].plot(timestamps, temperatures, color="#e74c3c", linewidth=1, alpha=0.8)
    axes[0].axhline(y=settings.TEMP_HIGH, color="red", linestyle="--", alpha=0.5, label=f"High ({settings.TEMP_HIGH}°C)")
    axes[0].axhline(y=settings.TEMP_LOW, color="blue", linestyle="--", alpha=0.5, label=f"Low ({settings.TEMP_LOW}°C)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Heart Rate
    axes[1].plot(timestamps, bpms, color="#2ecc71", linewidth=1, alpha=0.8)
    if any(b > 0 for b in bpms):
        axes[1].axhline(y=settings.BPM_HIGH, color="red", linestyle="--", alpha=0.5, label=f"High ({settings.BPM_HIGH})")
        axes[1].axhline(y=settings.BPM_LOW, color="blue", linestyle="--", alpha=0.5, label=f"Low ({settings.BPM_LOW})")
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Heart Rate (BPM)")
    axes[1].grid(True, alpha=0.3)

    # Activity
    axes[2].plot(timestamps, activities, color="#3498db", linewidth=1, alpha=0.8)
    axes[2].axhline(y=settings.ACTIVITY_LOW, color="orange", linestyle="--", alpha=0.5, label=f"Low ({settings.ACTIVITY_LOW})")
    axes[2].set_ylabel("Activity (accel mag)")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Format x-axis
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    # Export to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_no_data_graph(cid: int) -> bytes:
    """Generate a placeholder graph when no data is available."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.text(0.5, 0.5, f"No sensor data available\nfor Cattle {cid}", ha="center", va="center", fontsize=16, color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(f"Cattle {cid} — Health Data", fontsize=14, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
