"""
Pydantic models for the health alert and notification system.
"""

from datetime import datetime
from typing import Optional
from typing import Optional
from pydantic import BaseModel, Field


# ── Health Evaluation ──


class HealthCondition(BaseModel):
    """Result of evaluating a single sensor reading."""
    status: str = Field(..., description="healthy, warning, or bad")
    reasons: list[str] = Field(default_factory=list)
    temperature: Optional[float] = None
    bpm: Optional[float] = None
    activity_magnitude: Optional[float] = None


class EvaluationResult(BaseModel):
    """Full result of evaluating a cattle's health."""
    cid: int
    status: str
    consecutive_bad_count: int
    alert_level: Optional[str] = Field(None, description="warning or critical")
    alert_triggered: bool = False
    email_sent: bool = False
    conditions: list[HealthCondition] = Field(default_factory=list)
    message: str = ""


# ── Alert Records ──


class HealthAlertRecord(BaseModel):
    """A health alert stored in the health_alerts collection."""
    cid: int
    doctor_id: str = "default"
    doctor_email: str = ""
    doctor_name: str = ""
    status: str = Field(..., description="warning or critical")
    consecutive_count: int
    email_sent: bool = False
    health_details: dict = Field(default_factory=dict)
    graph_generated: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AlertCounterRecord(BaseModel):
    """Tracks consecutive bad readings per cattle."""
    cid: int
    consecutive_bad_count: int = 0
    last_status: str = "healthy"
    last_checked: datetime = Field(default_factory=datetime.utcnow)


# ── API Response Models ──


class AlertResponse(BaseModel):
    """Single alert returned by API."""
    cid: int
    doctor_id: Optional[str] = "default"
    doctor_email: Optional[str] = ""
    doctor_name: Optional[str] = ""
    status: str
    consecutive_count: int
    email_sent: bool
    health_details: dict
    graph_generated: bool
    timestamp: datetime


class AlertSummaryResponse(BaseModel):
    """Summary response for batch evaluation."""
    total_evaluated: int
    alerts_triggered: int
    emails_sent: int
    results: list[EvaluationResult]
