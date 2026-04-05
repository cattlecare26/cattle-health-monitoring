"""
Pydantic models for request validation and database schema.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Request Models (raw ESP32 input) ──

class SensorRow(BaseModel):
    """A single raw sensor reading from the ESP32 device."""
    timestamp_iso: str
    timestamp_ms: int
    temp_c: float
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    signal: int
    peak: int
    down: int
    bpm: float

    @field_validator("timestamp_iso")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp: {v}")
        return v


class SensorBulkRequest(BaseModel):
    """Bulk sensor data request from an ESP32 device."""
    cid: int = Field(..., gt=0, description="Cattle ID")
    data: list[SensorRow] = Field(..., min_length=1, max_length=5000)


# ── Database Schema Models (structured MongoDB documents) ──

class AccelData(BaseModel):
    ax: int
    ay: int
    az: int


class GyroData(BaseModel):
    gx: int
    gy: int
    gz: int


class HeartData(BaseModel):
    signal: int
    peak: int
    down: int
    bpm: float


class SensorDocument(BaseModel):
    """Structured sensor document stored in MongoDB."""
    cid: int
    timestamp_iso: datetime
    timestamp_ms: int
    temperature: float
    accel: AccelData
    gyro: GyroData
    heart: HeartData
    created_at: datetime


# ── Cattle Request Models ──

class CattleCreate(BaseModel):
    """Request body for creating a new cattle."""
    cid: int = Field(..., gt=0, description="Cattle ID")
    name: str = Field(..., min_length=1)
    farm_id: str = Field(..., min_length=1)
    breed: str = Field(..., min_length=1)
    age: int = Field(..., ge=0)
    doctor_id: Optional[str] = Field(default=None, description="Username of the assigned veterinary doctor")
    owner_id: Optional[str] = Field(default=None, description="Username of the farmer/owner")
    status: str = "active"


class CattleUpdate(BaseModel):
    """Request body for updating cattle. All fields optional."""
    name: Optional[str] = None
    farm_id: Optional[str] = None
    breed: Optional[str] = None
    age: Optional[int] = Field(default=None, ge=0)
    doctor_id: Optional[str] = Field(default=None, description="Username of the assigned veterinary doctor")
    owner_id: Optional[str] = Field(default=None, description="Username of the farmer/owner")
    status: Optional[str] = None


# ── Health Event Model ──

class HealthEventModel(BaseModel):
    cid: int
    event: str
    value: float
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── ML Prediction Models ──

class PredictionDetail(BaseModel):
    """ML model prediction details included in API responses."""
    behavior: str = Field(..., description="Predicted behavior (e.g., Grazing, Lying, Walking)")
    status: str = Field(..., description="Health status derived from prediction (normal/warning/anomaly)")
    window_count: int = Field(0, description="Number of 10-second windows analyzed")
    window_predictions: list[str] = Field(default_factory=list, description="Per-window behavior predictions")


class CattleStatusResponse(BaseModel):
    """Real-time cattle health status from ML model."""
    cid: int
    behavior: str = Field(..., description="Current predicted behavior")
    status: str = Field(..., description="Health status: normal, warning, or anomaly")
    temperature: Optional[float] = None
    bpm: Optional[float] = None
    timestamp: Optional[datetime] = None


# ── Response Models ──

class BulkInsertResponse(BaseModel):
    success: bool
    cid: int
    inserted_count: int
    message: str
    prediction: Optional[PredictionDetail] = Field(None, description="ML behavior prediction")


class CattleCreateResponse(BaseModel):
    success: bool
    message: str
    cid: int


class CattleUpdateResponse(BaseModel):
    success: bool
    message: str
    cid: int


class SensorReadingResponse(BaseModel):
    """A single structured sensor reading returned by GET APIs."""
    cid: int
    timestamp_iso: datetime
    timestamp_ms: int
    temperature: float
    accel: AccelData
    gyro: GyroData
    heart: HeartData
    created_at: Optional[datetime] = None


class CattleResponse(BaseModel):
    """Full cattle metadata including created_at."""
    cid: int
    name: str
    farm_id: str
    breed: str
    age: int
    status: str
    doctor_id: Optional[str] = None
    owner_id: Optional[str] = None
    created_at: Optional[datetime] = None


class CattleLatestStatus(BaseModel):
    """Latest sensor snapshot for a single cattle (used in dashboard)."""
    cid: int
    timestamp_iso: datetime
    temperature: float
    accel: AccelData
    gyro: GyroData
    heart: HeartData


class HealthEventResponse(BaseModel):
    """Health event returned by GET APIs."""
    cid: int
    event: str
    value: float
    status: str
    timestamp: datetime
