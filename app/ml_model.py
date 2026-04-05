"""
ML Model integration module for cattle behavior classification.

Downloads the trained RandomForest pipeline from Google Drive (if not cached
locally) and provides feature extraction and prediction functions for
real-time cattle monitoring.

Model expects 24 features per 10-second window:
  - 8 base statistical (accel mean/std per axis, SMA, temp_mean)
  - 13 advanced (skew, kurtosis, FFT dominant freq/amp per axis, temp_std)
  - 3 lag features (prev_sma, prev_temp_mean, prev_label_enc)

Predicts 7 behavior classes:
  Drinking, Grazing, Lying, Other, Ruminating, Standing, Walking
"""

import joblib
import logging
import asyncio
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats
import gdown

from app.config import settings

logger = logging.getLogger(__name__)

# MPU6050 at ±2g range: raw integer → m/s²
ACCEL_SCALE = 9.81 / 16384.0

# Window duration for feature extraction (seconds)
WINDOW_SIZE_SEC = 10.0

# Default sampling rate fallback (Hz)
DEFAULT_SAMPLING_RATE = 5.0

# ── Module-level model state (loaded once at startup) ──

_pipeline = None
_label_encoder = None
_feature_cols: Optional[list[str]] = None
_behavior_map: Optional[dict] = None
_model_loaded = False

# Per-cattle lag state tracking across requests
_cattle_lag_state: dict[int, dict] = {}


# ══════════════════════════════════════════
#  Google Drive Download
# ══════════════════════════════════════════


def _gdrive_url(file_id: str) -> str:
    """Build a Google Drive direct-download URL."""
    return f"https://drive.google.com/uc?id={file_id}"


def _ensure_model_files() -> tuple[Path, Path]:
    """
    Ensure model and label encoder files exist locally.
    Downloads from Google Drive if missing.
    Returns (model_path, label_path).
    """
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / settings.MODEL_FILENAME
    label_path = model_dir / settings.LABEL_FILENAME

    if not model_path.exists():
        url = _gdrive_url(settings.MODEL_GDRIVE_ID)
        print(f"⬇️  Downloading ML model from Google Drive …")
        logger.info("Downloading model from Google Drive: %s", url)
        try:
            gdown.download(url, str(model_path), quiet=False)
            print(f"✅ Model downloaded → {model_path}")
            logger.info("Model downloaded successfully: %s", model_path)
        except Exception as e:
            logger.error("Model download failed: %s", e)
            print(f"❌ Model download failed: {e}")

    if not label_path.exists():
        url = _gdrive_url(settings.LABEL_GDRIVE_ID)
        print(f"⬇️  Downloading label encoder from Google Drive …")
        logger.info("Downloading label encoder from Google Drive: %s", url)
        try:
            gdown.download(url, str(label_path), quiet=False)
            print(f"✅ Label encoder downloaded → {label_path}")
            logger.info("Label encoder downloaded successfully: %s", label_path)
        except Exception as e:
            logger.error("Label encoder download failed: %s", e)
            print(f"❌ Label encoder download failed: {e}")

    return model_path, label_path


# ══════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════


def load_model() -> None:
    """
    Download (if needed) and load the trained ML model.
    Expected dict keys: pipeline, label_encoder, feature_cols, behavior_map.
    Call once at application startup.
    """
    global _pipeline, _label_encoder, _feature_cols, _behavior_map, _model_loaded

    model_path, _label_path = _ensure_model_files()

    if not model_path.exists():
        logger.warning("ML model file not available after download attempt: %s", model_path)
        print(f"⚠️  ML model file not available: {model_path}")
        return

    try:
        model_dict = joblib.load(str(model_path))

        _pipeline = model_dict["pipeline"]
        _label_encoder = model_dict["label_encoder"]
        _feature_cols = model_dict["feature_cols"]
        _behavior_map = model_dict.get("behavior_map", {})
        _model_loaded = True

        classes = list(_label_encoder.classes_)
        print(f"🧠 ML model loaded: {len(_feature_cols)} features, classes={classes}")
        logger.info("ML model loaded — %d features, %d classes", len(_feature_cols), len(classes))
    except Exception as e:
        logger.error("Failed to load ML model: %s", e)
        print(f"⚠️  ML model loading failed: {e}")


def is_model_loaded() -> bool:
    """Check whether the ML model has been loaded successfully."""
    return _model_loaded


def get_behavior_classes() -> list[str]:
    """Return the list of behavior class labels the model can predict."""
    if _label_encoder is not None:
        return list(_label_encoder.classes_)
    return []


# ══════════════════════════════════════════
#  Internal Helpers
# ══════════════════════════════════════════


def _estimate_sampling_rate(timestamps_ms: list[int]) -> float:
    """Estimate sampling rate (Hz) from consecutive timestamp deltas."""
    if len(timestamps_ms) < 2:
        return DEFAULT_SAMPLING_RATE
    deltas = np.diff(timestamps_ms)
    median_delta_ms = float(np.median(deltas))
    if median_delta_ms <= 0:
        return DEFAULT_SAMPLING_RATE
    return 1000.0 / median_delta_ms


def _fft_dominant(signal: np.ndarray, sampling_rate: float) -> tuple[float, float]:
    """Compute the dominant FFT frequency and its amplitude (excluding DC)."""
    if len(signal) < 4:
        return 0.0, 0.0
    centered = signal - np.mean(signal)
    fft_vals = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
    if len(fft_vals) > 1:
        idx = int(np.argmax(fft_vals[1:])) + 1
        return float(freqs[idx]), float(fft_vals[idx])
    return 0.0, 0.0


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) > 2:
        return float(scipy_stats.skew(arr))
    return 0.0


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) > 2:
        return float(scipy_stats.kurtosis(arr))
    return 0.0


# ══════════════════════════════════════════
#  Feature Extraction
# ══════════════════════════════════════════


def _compute_window_features(
    accel_x: np.ndarray,
    accel_y: np.ndarray,
    accel_z: np.ndarray,
    temperatures: np.ndarray,
    sampling_rate: float,
    prev_sma: float,
    prev_temp_mean: float,
    prev_label_enc: float,
) -> list[float]:
    """Compute the 24 ML features for a single time window."""
    n = len(accel_x)

    # Base statistics
    ax_mean = float(np.mean(accel_x))
    ax_std = float(np.std(accel_x)) if n > 1 else 0.0
    ay_mean = float(np.mean(accel_y))
    ay_std = float(np.std(accel_y)) if n > 1 else 0.0
    az_mean = float(np.mean(accel_z))
    az_std = float(np.std(accel_z)) if n > 1 else 0.0

    # Signal Magnitude Area
    sma = float(np.mean(np.abs(accel_x) + np.abs(accel_y) + np.abs(accel_z)))

    # Temperature
    temp_mean = float(np.mean(temperatures))
    temp_std = float(np.std(temperatures)) if n > 1 else 0.0

    # Skewness and kurtosis
    ax_skew = _safe_skew(accel_x)
    ax_kurt = _safe_kurtosis(accel_x)
    ay_skew = _safe_skew(accel_y)
    ay_kurt = _safe_kurtosis(accel_y)
    az_skew = _safe_skew(accel_z)
    az_kurt = _safe_kurtosis(accel_z)

    # FFT dominant frequency and amplitude per axis
    ax_fft_freq, ax_fft_amp = _fft_dominant(accel_x, sampling_rate)
    ay_fft_freq, ay_fft_amp = _fft_dominant(accel_y, sampling_rate)
    az_fft_freq, az_fft_amp = _fft_dominant(accel_z, sampling_rate)

    # Build feature dict and return in the exact column order the model expects
    feature_dict = {
        "accel_x_mean": ax_mean,
        "accel_x_std": ax_std,
        "accel_y_mean": ay_mean,
        "accel_y_std": ay_std,
        "accel_z_mean": az_mean,
        "accel_z_std": az_std,
        "sma": sma,
        "temp_mean": temp_mean,
        "accel_x_skew": ax_skew,
        "accel_x_kurt": ax_kurt,
        "accel_x_fft_dom_freq": ax_fft_freq,
        "accel_x_fft_dom_amp": ax_fft_amp,
        "accel_y_skew": ay_skew,
        "accel_y_kurt": ay_kurt,
        "accel_y_fft_dom_freq": ay_fft_freq,
        "accel_y_fft_dom_amp": ay_fft_amp,
        "accel_z_skew": az_skew,
        "accel_z_kurt": az_kurt,
        "accel_z_fft_dom_freq": az_fft_freq,
        "accel_z_fft_dom_amp": az_fft_amp,
        "temp_std": temp_std,
        "prev_sma": prev_sma,
        "prev_temp_mean": prev_temp_mean,
        "prev_label_enc": prev_label_enc,
    }

    return [feature_dict[col] for col in _feature_cols]


def _segment_and_extract(
    accel_x: np.ndarray,
    accel_y: np.ndarray,
    accel_z: np.ndarray,
    temperatures: np.ndarray,
    timestamps_ms: list[int],
    cid: int,
) -> np.ndarray:
    """
    Segment sensor arrays into 10-second windows and extract features.
    Returns an (N, 24) feature matrix where N is the number of windows.
    """
    if len(accel_x) == 0 or not _model_loaded:
        return np.array([])

    sampling_rate = _estimate_sampling_rate(timestamps_ms)
    samples_per_window = max(1, int(WINDOW_SIZE_SEC * sampling_rate))
    total = len(accel_x)
    n_full_windows = total // samples_per_window

    # Retrieve lag state for this cattle (safe defaults for missing keys)
    lag = _cattle_lag_state.get(cid, {})
    lag.setdefault("prev_sma", np.nan)
    lag.setdefault("prev_temp_mean", np.nan)
    lag.setdefault("prev_label_enc", np.nan)

    feature_rows: list[list[float]] = []

    for i in range(n_full_windows):
        start = i * samples_per_window
        end = start + samples_per_window

        features = _compute_window_features(
            accel_x[start:end], accel_y[start:end], accel_z[start:end],
            temperatures[start:end], sampling_rate,
            lag["prev_sma"], lag["prev_temp_mean"], lag["prev_label_enc"],
        )
        feature_rows.append(features)

        # Update lag for next window
        w_ax, w_ay, w_az = accel_x[start:end], accel_y[start:end], accel_z[start:end]
        lag["prev_sma"] = float(np.mean(np.abs(w_ax) + np.abs(w_ay) + np.abs(w_az)))
        lag["prev_temp_mean"] = float(np.mean(temperatures[start:end]))

    # Handle remaining samples as a partial window (need ≥3 for stats)
    remaining_start = n_full_windows * samples_per_window
    if remaining_start < total and (total - remaining_start) >= 3:
        features = _compute_window_features(
            accel_x[remaining_start:], accel_y[remaining_start:], accel_z[remaining_start:],
            temperatures[remaining_start:], sampling_rate,
            lag["prev_sma"], lag["prev_temp_mean"], lag["prev_label_enc"],
        )
        feature_rows.append(features)

        w_ax = accel_x[remaining_start:]
        w_ay = accel_y[remaining_start:]
        w_az = accel_z[remaining_start:]
        lag["prev_sma"] = float(np.mean(np.abs(w_ax) + np.abs(w_ay) + np.abs(w_az)))
        lag["prev_temp_mean"] = float(np.mean(temperatures[remaining_start:]))

    if not feature_rows:
        return np.array([])

    return np.array(feature_rows)


def _rows_to_arrays(rows: list[dict], key_ax: str, key_ay: str, key_az: str, key_temp: str) -> tuple:
    """Convert list of row dicts to numpy arrays with accel conversion to m/s²."""
    ax = np.array([r[key_ax] for r in rows], dtype=np.float64) * ACCEL_SCALE
    ay = np.array([r[key_ay] for r in rows], dtype=np.float64) * ACCEL_SCALE
    az = np.array([r[key_az] for r in rows], dtype=np.float64) * ACCEL_SCALE
    temps = np.array([r[key_temp] for r in rows], dtype=np.float64)
    ts_ms = [r.get("timestamp_ms", 0) for r in rows]
    return ax, ay, az, temps, ts_ms


# ══════════════════════════════════════════
#  Public Prediction API
# ══════════════════════════════════════════


def predict_from_raw_rows(rows: list[dict], cid: int) -> dict:
    """
    Predict cattle behavior from raw ESP32 sensor rows (flat format).

    rows: list of dicts with keys ax, ay, az, temp_c, timestamp_ms
    Returns: { prediction, window_predictions, window_count }
    """
    if not _model_loaded:
        return {"prediction": "model_not_loaded", "window_predictions": [], "window_count": 0}
    if not rows:
        return {"prediction": "insufficient_data", "window_predictions": [], "window_count": 0}

    sorted_rows = sorted(rows, key=lambda r: r.get("timestamp_ms", 0))
    ax, ay, az, temps, ts_ms = _rows_to_arrays(sorted_rows, "ax", "ay", "az", "temp_c")
    feature_matrix = _segment_and_extract(ax, ay, az, temps, ts_ms, cid)

    return _predict_from_matrix(feature_matrix, cid)


def predict_from_db_docs(docs: list[dict], cid: int) -> dict:
    """
    Predict cattle behavior from structured MongoDB sensor documents.

    docs: list of dicts with nested accel:{ax,ay,az}, temperature, timestamp_ms
    Returns: { prediction, window_predictions, window_count }
    """
    if not _model_loaded:
        return {"prediction": "model_not_loaded", "window_predictions": [], "window_count": 0}
    if not docs:
        return {"prediction": "insufficient_data", "window_predictions": [], "window_count": 0}

    # Flatten DB format to arrays
    flat_rows = []
    for doc in docs:
        accel = doc.get("accel", {})
        flat_rows.append({
            "ax": accel.get("ax", 0),
            "ay": accel.get("ay", 0),
            "az": accel.get("az", 0),
            "temp_c": doc.get("temperature", 0.0),
            "timestamp_ms": doc.get("timestamp_ms", 0),
        })
    sorted_rows = sorted(flat_rows, key=lambda r: r.get("timestamp_ms", 0))
    ax, ay, az, temps, ts_ms = _rows_to_arrays(sorted_rows, "ax", "ay", "az", "temp_c")
    feature_matrix = _segment_and_extract(ax, ay, az, temps, ts_ms, cid)

    return _predict_from_matrix(feature_matrix, cid)


def _predict_from_matrix(feature_matrix: np.ndarray, cid: int) -> dict:
    """Run prediction on a feature matrix and return result dict."""
    if feature_matrix.size == 0:
        return {"prediction": "insufficient_data", "window_predictions": [], "window_count": 0}

    try:
        predictions = _pipeline.predict(feature_matrix)
        labels = list(_label_encoder.inverse_transform(predictions))

        # Update lag state with the last prediction's encoding
        if cid and len(predictions) > 0:
            _cattle_lag_state.setdefault(cid, {})
            _cattle_lag_state[cid]["prev_label_enc"] = float(predictions[-1])

        # Most common behavior across all windows
        counter = Counter(labels)
        prediction = counter.most_common(1)[0][0]

        return {
            "prediction": prediction,
            "window_predictions": labels,
            "window_count": len(labels),
        }
    except Exception as e:
        logger.error("ML prediction failed: %s", e)
        return {"prediction": "error", "window_predictions": [], "window_count": 0}


def derive_health_status(
    behavior: str,
    temperature: Optional[float] = None,
    bpm: Optional[float] = None,
) -> str:
    """
    Derive a health status by combining ML behavior prediction
    with sensor threshold checks.

    Returns: 'normal', 'warning', or 'anomaly'
    """
    # Sensor threshold checks (aligned with config thresholds)
    if temperature is not None:
        if temperature > 39.5 or temperature < 35.0:
            return "anomaly"
    if bpm is not None and bpm > 0:
        if bpm > 100.0 or bpm < 30.0:
            return "anomaly"

    # Behavior-based status
    if behavior in ("error", "model_not_loaded", "insufficient_data"):
        return "unknown"
    if behavior == "Other":
        return "warning"

    return "normal"


# ── Async wrappers (run CPU-bound work in thread pool) ──


async def predict_from_raw_rows_async(rows: list[dict], cid: int) -> dict:
    """Async wrapper for predict_from_raw_rows."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict_from_raw_rows, rows, cid)


async def predict_from_db_docs_async(docs: list[dict], cid: int) -> dict:
    """Async wrapper for predict_from_db_docs."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict_from_db_docs, docs, cid)
