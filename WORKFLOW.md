# Cattle Health Monitoring System — Complete Workflow

End-to-end intelligent monitoring pipeline: from hardware sensors to ML-powered health alerts.

---

## System Overview

```
┌────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  ESP32      │────▶│  FastAPI      │────▶│  MongoDB     │────▶│  ML Engine   │────▶│  Email   │
│  Hardware   │     │  Backend     │     │  Database    │     │  + Rules     │     │  Alert   │
└────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
      │                   │                    │                     │                   │
  Sensors            Validate &            Store Time           Predict &           Notify
  Collect            Transform             Series Data          Evaluate            Admin
```

---

## 1. Application Startup

When the server starts (`uvicorn app.main:app`), the lifespan handler runs:

```
app/main.py → lifespan()
    │
    ├── 1. connect_db()                    [app/database.py]
    │       ├── Connect to MongoDB Atlas
    │       ├── Create time-series collections (cattle_sensor_data_ts, logs)
    │       └── Create all indexes (cattle, sensors, users, alerts, ml_predictions, logs)
    │
    ├── 2. load_model()                    [app/ml_model.py]
    │       ├── Load cattle_model_v4.pkl via joblib (259 MB)
    │       │     ├── pipeline    → SimpleImputer + SMOTE + RandomForest (500 trees)
    │       │     ├── label_encoder → ['Drinking','Grazing','Lying','Other','Ruminating','Standing','Walking']
    │       │     ├── feature_cols → 24 feature names
    │       │     └── behavior_map → {0:'Grazing', 1:'Walking', 2:'Standing', ...}
    │       └── Model stays in memory — reused for all requests (no reload)
    │
    └── 3. Register routers
            ├── auth_router      → /api/v1/auth/*
            ├── sensor_router    → /api/v1/cattle/sensor/*, /api/v1/cattle/latest, /api/v1/cattle/{cid}/*
            ├── health_router    → /api/v1/health-events/*
            ├── cattle_router    → /api/v1/cattle (CRUD)
            └── alert_router     → /api/v1/alerts/*
```

**Startup Output:**
```
📊 Created time series collection: cattle_sensor_data_ts
📊 Created time series collection: logs
✅ Connected to MongoDB: CDataBase
🧠 ML model loaded: 24 features, classes=['Drinking', 'Grazing', 'Lying', 'Other', 'Ruminating', 'Standing', 'Walking']
```

---

## 2. Authentication Flow

All endpoints require one of two auth methods:

```
┌─────────────────────────────────────────────────────┐
│                   REQUEST ARRIVES                     │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  Check auth headers      │     [app/auth.py]
          │  (get_current_user)      │
          └────────────┬────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼                           ▼
  ┌──────────────┐          ┌──────────────┐
  │ Authorization │          │  X-API-Key   │
  │ Bearer <JWT>  │          │  header      │
  └──────┬───────┘          └──────┬───────┘
         │                         │
         ▼                         ▼
  ┌──────────────┐          ┌──────────────┐
  │ Decode JWT   │          │ Compare with │
  │ Verify exp   │          │ API_SECRET   │
  │ Extract user │          │ _KEY config  │
  └──────┬───────┘          └──────┬───────┘
         │                         │
         ▼                         ▼
  ┌──────────────┐          ┌──────────────┐
  │ User context │          │ No user ctx  │
  │ username     │          │ (device auth)│
  │ role (admin/ │          │              │
  │   user)      │          │              │
  │ farm_ids     │          │              │
  └──────┬───────┘          └──────┬───────┘
         └──────────┬──────────────┘
                    ▼
           ┌────────────────┐
           │  RBAC Check     │
           │  admin → CRUD   │
           │  user  → Read   │
           │  Farm scoped    │
           └────────────────┘
```

### JWT Login Flow
```
POST /api/v1/auth/login
  → Verify username + bcrypt password       [app/user_services.py]
  → Generate JWT (HS256, 60 min expiry)
  → Return { access_token, user profile }
```

---

## 3. Cattle Registration

Before sensor data can be ingested, the cattle must be registered:

```
POST /api/v1/cattle                         [app/routes.py → cattle_router]
  │
  ├── Auth: Admin JWT or API Key required
  ├── Farm access check (admin must belong to that farm)
  │
  ├── Validate: { cid, name, farm_id, breed, age, status }     [app/models.py → CattleCreate]
  │
  ├── Check: cid must not already exist
  │
  ├── Store in MongoDB → cattle collection      [app/services.py → create_cattle()]
  │     { cid: 1, name: "Cow-01", farm_id: "Farm-A", breed: "HF", age: 4, status: "active" }
  │
  ├── Log: action="create_cattle"               [app/logger.py → logs collection]
  │
  └── Response: { success: true, message: "Cattle created", cid: 1 }
```

---

## 4. Sensor Data Ingestion (Core Pipeline)

This is the **main workflow** — triggered every time the ESP32 sends data:

```
POST /api/v1/cattle/sensor/bulk             [app/routes.py → sensor_router]
  │
  │  Request Body:
  │  {
  │    "cid": 1,
  │    "data": [
  │      { "timestamp_iso": "2026-02-04T23:31:48.120", "timestamp_ms": 2122,
  │        "temp_c": 38.47, "ax": -3384, "ay": -8684, "az": 13392,
  │        "gx": -145, "gy": 84, "gz": -12,
  │        "signal": 1818, "peak": 1, "down": 0, "bpm": 72 },
  │      ... (up to 5000 rows)
  │    ]
  │  }
  │
  ▼
```

### Step 4.1 — Validation

```
  ├── Auth: Admin JWT or API Key
  │
  ├── Pydantic validation                      [app/models.py → SensorBulkRequest]
  │     ├── cid > 0
  │     ├── data: 1–5000 SensorRow items
  │     └── Each SensorRow: validate timestamp_iso is valid ISO format
  │
  ├── Cattle existence check                   [app/services.py → bulk_insert_sensor_data()]
  │     └── db.cattle.find_one({cid: 1})
  │         ├── EXISTS → continue
  │         └── NOT FOUND → 404 + log "invalid_cattle_id" → STOP
```

### Step 4.2 — Transform

```
  ├── Transform raw ESP32 data → structured MongoDB format     [app/services.py → transform_sensor_row()]
  │
  │     RAW (flat from ESP32)              STRUCTURED (for MongoDB)
  │     ─────────────────────              ────────────────────────
  │     temp_c: 38.47            ──▶       temperature: 38.47
  │     ax: -3384                ──▶       accel: { ax: -3384,
  │     ay: -8684                              ay: -8684,
  │     az: 13392                              az: 13392 }
  │     gx: -145                 ──▶       gyro: { gx: -145,
  │     gy: 84                                 gy: 84,
  │     gz: -12                                gz: -12 }
  │     signal: 1818             ──▶       heart: { signal: 1818,
  │     peak: 1                                peak: 1,
  │     down: 0                                down: 0,
  │     bpm: 72                                bpm: 72 }
  │     timestamp_iso: "..."     ──▶       timestamp_iso: datetime(...)
  │                              ──▶       cid: 1
  │                              ──▶       created_at: datetime.utcnow()
```

### Step 4.3 — Store in Database

```
  ├── Bulk insert into MongoDB                  [cattle_sensor_data_ts collection]
  │     └── db[SENSOR_COLLECTION].insert_many(documents)
  │         └── Time-series collection (timeField: timestamp_iso, metaField: cid)
  │
  ├── Log: action="bulk_insert", records_count=N       [logs collection]
```

### Step 4.4 — ML Prediction

```
  ├── ML Behavior Prediction                    [app/ml_model.py]
  │     │
  │     ├── Convert raw rows to dict list
  │     │
  │     ├── predict_from_raw_rows_async(rows, cid)     (runs in thread pool)
  │     │     │
  │     │     ├── Sort by timestamp_ms
  │     │     │
  │     │     ├── Convert accelerometer to m/s²
  │     │     │     raw_value × (9.81 / 16384)          MPU6050 ±2g scale
  │     │     │
  │     │     ├── Estimate sampling rate from timestamps
  │     │     │     median(Δt) → Hz   (typically ~5 Hz)
  │     │     │
  │     │     ├── Segment into 10-second windows
  │     │     │     samples_per_window = 10 × sampling_rate  (≈50 samples)
  │     │     │
  │     │     ├── For each window, compute 24 features:
  │     │     │     │
  │     │     │     │  ┌─ STATISTICAL (8) ──────────────────────────────────┐
  │     │     │     │  │  accel_x_mean    accel_x_std                      │
  │     │     │     │  │  accel_y_mean    accel_y_std                      │
  │     │     │     │  │  accel_z_mean    accel_z_std                      │
  │     │     │     │  │  sma = mean(|ax| + |ay| + |az|)                   │
  │     │     │     │  │  temp_mean                                        │
  │     │     │     │  └───────────────────────────────────────────────────┘
  │     │     │     │
  │     │     │     │  ┌─ ADVANCED (13) ────────────────────────────────────┐
  │     │     │     │  │  Per axis (x, y, z):                              │
  │     │     │     │  │    skewness, kurtosis                             │
  │     │     │     │  │    FFT dominant frequency, FFT dominant amplitude  │
  │     │     │     │  │  temp_std                                         │
  │     │     │     │  └───────────────────────────────────────────────────┘
  │     │     │     │
  │     │     │     │  ┌─ LAG FEATURES (3) ─────────────────────────────────┐
  │     │     │     │  │  prev_sma          (previous window's activity)    │
  │     │     │     │  │  prev_temp_mean    (previous window's temperature) │
  │     │     │     │  │  prev_label_enc    (previous window's prediction)  │
  │     │     │     │  └───────────────────────────────────────────────────┘
  │     │     │     │
  │     │     │     └── Returns feature vector [24 floats]
  │     │     │
  │     │     ├── Pipeline prediction
  │     │     │     ├── SimpleImputer → fill NaN with median
  │     │     │     ├── SMOTE → skipped during inference
  │     │     │     └── RandomForest (500 trees) → integer class [0–6]
  │     │     │
  │     │     ├── Label decoding
  │     │     │     └── label_encoder.inverse_transform([4]) → "Ruminating"
  │     │     │
  │     │     └── Returns most common behavior across all windows
  │     │
  │     ├── Result: { prediction: "Grazing", window_predictions: [...], window_count: N }
  │     │
  │     ├── Store prediction → ml_predictions collection
  │     │     { cid, prediction, status, window_count, window_predictions, timestamp }
  │     │
  │     └── Log: action="ml_prediction", prediction="Grazing"     [logs collection]
```

### Step 4.5 — Health Evaluation (Rule-Based + ML Combined)

```
  ├── evaluate_cattle_health(cid)               [app/alert_services.py]
  │     │
  │     ├── Fetch last 5 sensor readings from DB
  │     │
  │     ├── RULE-BASED EVALUATION               [app/health_evaluator.py]
  │     │     │
  │     │     ├── For each reading, check thresholds:
  │     │     │     ├── Temperature > 39.5°C         → bad  (fever)
  │     │     │     ├── Temperature < 35.0°C         → bad  (hypothermia)
  │     │     │     ├── BPM > 100                    → bad  (tachycardia)
  │     │     │     ├── BPM < 30 (and BPM > 0)      → bad  (bradycardia)
  │     │     │     ├── Activity magnitude < 500     → warning (lethargy)
  │     │     │     └── All normal                   → healthy
  │     │     │
  │     │     └── Overall: any "bad" → bad, any "warning" → warning, else healthy
  │     │
  │     ├── ML BEHAVIOR PREDICTION              [app/ml_model.py]
  │     │     │
  │     │     ├── Fetch last 150 readings (for 10-second windows)
  │     │     ├── predict_from_db_docs(readings, cid)
  │     │     │     └── Extracts features → RandomForest predict → "Grazing"
  │     │     │
  │     │     └── derive_health_status(behavior, temperature, bpm)
  │     │           ├── temp out of range      → "anomaly"
  │     │           ├── BPM out of range       → "anomaly"
  │     │           ├── behavior == "Other"    → "warning"
  │     │           └── normal behavior        → "normal"
  │     │
  │     ├── COMBINE STATUSES                    [_combine_statuses()]
  │     │     │
  │     │     │  ┌──────────────┬──────────────┬──────────────┐
  │     │     │  │  Rule-Based   │  ML Status   │  Combined    │
  │     │     │  ├──────────────┼──────────────┼──────────────┤
  │     │     │  │  healthy      │  normal      │  healthy     │
  │     │     │  │  healthy      │  warning     │  warning  ▲  │
  │     │     │  │  healthy      │  anomaly     │  bad      ▲  │
  │     │     │  │  warning      │  normal      │  warning     │
  │     │     │  │  warning      │  anomaly     │  bad      ▲  │
  │     │     │  │  bad          │  normal      │  bad         │
  │     │     │  │  bad          │  anomaly     │  bad         │
  │     │     │  └──────────────┴──────────────┴──────────────┘
  │     │     │       ▲ = ML escalated the status
  │     │     │
  │     │     └── Worst of both wins — either system can escalate
  │     │
  │     └── Continue to Alert Escalation...
```

### Step 4.6 — Alert Escalation

```
  │     ├── UPDATE CONSECUTIVE COUNTER           [alert_counters collection]
  │     │     │
  │     │     ├── combined status is "bad" or "warning"
  │     │     │     └── INCREMENT consecutive_bad_count by 1
  │     │     │
  │     │     └── combined status is "healthy"
  │     │           └── RESET consecutive_bad_count to 0
  │     │
  │     ├── DETERMINE ALERT LEVEL
  │     │     │
  │     │     ├── count = 0                → None       (no alert)
  │     │     ├── count = 1–3              → "warning"  (alert logged only)
  │     │     └── count ≥ 4               → "critical"  (alert + graph + EMAIL)
  │     │                                    ▲
  │     │                      ALERT_THRESHOLD (configurable, default: 4)
  │     │
  │     │  ┌─────────────────────────────────────────────────────────┐
  │     │  │                 ALERT TRIGGERED?                        │
  │     │  │                                                         │
  │     │  │   combined ≠ healthy  AND  alert_level ≠ None          │
  │     │  │                                                         │
  │     │  │   YES ──────────────────────────────────────▶ Step 4.7  │
  │     │  │   NO  ──────────────────────────────────────▶ Step 4.8  │
  │     │  └─────────────────────────────────────────────────────────┘
```

### Step 4.7 — Alert Notification (when triggered)

Alert is ONLY triggered when `consecutive_bad_count >= ALERT_THRESHOLD` (critical level).
Occasional anomalies (count 1-3) are recorded but do NOT send emails.

```
  │     ├── IF CRITICAL ALERT TRIGGERED (count >= threshold):
  │     │     │
  │     │     ├── Build health summary
  │     │     │     ├── Rule reasons: "High temperature: 40.2°C (threshold: 39.5°C)"
  │     │     │     └── ML reasons:  "ML detected behavior: Other (status: warning)"
  │     │     │
  │     │     ├── Generate 48-hour health graph          [app/graph_service.py]
  │     │     │     ├── Fetch sensor data for last 48 hours
  │     │     │     ├── Plot temperature, BPM, activity with matplotlib
  │     │     │     └── Export as PNG bytes
  │     │     │
  │     │     ├── Resolve email recipient                [app/alert_services.py]
  │     │     │     ├── Fetch cattle → get doctor_id
  │     │     │     ├── IF doctor_id exists:
  │     │     │     │     └── Look up doctor in users collection → get email
  │     │     │     └── IF no doctor_id OR doctor not found:
  │     │     │           └── Use DEFAULT_DOCTOR_EMAIL (kowsiganmv@gmail.com)
  │     │     │
  │     │     ├── Send email to the doctor               [app/email_service.py]
  │     │     │     │
  │     │     │     ├── Check SMTP configured?
  │     │     │     │     ├── NO  → Log "email_skipped" → continue
  │     │     │     │     └── YES → continue
  │     │     │     │
  │     │     │     ├── Build HTML email:
  │     │     │     │     ├── Subject: "🔴 CRITICAL Health Alert — Cattle 1"
  │     │     │     │     ├── Body: alert status, cattle ID, consecutive count,
  │     │     │     │     │         health summary (includes ML behavior), timestamp
  │     │     │     │     └── Embedded: 48-hour health graph PNG
  │     │     │     │
  │     │     │     └── Send via SMTP (TLS on port 587)
  │     │     │           ├── server.starttls()
  │     │     │           ├── server.login(SMTP_USER, SMTP_PASSWORD)
  │     │     │           └── server.sendmail(FROM, doctor_email)
  │     │     │
  │     │     ├── Store alert in DB                      [health_alerts collection]
  │     │     │     {
  │     │     │       cid: 1,
  │     │     │       doctor_id: "admin",
  │     │     │       doctor_email: "doctor@clinic.com",
  │     │     │       doctor_name: "Dr. Veterinary",
  │     │     │       status: "critical",
  │     │     │       consecutive_count: 4,
  │     │     │       email_sent: true,
  │     │     │       health_details: {
  │     │     │         overall_status: "bad",
  │     │     │         rule_status: "bad",
  │     │     │         ml_behavior: "Other",
  │     │     │         ml_status: "warning",
  │     │     │         reasons: ["High temperature: 40.2°C", "ML detected behavior: Other"],
  │     │     │         latest_temperature: 40.2,
  │     │     │         latest_bpm: 73
  │     │     │       },
  │     │     │       graph_generated: true,
  │     │     │       timestamp: "2026-03-08T..."
  │     │     │     }
  │     │     │
  │     │     └── Log alert                              [logs collection]
  │     │           action: "critical_alert"
  │     │           prediction: "Other"
  │     │           prediction_status: "warning"
```

### Step 4.8 — API Response

```
  └── Return response to caller                 [app/routes.py]

      {
        "success": true,
        "cid": 1,
        "inserted_count": 1000,
        "message": "Successfully inserted 1000 sensor readings for cattle 1",
        "prediction": {
          "behavior": "Grazing",
          "status": "normal",
          "window_count": 20,
          "window_predictions": ["Grazing", "Grazing", "Walking", ...]
        }
      }
```

---

## 5. Real-Time Status Endpoint

On-demand ML prediction for any cattle:

```
GET /api/v1/cattle/{cid}/status              [app/routes.py → sensor_router]
  │
  ├── Auth check + farm access verification
  │
  ├── get_cattle_status(cid)                  [app/services.py]
  │     │
  │     ├── Fetch last 150 sensor readings from DB
  │     │     └── Sorted chronologically for window extraction
  │     │
  │     ├── Get latest vitals
  │     │     ├── temperature from most recent reading
  │     │     └── bpm from most recent reading
  │     │
  │     ├── ML prediction (if model loaded)
  │     │     ├── predict_from_db_docs_async(docs, cid)
  │     │     ├── derive_health_status(behavior, temperature, bpm)
  │     │     ├── Store prediction → ml_predictions collection
  │     │     └── Log prediction → logs collection
  │     │
  │     ├── Fallback: if model not loaded
  │     │     └── Use most recent stored prediction from ml_predictions
  │     │
  │     └── Return status dict
  │
  └── Response:
      {
        "cid": 1,
        "behavior": "Grazing",
        "status": "normal",
        "temperature": 38.5,
        "bpm": 72.0,
        "timestamp": "2026-03-08T18:30:00Z"
      }
```

---

## 6. ML Model Details

### Model Architecture

```
cattle_model_v4.pkl (loaded via joblib)
  │
  ├── pipeline (imblearn Pipeline)
  │     ├── Step 1: SimpleImputer(strategy='median')
  │     │             └── Fills NaN values (from zero-variance windows)
  │     ├── Step 2: SMOTE(k_neighbors=3)
  │     │             └── Training only — skipped during inference
  │     └── Step 3: RandomForestClassifier
  │                   ├── n_estimators: 500 trees
  │                   ├── max_depth: 40
  │                   ├── min_samples_leaf: 5
  │                   ├── max_features: 0.4
  │                   ├── class_weight: balanced
  │                   └── n_jobs: -1 (all CPU cores)
  │
  ├── label_encoder (LabelEncoder)
  │     └── 0:Drinking  1:Grazing  2:Lying  3:Other  4:Ruminating  5:Standing  6:Walking
  │
  ├── feature_cols (24 features in order)
  │     └── [accel_x_mean, accel_x_std, accel_y_mean, accel_y_std, accel_z_mean,
  │          accel_z_std, sma, temp_mean, accel_x_skew, accel_x_kurt,
  │          accel_x_fft_dom_freq, accel_x_fft_dom_amp, accel_y_skew, accel_y_kurt,
  │          accel_y_fft_dom_freq, accel_y_fft_dom_amp, accel_z_skew, accel_z_kurt,
  │          accel_z_fft_dom_freq, accel_z_fft_dom_amp, temp_std, prev_sma,
  │          prev_temp_mean, prev_label_enc]
  │
  └── behavior_map
        └── {0:'Grazing', 1:'Walking', 2:'Standing', 3:'Lying', 4:'Ruminating', 6:'Drinking', 7:'Other'}
```

### Feature Extraction Pipeline

```
Raw Sensor Data (from ESP32)
  │
  ├── Accelerometer conversion
  │     raw_ax × (9.81 / 16384) = accel_x_mps2     (MPU6050 ±2g scale)
  │
  ├── Window segmentation
  │     Estimate sampling rate → ~5 Hz
  │     Window size: 10 seconds → ~50 samples/window
  │
  └── Per-window feature computation
        │
        ├── Statistical:  mean, std for each axis (6)
        ├── Activity:     SMA = mean(|ax| + |ay| + |az|) (1)
        ├── Temperature:  mean, std (2)
        ├── Shape:        skewness, kurtosis per axis (6)
        ├── Frequency:    FFT dominant freq & amplitude per axis (6)
        └── Temporal:     previous window's SMA, temp, label (3)
                                                        ─────
                                                     24 total
```

### Behavior Classes

```
┌─────────────┬──────────────────────────────────────────────────────┐
│  Behavior    │  Description                                        │
├─────────────┼──────────────────────────────────────────────────────┤
│  Grazing     │  Consuming grass/feed — rhythmic jaw ~0.5–1 Hz      │
│  Walking     │  Directed locomotion — leg cadence ~1.5–2 Hz        │
│  Standing    │  Stationary upright — low SMA, constant accel       │
│  Lying       │  Recumbent posture — very low SMA, gravity-dominant │
│  Ruminating  │  Re-chewing cud — periodic jaw ~0.5 Hz              │
│  Drinking    │  Consuming water — sharp Z-axis spikes              │
│  Other       │  Miscellaneous/transition — mixed signals           │
└─────────────┴──────────────────────────────────────────────────────┘
```

---

## 7. Health Status Derivation

Two independent systems combine to produce the final health status:

```
┌──────────────────────────────────────────────────────────────────┐
│                    HEALTH STATUS LOGIC                            │
│                                                                  │
│   RULE-BASED (health_evaluator.py)                               │
│   ──────────────────────────────────                             │
│   temp > 39.5°C or temp < 35.0°C      → bad                     │
│   BPM > 100 or BPM < 30               → bad                     │
│   activity_magnitude < 500             → warning                 │
│   all normal                           → healthy                 │
│                                                                  │
│   ML-BASED (ml_model.py)                                         │
│   ──────────────────────                                         │
│   behavior == "Other"                  → warning                 │
│   temp/BPM out of range               → anomaly                 │
│   normal behavior + good vitals        → normal                  │
│                                                                  │
│   COMBINED (alert_services.py → _combine_statuses)               │
│   ─────────────────────────────────────────────────              │
│   Take the WORSE of both:                                        │
│     healthy + normal   = healthy                                 │
│     healthy + warning  = warning     ← ML escalated              │
│     healthy + anomaly  = bad         ← ML escalated              │
│     warning + normal   = warning                                 │
│     bad     + anything = bad                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Alert Escalation System

```
                    Each evaluation cycle
                          │
                          ▼
              ┌───────────────────────┐
              │  Combined status?      │
              └───────┬───────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      "healthy"   "warning"     "bad"
          │           │           │
          ▼           └─────┬─────┘
     RESET counter          ▼
     to 0              INCREMENT
                       counter +1
                            │
                            ▼
              ┌───────────────────────┐
              │  Consecutive count?    │
              └───────┬───────────────┘
                      │
          ┌───────────┼───────────────┐
          ▼           ▼               ▼
       count=0    count 1–3      count ≥ 4
          │           │               │
          ▼           ▼               ▼
       No alert   ┌────────┐    ┌──────────┐
                  │WARNING │    │ CRITICAL │
                  │        │    │          │
                  │ • Log  │    │ • Log    │
                  │   alert│    │   alert  │
                  │        │    │ • 48h    │
                  │        │    │   graph  │
                  │        │    │ • EMAIL  │
                  │        │    │   to farm│
                  │        │    │   admins │
                  └────────┘    └──────────┘

        Alert auto-resets when a "healthy" reading arrives.
        ALERT_THRESHOLD is configurable (default: 4).
```

---

## 9. Email Notification Content

When a **CRITICAL** alert fires:

```
┌──────────────────────────────────────────────────────┐
│  🔴 CRITICAL Health Alert — Cattle 1                  │
│  ─────────────────────────────────────────────────    │
│                                                       │
│  Hello Farm Admin,                                    │
│                                                       │
│  A CRITICAL health alert has been triggered.          │
│                                                       │
│  ┌────────────────────┬──────────────────────┐       │
│  │ Cattle ID           │ 1                    │       │
│  │ Alert Level         │ CRITICAL             │       │
│  │ Consecutive Readings│ 4                    │       │
│  │ Detected At         │ 2026-03-08 18:30 UTC │       │
│  └────────────────────┴──────────────────────┘       │
│                                                       │
│  🩺 Health Summary                                    │
│  ┌──────────────────────────────────────────────┐    │
│  │ High temperature: 40.2°C (threshold: 39.5°C) │    │
│  │ ML detected behavior: Other (status: warning) │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  📊 Health Data (Last 48h)                            │
│  ┌──────────────────────────────────────────────┐    │
│  │  [Temperature / BPM / Activity Graph PNG]     │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ─────────────────────────────────────────────────    │
│  Automated alert from Cattle Health Monitoring System │
└──────────────────────────────────────────────────────┘
```

---

## 10. Database Collections

```
MongoDB Atlas → Database: CDataBase
  │
  ├── cattle                    Cattle metadata (cid, name, farm, breed)
  ├── cattle_sensor_data_ts     Time-series sensor readings (main data)
  ├── cattle_health_events      Health event records
  ├── users                     User accounts (admin/user roles, bcrypt)
  ├── health_alerts             Alert history with email tracking
  ├── alert_counters            Consecutive bad reading counters per cattle
  ├── ml_predictions            ML behavior prediction results
  └── logs                      System operation logs (time-series)
```

---

## 11. Logging

Every significant action is logged to the `logs` time-series collection:

```
┌──────────────┬─────────────────────────┬────────────────────────────────────┐
│  Service      │  Action                 │  When                              │
├──────────────┼─────────────────────────┼────────────────────────────────────┤
│  sensor_api   │  bulk_insert            │  Sensor data stored successfully   │
│  sensor_api   │  invalid_cattle_id      │  Data rejected — cattle not found  │
│  cattle_api   │  create_cattle          │  New cattle registered             │
│  cattle_api   │  update_cattle          │  Cattle metadata updated           │
│  ml_engine    │  ml_prediction          │  ML prediction completed           │
│  alert_system │  warning_alert          │  Warning alert (1–3 bad readings)  │
│  alert_system │  critical_alert         │  Critical alert (≥4 bad readings)  │
│  email_service│  email_sent             │  Alert email delivered             │
│  email_service│  email_skipped          │  SMTP not configured               │
│  email_service│  email_failed           │  SMTP delivery failed              │
└──────────────┴─────────────────────────┴────────────────────────────────────┘
```

ML prediction logs include extra fields:
```json
{
  "service": "ml_engine",
  "action": "ml_prediction",
  "cid": 1,
  "prediction": "Grazing",
  "prediction_status": "normal",
  "message": "ML prediction for CID 1: Grazing"
}
```

---

## 12. API Endpoints Summary

```
AUTHENTICATION
  POST   /api/v1/auth/bootstrap          Create first admin (once)
  POST   /api/v1/auth/login              Login → JWT token
  GET    /api/v1/auth/me                 Current user profile
  POST   /api/v1/auth/register           Register user (admin only)
  GET    /api/v1/auth/users              List users (admin only)
  PUT    /api/v1/auth/users/{username}   Update user (admin only)
  DELETE /api/v1/auth/users/{username}   Deactivate user (admin only)

CATTLE MANAGEMENT
  POST   /api/v1/cattle                  Register new cattle
  GET    /api/v1/cattle                  List all cattle
  GET    /api/v1/cattle/{cid}            Get cattle metadata
  PUT    /api/v1/cattle/{cid}            Update cattle metadata

SENSOR DATA
  POST   /api/v1/cattle/sensor/bulk      Bulk upload + ML prediction
  GET    /api/v1/cattle/latest           Latest reading (all cattle)
  GET    /api/v1/cattle/{cid}/latest     Most recent reading
  GET    /api/v1/cattle/{cid}/status     ML health status (real-time)
  GET    /api/v1/cattle/{cid}/recent     Last N records
  GET    /api/v1/cattle/{cid}/last-hour  Past hour readings
  GET    /api/v1/cattle/{cid}/range      Time range query

HEALTH EVENTS
  GET    /api/v1/cattle/{cid}/health-events   Events for cattle
  GET    /api/v1/health-events/recent         Recent events (all)

ALERTS
  POST   /api/v1/alerts/evaluate/{cid}   Evaluate cattle health
  POST   /api/v1/alerts/evaluate-all     Evaluate all cattle
  GET    /api/v1/alerts/{cid}            Alert history
  GET    /api/v1/alerts/recent/all       Recent alerts (all)
  GET    /api/v1/alerts/{cid}/counter    Bad-reading counter

SYSTEM
  GET    /                               Health check
```

---

## 13. Configuration

All settings are loaded from `.env` via pydantic-settings:

```
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# Database
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
DATABASE_NAME=CDataBase

# Authentication
API_SECRET_KEY=cattle_monitoring_secure_key
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# SMTP Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=your-email@gmail.com

# Alert Thresholds
ALERT_THRESHOLD=4          # Consecutive bad readings for CRITICAL
GRAPH_TIME_WINDOW=48       # Hours of data in health graph

# Health Thresholds
TEMP_HIGH=39.5             # °C — fever threshold
TEMP_LOW=35.0              # °C — hypothermia threshold
BPM_HIGH=100.0             # Heart rate upper limit
BPM_LOW=30.0               # Heart rate lower limit
ACTIVITY_LOW=500.0         # Activity magnitude threshold
```

---

## 14. File Map

```
app/
  main.py              Startup: connect DB → load ML model → register routes
  config.py            All settings from .env (thresholds, SMTP, JWT, DB)
  database.py          MongoDB connection, collections, indexes
  auth.py              JWT decode + API key check + RBAC middleware
  models.py            Pydantic schemas (request validation + response models)
  routes.py            API endpoints (cattle, sensor, health, status)
  services.py          Core logic: transform, insert, ML predict, cattle status
  ml_model.py          ML model loading, 24-feature extraction, prediction
  health_evaluator.py  Rule-based threshold checks (temp, BPM, activity)
  alert_services.py    Alert pipeline: rules + ML → combine → counter → email
  graph_service.py     48-hour matplotlib health graph generation
  email_service.py     SMTP email with embedded graph
  logger.py            Async structured logging to MongoDB
  user_services.py     User CRUD, bcrypt hashing, JWT creation
  user_routes.py       Auth & user management endpoints
  user_models.py       User/auth pydantic models
  alert_routes.py      Alert evaluation & query endpoints
  alert_models.py      Alert pydantic models
```
