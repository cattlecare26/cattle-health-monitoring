# Cattle Health Monitoring System — REST API

Backend service that receives bulk sensor data from ESP32 devices and stores it in MongoDB.

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python 3.12 | Clean, typed, great ecosystem |
| Framework | FastAPI | Async-native, auto-validation, OpenAPI docs |
| DB Driver | Motor | Async MongoDB driver, supports `insertMany` |
| Validation | Pydantic v2 | Automatic request validation with clear errors |
| Config | pydantic-settings | Type-safe env var loading from `.env` |
| Auth | python-jose + bcrypt | JWT tokens + industry-standard password hashing |
| Alerts | matplotlib + smtplib | Health graph generation + email notifications |

## Project Structure

```
C_RESTAPI/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app entry point
│   ├── auth.py            # Dual auth: JWT Bearer + API key, RBAC dependencies
│   ├── config.py          # Environment variable settings (JWT, SMTP, thresholds)
│   ├── database.py        # MongoDB connection, indexes & collection setup
│   ├── logger.py          # Async logging to MongoDB logs collection
│   ├── models.py          # Pydantic models (cattle/sensor)
│   ├── user_models.py     # Pydantic models (users/auth)
│   ├── alert_models.py    # Pydantic models (health alerts)
│   ├── routes.py          # Cattle/sensor/health endpoints (RBAC-protected)
│   ├── user_routes.py     # Auth & user management endpoints
│   ├── alert_routes.py    # Health alert & evaluation endpoints
│   ├── services.py        # Data transformation & DB operations
│   ├── user_services.py   # User CRUD, bcrypt, JWT management
│   ├── alert_services.py  # Alert orchestration, counter tracking, notifications
│   ├── health_evaluator.py # Sensor data → health status evaluation engine
│   ├── graph_service.py   # Matplotlib time-series graph generation
│   └── email_service.py   # SMTP email with embedded health graphs
├── data.json              # Sample ESP32 sensor data
├── requirements.txt
├── Dockerfile
├── .env                   # Local environment variables (not committed)
├── .env.example           # Template for .env
└── .gitignore
```

## System Architecture

```
ESP32 Device → FastAPI Backend → MongoDB Atlas (Database: CDataBase)
```

Collections: `cattle` · `cattle_sensor_data_ts` · `cattle_health_events` · `users` · `health_alerts` · `alert_counters` · `logs`

## MongoDB Schema

**Database:** `CDataBase`

### Collection: `cattle`
Stores cattle metadata (CID, name, farm, breed, age, status).

### Collection: `cattle_sensor_data_ts` *(Time Series)*
Stores transformed time-series sensor readings from ESP32 devices.

| Config | Value |
|--------|-------|
| Type | Time Series |
| timeField | `timestamp_iso` |
| metaField | `cid` |
| granularity | `seconds` |

Example document:
```json
{
  "cid": 1,
  "timestamp_iso": "2026-02-04T23:31:48.120Z",
  "timestamp_ms": 2122,
  "temperature": 28.47,
  "accel": { "ax": -3384, "ay": -8684, "az": 13392 },
  "gyro":  { "gx": -145,  "gy": 84,    "gz": -12 },
  "heart": { "signal": 1818, "peak": 1, "down": 0, "bpm": 0 },
  "created_at": "2026-02-04T23:35:00Z"
}
```

### Collection: `users`
Stores user accounts with bcrypt-hashed passwords and RBAC roles.

Example document:
```json
{
  "username": "admin1",
  "email": "admin@farm.com",
  "hashed_password": "$2b$12$...",
  "full_name": "Farm Admin",
  "role": "admin",
  "farm_ids": ["Farm-A", "Farm-B"],
  "is_active": true,
  "created_at": "2026-03-22T...",
  "updated_at": "2026-03-22T..."
}
```

### Collection: `cattle_health_events`
Stores alerts and anomaly events.

### Collection: `health_alerts`
Stores health alert records with admin notification tracking.

Example document:
```json
{
  "cid": 1,
  "admin_username": "admin",
  "admin_email": "admin@farm.com",
  "status": "critical",
  "consecutive_count": 4,
  "email_sent": true,
  "health_details": { "overall_status": "bad", "reasons": ["High temperature: 40.2°C"], "latest_temperature": 40.2, "latest_bpm": 73 },
  "graph_generated": true,
  "timestamp": "2026-03-22T..."
}
```

### Collection: `alert_counters`
Tracks consecutive bad readings per cattle for alert escalation.

### Collection: `logs` *(Time Series)*
Stores structured backend operation logs for debugging and monitoring.

| Config | Value |
|--------|-------|
| Type | Time Series |
| timeField | `timestamp` |
| metaField | `service` |
| granularity | `seconds` |

Example document:
```json
{
  "timestamp": "2026-03-08T10:45:22Z",
  "service": "sensor_api",
  "level": "INFO",
  "action": "bulk_insert",
  "collection": "cattle_sensor_data_ts",
  "cid": 1,
  "records_count": 1000,
  "message": "Sensor data uploaded successfully"
}
```

### Indexes
```
cattle:                  { cid: 1 }  (unique)
cattle_sensor_data_ts:   { cid: 1, timestamp_iso: -1 }
cattle_health_events:    { cid: 1 }, { timestamp: -1 }
users:                   { username: 1 } (unique), { email: 1 } (unique)
health_alerts:           { cid: 1 }, { timestamp: -1 }, { cid: 1, timestamp: -1 }
alert_counters:          { cid: 1 } (unique)
logs:                    { service: 1, timestamp: -1 }, { cid: 1 }
```

## API Authentication

The API supports **two authentication methods**:

### 1. JWT Bearer Token (for users/dashboard)

First bootstrap the admin user, then login to get a token:

```bash
# Bootstrap first admin (only works once when no users exist)
curl -X POST http://localhost:8000/api/v1/auth/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@farm.com","password":"SecurePass123","full_name":"Farm Admin","farm_ids":["Farm-A"]}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"SecurePass123"}'

# Use the token
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/cattle
```

### 2. API Key (for ESP32 devices)

```
X-API-Key: cattle_monitoring_secure_key
```

### Role-Based Access Control (RBAC)

| Role | Permissions |
|------|------------|
| **admin** | Full CRUD on cattle in assigned farms, manage users, ingest sensor data |
| **user** | Read-only access to cattle/sensor/health data in assigned farms |

### Resource Ownership

- Users are assigned to `farm_ids` (e.g., `["Farm-A", "Farm-B"]`)
- Admin of "Farm-A" owns and manages all cattle with `farm_id: "Farm-A"`
- Users of "Farm-A" can view (but not modify) those cattle
- Admins with empty `farm_ids` have super-admin access to all farms

## Data Transformation

ESP32 sends flat data → backend converts to structured format:

| Raw (ESP32) | Structured (MongoDB) |
|-------------|---------------------|
| `temp_c` | `temperature` |
| `ax, ay, az` | `accel: { ax, ay, az }` |
| `gx, gy, gz` | `gyro: { gx, gy, gz }` |
| `signal, peak, down, bpm` | `heart: { signal, peak, down, bpm }` |

## API

### User Management & Authentication

#### `POST /api/v1/auth/bootstrap`
Create the first admin user. **Only works when no users exist** (open, no auth).

```bash
curl -X POST http://localhost:8000/api/v1/auth/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@farm.com","password":"SecurePass123","full_name":"Farm Admin","farm_ids":["Farm-A"]}'
```

#### `POST /api/v1/auth/login`
Authenticate and receive a JWT token.

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"SecurePass123"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": { "username": "admin", "email": "admin@farm.com", "role": "admin", ... }
}
```

#### `POST /api/v1/auth/register`
Register a new user. **Admin JWT required.**

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{"username":"farmuser","email":"user@farm.com","password":"UserPass123","full_name":"Farm User","role":"user","farm_ids":["Farm-A"]}'
```

#### `GET /api/v1/auth/me`
Get current user profile. **JWT required.**

#### `GET /api/v1/auth/users`
List all users. **Admin JWT required.**

#### `PUT /api/v1/auth/users/{username}`
Update user role, farm assignments, or active status. **Admin JWT required.**

#### `DELETE /api/v1/auth/users/{username}`
Deactivate a user account. **Admin JWT required.**

---

### Health Alerts & Notifications

#### `POST /api/v1/alerts/evaluate/{cid}`
Evaluate health for a specific cattle. Processes latest sensor data, updates the consecutive counter, and triggers alerts + email if thresholds are met. **Admin / API key required.**

```bash
curl -X POST http://localhost:8000/api/v1/alerts/evaluate/1 \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "cid": 1,
  "status": "bad",
  "consecutive_bad_count": 4,
  "alert_level": "critical",
  "alert_triggered": true,
  "email_sent": true,
  "conditions": [{ "status": "bad", "reasons": ["High temperature: 40.2°C (threshold: 39.5°C)"], "temperature": 40.2, "bpm": 73.0, "activity_magnitude": 16312.49 }],
  "message": "Cattle 1 status: bad (4 consecutive bad readings). Alert level: CRITICAL. Email notification sent."
}
```

#### `POST /api/v1/alerts/evaluate-all`
Evaluate health for all registered cattle. **Admin / API key required.**

#### `GET /api/v1/alerts/{cid}`
Get alert history for a specific cattle, newest first.

#### `GET /api/v1/alerts/recent/all`
Get recent alerts across all cattle (dashboard alert panel).

#### `GET /api/v1/alerts/{cid}/counter`
Get the current consecutive bad-reading counter for a cattle.

#### Alert Levels

| Level | Condition | Action |
|-------|-----------|--------|
| **Warning** | 1–3 consecutive bad readings | Alert logged to `health_alerts` |
| **Critical** | ≥ 4 consecutive bad readings | Alert logged + 48h graph generated + email sent to farm admin |

The counter resets automatically when a healthy reading is detected.

**Automatic evaluation** also runs after each sensor bulk ingestion (`POST /api/v1/cattle/sensor/bulk`).

---

### Cattle Management

#### `POST /api/v1/cattle`
Register a new cattle in the system.

```bash
curl -X POST http://localhost:8000/api/v1/cattle \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cattle_monitoring_secure_key" \
  -d '{ "cid": 1, "name": "Cow-01", "farm_id": "Farm-A", "breed": "HF", "age": 4, "status": "active" }'
```

**Response:**
```json
{ "success": true, "message": "Cattle created successfully", "cid": 1 }
```

#### `GET /api/v1/cattle`
List all registered cattle.

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" http://localhost:8000/api/v1/cattle
```

**Response:**
```json
[
  { "cid": 1, "name": "Cow-01", "farm_id": "Farm-A", "breed": "HF", "age": 4, "status": "active", "created_at": "2026-02-04T23:31:48Z" },
  { "cid": 2, "name": "Cow-02", "farm_id": "Farm-A", "breed": "Jersey", "age": 3, "status": "active", "created_at": "2026-02-05T10:00:00Z" }
]
```

#### `GET /api/v1/cattle/{cid}`
Get metadata for a specific cattle.

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" http://localhost:8000/api/v1/cattle/1
```

**Response:**
```json
{ "cid": 1, "name": "Cow-01", "farm_id": "Farm-A", "breed": "HF", "age": 4, "status": "active", "created_at": "2026-02-04T23:31:48Z" }
```

#### `PUT /api/v1/cattle/{cid}`
Update cattle metadata. Only send fields you want to change.

```bash
curl -X PUT http://localhost:8000/api/v1/cattle/1 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cattle_monitoring_secure_key" \
  -d '{ "breed": "Jersey", "age": 5 }'
```

**Response:**
```json
{ "success": true, "message": "Cattle updated successfully", "cid": 1 }
```

---

### Sensor Data Ingestion

#### `POST /api/v1/cattle/sensor/bulk`

Receive bulk sensor data from an ESP32 device, transform, and store in MongoDB.

> **Validation:** The cattle must be registered (via `POST /api/v1/cattle`) before sensor data can be ingested. Returns `404` if the cattle does not exist.

**Request:**
```json
{
  "cid": 1,
  "data": [
    {
      "timestamp_iso": "2026-02-04T23:31:48.120",
      "timestamp_ms": 2122,
      "temp_c": 28.47,
      "ax": -3384, "ay": -8684, "az": 13392,
      "gx": -145, "gy": 84, "gz": -12,
      "signal": 1818, "peak": 1, "down": 0, "bpm": 0
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "cid": 1,
  "inserted_count": 1,
  "message": "Successfully inserted 1 sensor readings for cattle 1"
}
```

---

### Sensor Data Retrieval

#### `GET /`
Health check — verify the server is running.

#### `GET /api/v1/cattle/latest`
Get the latest sensor reading for **every cattle** (dashboard overview).

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" http://localhost:8000/api/v1/cattle/latest
```

**Response:**
```json
[
  {
    "cid": 1,
    "timestamp_iso": "2026-02-04T23:31:50.120",
    "temperature": 28.51,
    "accel": { "ax": -3400, "ay": -8688, "az": 13424 },
    "gyro": { "gx": -161, "gy": 62, "gz": -11 },
    "heart": { "signal": 1833, "peak": 0, "down": 0, "bpm": 0 }
  }
]
```

#### `GET /api/v1/cattle/{cid}/latest`
Get the most recent sensor reading for a cattle.

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" http://localhost:8000/api/v1/cattle/1/latest
```

**Response:**
```json
{
  "cid": 1,
  "timestamp_iso": "2026-02-04T23:31:50.120",
  "timestamp_ms": 4122,
  "temperature": 28.51,
  "accel": { "ax": -3400, "ay": -8688, "az": 13424 },
  "gyro": { "gx": -161, "gy": 62, "gz": -11 },
  "heart": { "signal": 1833, "peak": 0, "down": 0, "bpm": 0 },
  "created_at": "2026-02-04T23:35:00Z"
}
```

#### `GET /api/v1/cattle/{cid}/recent?limit=100`
Get the last N sensor records (newest first).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Number of records (1–5000) |

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" "http://localhost:8000/api/v1/cattle/1/recent?limit=50"
```

#### `GET /api/v1/cattle/{cid}/last-hour`
Get all sensor readings from the past 1 hour (sorted ascending).

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" http://localhost:8000/api/v1/cattle/1/last-hour
```

#### `GET /api/v1/cattle/{cid}/range?start=...&end=...`
Get sensor readings between two ISO 8601 timestamps.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start` | datetime | ✅ | Start time (ISO 8601) |
| `end` | datetime | ✅ | End time (ISO 8601) |

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" "http://localhost:8000/api/v1/cattle/1/range?start=2026-02-04T22:00:00&end=2026-02-04T23:59:59"
```

---

### Health Events

#### `GET /api/v1/cattle/{cid}/health-events?limit=50`
Get health alerts for a specific cattle, newest first.

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" "http://localhost:8000/api/v1/cattle/1/health-events?limit=20"
```

**Response:**
```json
[
  { "cid": 1, "event": "High Temperature", "value": 39.2, "status": "warning", "timestamp": "2026-02-04T23:40:00Z" }
]
```

#### `GET /api/v1/health-events/recent?limit=50`
Get recent health alerts across all cattle (dashboard alert panel).

```bash
curl -H "X-API-Key: cattle_monitoring_secure_key" "http://localhost:8000/api/v1/health-events/recent?limit=20"
```

**Response:**
```json
[
  { "cid": 1, "event": "High Temperature", "value": 39.2, "status": "warning", "timestamp": "2026-02-04T23:40:00Z" },
  { "cid": 2, "event": "Low Activity", "value": 12.0, "status": "alert", "timestamp": "2026-02-04T23:38:00Z" }
]
```

---

### API Summary

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `GET` | `/` | Public | Health check |
| `POST` | `/api/v1/auth/bootstrap` | Public (once) | Create first admin user |
| `POST` | `/api/v1/auth/login` | Public | Login, get JWT token |
| `POST` | `/api/v1/auth/register` | Admin | Register a new user |
| `GET` | `/api/v1/auth/me` | JWT | Current user profile |
| `GET` | `/api/v1/auth/users` | Admin | List all users |
| `PUT` | `/api/v1/auth/users/{username}` | Admin | Update user |
| `DELETE` | `/api/v1/auth/users/{username}` | Admin | Deactivate user |
| `POST` | `/api/v1/cattle` | Admin / API Key | Register a new cattle |
| `GET` | `/api/v1/cattle` | Authenticated | List all cattle |
| `GET` | `/api/v1/cattle/{cid}` | Authenticated | Get cattle metadata |
| `PUT` | `/api/v1/cattle/{cid}` | Admin / API Key | Update cattle metadata |
| `POST` | `/api/v1/cattle/sensor/bulk` | Admin / API Key | Bulk ingest sensor data |
| `GET` | `/api/v1/cattle/latest` | Authenticated | Latest reading (all cattle) |
| `GET` | `/api/v1/cattle/{cid}/latest` | Authenticated | Most recent sensor reading |
| `GET` | `/api/v1/cattle/{cid}/recent?limit=N` | Authenticated | Last N sensor records |
| `GET` | `/api/v1/cattle/{cid}/last-hour` | Authenticated | Readings from past hour |
| `GET` | `/api/v1/cattle/{cid}/range?start=&end=` | Authenticated | Readings in time range |
| `GET` | `/api/v1/cattle/{cid}/health-events` | Authenticated | Health events for cattle |
| `GET` | `/api/v1/health-events/recent?limit=N` | Authenticated | Recent health alerts |
| `POST` | `/api/v1/alerts/evaluate/{cid}` | Admin / API Key | Evaluate cattle health |
| `POST` | `/api/v1/alerts/evaluate-all` | Admin / API Key | Evaluate all cattle |
| `GET` | `/api/v1/alerts/{cid}` | Authenticated | Alert history for cattle |
| `GET` | `/api/v1/alerts/recent/all` | Authenticated | Recent alerts (all cattle) |
| `GET` | `/api/v1/alerts/{cid}/counter` | Authenticated | Bad-reading counter |

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your MongoDB URI and settings
```

### 3. Run Development Server
```bash
python -m app.main
# or
uvicorn app.main:app --reload --port 8000
```

### 4. Open API Docs
Visit `http://localhost:8000/docs` for interactive Swagger UI.

## Deployment

### Docker
```bash
docker build -t cattle-api .
docker run -p 8000:8000 --env-file .env cattle-api
```

### Cloud Platforms (Render / Railway / AWS)
1. Push code to GitHub
2. Connect repository to platform
3. Set environment variables (`MONGODB_URI`, `DATABASE_NAME`, `API_SECRET_KEY`)
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Deploy

### VPS (Ubuntu)
```bash
# Install Python 3.12
sudo apt update && sudo apt install python3.12 python3.12-venv

# Clone and setup
git clone <repo-url> && cd C_RESTAPI
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env && nano .env  # Set MONGODB_URI

# Run with process manager
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## System Logging

The backend records all database actions and important system events in the `logs` Time Series collection.

**Logged operations:**
- Cattle creation and updates
- Sensor bulk uploads (success and failure)
- Invalid cattle ID rejections
- Database errors

**Log levels:** `INFO`, `WARNING`, `ERROR`, `DEBUG`

**Service identifiers:** `sensor_api`, `cattle_api`, `health_event_api`, `system`

---

## Test with Sample Data

First register a cattle, then send sensor data:

```bash
# 1. Register the cattle
curl -X POST http://localhost:8000/api/v1/cattle \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cattle_monitoring_secure_key" \
  -d '{ "cid": 1, "name": "Cow-01", "farm_id": "Farm-A", "breed": "HF", "age": 4, "status": "active" }'

# 2. Send sensor data
curl -X POST http://localhost:8000/api/v1/cattle/sensor/bulk \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cattle_monitoring_secure_key" \
  -H "Content-Type: application/json" \
  -d '{
    "cid": 1,
    "data": [{
      "timestamp_iso": "2026-02-04T23:31:48.120",
      "timestamp_ms": 2122,
      "temp_c": 28.47,
      "ax": -3384, "ay": -8684, "az": 13392,
      "gx": -145, "gy": 84, "gz": -12,
      "signal": 1818, "peak": 1, "down": 0, "bpm": 0
    }]
  }'
```
