# Cattle Health Monitoring — API List

Base URL: `http://localhost:8000`

---

## Authentication

The API supports **two authentication methods**:

### 1. JWT Bearer Token (for users/dashboard)

Obtain a token via `POST /api/v1/auth/login`, then include it in all requests:

```
Authorization: Bearer <your_jwt_token>
```

### 2. API Key (for ESP32 devices)

All endpoints also accept the `X-API-Key` header for backward compatibility:

```
X-API-Key: cattle_monitoring_secure_key
```

Missing or invalid credentials → `401 Unauthorized`.

### Access Control (RBAC)

| Role | Permissions |
|------|------------|
| **admin** | Full CRUD on cattle in assigned farms, manage users, ingest sensor data |
| **user** | Read-only access to cattle/sensor/health data in assigned farms |

---

## User Management & Auth

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `POST` | `/api/v1/auth/bootstrap` | Public (once) | Create the first admin user (only when no users exist) |
| `POST` | `/api/v1/auth/register` | Admin | Register a new user |
| `POST` | `/api/v1/auth/login` | Public | Login and receive JWT token |
| `GET` | `/api/v1/auth/me` | Authenticated | Get current user profile |
| `GET` | `/api/v1/auth/users` | Admin | List all users |
| `PUT` | `/api/v1/auth/users/{username}` | Admin | Update user role/farms/status |
| `DELETE` | `/api/v1/auth/users/{username}` | Admin | Deactivate a user |

---

## Health Alerts

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `POST` | `/api/v1/alerts/evaluate/{cid}` | Admin / API Key | Evaluate health for a specific cattle and trigger alerts |
| `POST` | `/api/v1/alerts/evaluate-all` | Admin / API Key | Evaluate health for all cattle |
| `GET` | `/api/v1/alerts/{cid}` | Authenticated | Get alert history for a cattle |
| `GET` | `/api/v1/alerts/recent/all` | Authenticated | Get recent alerts across all cattle |
| `GET` | `/api/v1/alerts/{cid}/counter` | Authenticated | Get consecutive bad-reading counter |

### Alert Levels

| Level | Condition | Action |
|-------|-----------|--------|
| **Warning** | 1–3 consecutive bad readings | Alert logged |
| **Critical** | ≥ 4 consecutive bad readings | Alert logged + email with 48h graph sent to farm admin |

Counter resets when a healthy reading is detected.

---

## Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Check if the API server is running |

---

## Cattle Management

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `POST` | `/api/v1/cattle` | Admin / API Key | Register a new cattle in the system |
| `GET` | `/api/v1/cattle` | Authenticated | List all registered cattle (filtered by farm access) |
| `GET` | `/api/v1/cattle/{cid}` | Authenticated | Get metadata for a specific cattle |
| `PUT` | `/api/v1/cattle/{cid}` | Admin / API Key | Update cattle metadata (name, breed, age, status, farm_id) |

---

## Sensor Data Ingestion

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `POST` | `/api/v1/cattle/sensor/bulk` | Admin / API Key | Bulk ingest sensor data from ESP32, transform and store in MongoDB. **Cattle must exist first (404 if not).** |

**Body:** `{ "cid": int, "data": [ ...sensor rows ] }`

---

## Sensor Data Retrieval

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `GET` | `/api/v1/cattle/latest` | Authenticated | Latest sensor reading for **all cattle** — dashboard overview |
| `GET` | `/api/v1/cattle/{cid}/latest` | Authenticated | Most recent single sensor reading for a cattle |
| `GET` | `/api/v1/cattle/{cid}/recent?limit=N` | Authenticated | Last **N** sensor records (default 100, max 5000), newest first |
| `GET` | `/api/v1/cattle/{cid}/last-hour` | Authenticated | All sensor readings from the **past 1 hour**, sorted ascending |
| `GET` | `/api/v1/cattle/{cid}/range?start=ISO&end=ISO` | Authenticated | Sensor readings between two timestamps — for charts and analytics |

---

## Health Events

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| `GET` | `/api/v1/cattle/{cid}/health-events?limit=N` | Authenticated | Health alerts for a specific cattle (default 50) |
| `GET` | `/api/v1/health-events/recent?limit=N` | Authenticated | Recent health alerts across all cattle — dashboard alert panel |

---

## Parameters

| Parameter | Location | Type | Description |
|-----------|----------|------|-------------|
| `cid` | path | int | Cattle ID |
| `limit` | query | int | Number of records to return |
| `start` | query | datetime | Start time in ISO 8601 format |
| `end` | query | datetime | End time in ISO 8601 format |

---

## Interactive Docs

| UI | URL |
|----|-----|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
