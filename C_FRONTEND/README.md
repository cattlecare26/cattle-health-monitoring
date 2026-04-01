# 🐄 Cattle Health Monitoring System — Frontend

A production-grade **Streamlit** frontend for real-time cattle health monitoring, featuring role-based dashboards, interactive analytics, multi-language support, and an enterprise-grade theming system.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Role-Based Access Control** | Three-tier hierarchy — Super Admin, Admin (Vet Doctor), and User (Farmer) — each with scoped views and permissions |
| **Interactive Analytics** | Plotly-powered gauges, time-series charts for temperature, heart rate, activity, acceleration, gyroscope, and heart signal |
| **Real-Time Alerts** | Health evaluation engine with color-coded alert cards, consecutive-reading counters, and email notification triggers |
| **Multi-Language (i18n)** | Full support for **English**, **Tamil**, and **Hindi** with 150+ translated keys |
| **Light / Dark Theme** | GitHub-inspired design system with comprehensive color palettes, togglable from any page |
| **Cattle Management** | Complete CRUD operations for cattle records (Admin / Super Admin) |
| **User Management** | Role-aware user creation, editing, farm assignment, and deactivation |
| **Docker Ready** | Production Dockerfile with health checks included |

---

## 🛠️ Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/) 1.45
- **Visualization:** [Plotly](https://plotly.com/python/) 6.0
- **Data Handling:** [Pandas](https://pandas.pydata.org/) 2.2
- **HTTP Client:** [Requests](https://docs.python-requests.org/) 2.32
- **Config:** [python-dotenv](https://github.com/theskumar/python-dotenv) 1.0
- **Runtime:** Python 3.11

---

## 📁 Project Structure

```
C_FRONTEND/
├── app.py                        # Entry point — routing, theme CSS injection
├── Dockerfile                    # Production container image
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .streamlit/
│   └── config.toml               # Streamlit server & theme config
├── assets/
│   └── logo.png                  # Application logo
├── views/                        # Page modules
│   ├── login.py                  # Authentication page
│   ├── dashboard_super_admin.py  # System-wide overview & farm mapping
│   ├── dashboard_admin.py        # Farm-scoped dashboard
│   ├── dashboard_user.py         # Farmer's cattle grid
│   ├── cattle_detail.py          # Detailed health analytics & charts
│   ├── cattle_management.py      # Cattle CRUD (admin+)
│   ├── user_management.py        # User CRUD (role-aware)
│   ├── alerts.py                 # Alert management & health evaluation
│   ├── messages.py               # Alert messages with owner contact info
│   └── profile.py                # User profile display
├── components/                   # Reusable UI components
│   ├── sidebar.py                # Navigation sidebar (role-aware)
│   ├── navbar.py                 # Top bar — logo, theme toggle, language
│   └── charts.py                 # Plotly chart builders (gauges, trends)
├── services/
│   └── api_client.py             # HTTP client for all backend endpoints
└── utils/
    ├── auth.py                   # Session state, login/logout, role helpers
    ├── theme.py                  # Light/dark palettes & status colors
    ├── translations.py           # i18n dictionaries (EN / TA / HI)
    ├── icons.py                  # 25+ inline SVG icons
    └── logo.py                   # Base64-encoded logo utility
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A running instance of the **CHM Backend API**

### Local Setup

```bash
# 1. Clone & navigate
cd C_FRONTEND

# 2. Create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set BASE_URL to your backend API address

# 5. Run the app
streamlit run app.py
```

The app will be available at **http://localhost:8501**.

### Docker

```bash
# Build
docker build -t chm-frontend .

# Run
docker run -p 8501:8501 \
  -e BASE_URL=http://your-backend:8000 \
  chm-frontend
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URL` | `http://127.0.0.1:8000` | Backend API base URL (no trailing slash) |
| `DEFAULT_LANG` | `en` | Default UI language — `en`, `ta`, or `hi` |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Streamlit bind address |

---

## 👥 Role-Based Access

### Super Admin
- Full system overview with farm → admin → user → cattle hierarchy
- Create and manage admins and users across all farms
- Evaluate health for individual cattle or entire herd (triggers email alerts)
- Access all dashboards, cattle records, and alerts

### Admin (Veterinary Doctor)
- Dashboard scoped to assigned farms
- Manage cattle and create farmer accounts within own farms
- Monitor alerts and sensor readings for farm cattle

### User (Farmer)
- Personal cattle grid with search
- Detailed health analytics per cattle (gauges, trends, event timeline)
- View alerts, messages, and profile information

---

## 📊 Analytics & Charts

The cattle detail page provides interactive Plotly visualizations:

- **Gauges** — Real-time temperature, heart rate, and activity level indicators with threshold zones
- **Overview Trends** — Multi-panel time-series for temperature, BPM, and activity
- **Acceleration** — X, Y, Z axis time-series
- **Gyroscope** — X, Y, Z axis time-series
- **Heart Signal** — Waveform plot
- **Health Events** — Color-coded timeline (critical / warning / healthy)

Time range options: *Last Hour*, *Last 24 Hours*, *Recent 500 Readings*, and *Custom Range*.

---

## 🔌 Backend API Integration

The `services/api_client.py` module communicates with the backend across these endpoint groups:

| Group | Operations |
|-------|------------|
| **Auth** | Login, register, fetch current user, list/update/delete users |
| **Cattle** | List, create, update, fetch by ID |
| **Sensor Data** | Latest readings, recent history, last-hour data, custom date range |
| **Alerts** | Fetch by cattle, recent alerts, alert counters, evaluate health |
| **Health Events** | Recent events (system-wide), events per cattle |

All requests use **Bearer token** authentication.

---

## 🔒 Security

- Bearer token authentication on every API call
- Role-based routing enforced at the UI layer
- Client-side form validation (server validates independently)
- Logout clears all session state
- No secrets stored client-side

---

## 📜 License

This project is part of the **Cattle Health Monitoring System** full-stack application.
