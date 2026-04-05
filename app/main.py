"""
Cattle Health Monitoring System — REST API
Main application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import connect_db, close_db
from app.routes import cattle_router, sensor_router, health_router
from app.user_routes import auth_router
from app.alert_routes import alert_router
from app.config import settings
from app.ml_model import load_model


async def _seed_default_users() -> None:
    """Seed default users if the users collection is empty."""
    from app.user_services import get_user_count, create_user

    count = await get_user_count()
    if count > 0:
        return

    default_users = [
        {
            "username": "dev",
            "email": "dev@cattlemonitor.com",
            "password": "dev@123!Secure",
            "full_name": "System Developer",
            "phone": "0000000000",
            "role": "super_admin",
            "farm_ids": [],
        },
        {
            "username": "admin",
            "email": "kowsiganmv@gmail.com",
            "password": "admin@123!Secure",
            "full_name": "Default Veterinary Doctor",
            "phone": "9876543210",
            "role": "admin",
            "farm_ids": ["farm_001"],
        },
        {
            "username": "farmer",
            "email": "farmer@cattlemonitor.com",
            "password": "farmer@123!Secure",
            "full_name": "Default Farmer",
            "phone": "9876543211",
            "role": "user",
            "farm_ids": ["farm_001"],
        },
    ]

    for u in default_users:
        try:
            await create_user(**u)
            print(f"  👤 Seeded user: {u['username']} (role: {u['role']})")
        except Exception as e:
            print(f"  ⚠️  Failed to seed {u['username']}: {e}")

    print("✅ Default users seeded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    await connect_db()
    await _seed_default_users()
    load_model()
    yield
    await close_db()


app = FastAPI(
    title="Cattle Health Monitoring API",
    description="REST API for ingesting sensor data from ESP32 devices attached to cattle.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(sensor_router)
app.include_router(health_router)
app.include_router(cattle_router)
app.include_router(alert_router)


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "Cattle Health Monitoring API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
    )
