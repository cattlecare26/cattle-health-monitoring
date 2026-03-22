"""
Cattle Health Monitoring System — REST API
Main application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import connect_db, close_db
from app.routes import cattle_router, sensor_router, health_router
from app.user_routes import auth_router
from app.alert_routes import alert_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="Cattle Health Monitoring API",
    description="REST API for ingesting sensor data from ESP32 devices attached to cattle.",
    version="1.0.0",
    lifespan=lifespan,
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
