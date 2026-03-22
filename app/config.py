"""
Application settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGODB_URI: str
    DATABASE_NAME: str = "CDataBase"
    API_SECRET_KEY: str = "changeme"
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # JWT Configuration
    JWT_SECRET_KEY: str = "change-this-to-a-secure-random-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # SMTP / Email Configuration
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = ""

    # Alert System Configuration
    ALERT_THRESHOLD: int = 4
    GRAPH_TIME_WINDOW: int = 48

    # Health Thresholds (configurable per deployment)
    TEMP_HIGH: float = 39.5
    TEMP_LOW: float = 35.0
    BPM_HIGH: float = 100.0
    BPM_LOW: float = 30.0
    ACTIVITY_LOW: float = 500.0

    class Config:
        env_file = ".env"


settings = Settings()
