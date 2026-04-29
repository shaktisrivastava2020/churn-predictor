"""
Database module for Churn Predictor.
Manages Cloud SQL PostgreSQL connection via Cloud SQL Python Connector.
Reuses the QuickShop AI Cloud SQL instance.
"""
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from config import settings

_connector: Connector | None = None
_engine: Engine | None = None


def _get_connector() -> Connector:
    """Lazy-init the Cloud SQL Connector."""
    global _connector
    if _connector is None:
        _connector = Connector(ip_type=IPTypes.PUBLIC)
    return _connector


def _getconn():
    """Connection factory used by SQLAlchemy."""
    connector = _get_connector()
    return connector.connect(
        settings.INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        db=settings.DB_NAME,
    )


def get_engine() -> Engine:
    """Return a singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            "postgresql+pg8000://",
            creator=_getconn,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=2,
        )
    return _engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def get_session():
    """FastAPI dependency: yields a DB session, closes on completion."""
    SessionLocal.configure(bind=get_engine())
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
