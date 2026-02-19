

import os
from datetime import date, timedelta
from typing import Optional
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import QueuePool
import pandas as pd

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "141.94.31.144"),
    "port": os.getenv("DB_PORT", "3306"),
    "user": os.getenv("DB_USER", "AdminNidec"),
    "password": os.getenv("DB_PASSWORD", "u6Ehe987XBSXxa4"),
    "database": os.getenv("DB_NAME", "indicator"),
}

LOGV0_BASE_URL = os.getenv("LOGV0_BASE_URL", "http://10.154.206.186:8015/")

DATABASE_URL = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
)


def get_sites() -> list[str]:
    query = """
        SELECT DISTINCT Site
        FROM kpi_sessions
        WHERE Site IS NOT NULL
        ORDER BY Site
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchall()
        return [row[0] for row in result]


def get_date_range() -> dict:
    query = """
        SELECT
            MIN(DATE(`Datetime start`)) as date_min,
            MAX(DATE(`Datetime start`)) as date_max
        FROM kpi_sessions
    """
    with engine.connect() as conn:
        result = conn.execute(text(query)).fetchone()
        return {
            "min": result[0] or date.today() - timedelta(days=365),
            "max": result[1] or date.today(),
        }


def query_df(sql: str, params: dict = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def table_exists(table_name: str) -> bool:
    inspector = inspect(engine)
    return inspector.has_table(table_name)


def ensure_users_table() -> None:
    if table_exists("users"):
        return

    create_sql = """
        CREATE TABLE users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()


def get_user_by_username(username: str) -> Optional[dict]:
    query = text(
        "SELECT id, username, password_hash, is_active, created_at FROM users WHERE username = :username"
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"username": username}).fetchone()
        if row is None:
            return None
        return {
            "id": row.id,
            "username": row.username,
            "password_hash": row.password_hash,
            "is_active": bool(row.is_active),
            "created_at": row.created_at,
        }


def create_user(username: str, password_hash: str, is_active: bool = True) -> dict:
    insert_sql = text(
        "INSERT INTO users (username, password_hash, is_active) VALUES (:username, :password_hash, :is_active)"
    )
    with engine.connect() as conn:
        result = conn.execute(
            insert_sql,
            {
                "username": username,
                "password_hash": password_hash,
                "is_active": is_active,
            },
        )
        conn.commit()

        return {
            "id": result.lastrowid,
            "username": username,
            "password_hash": password_hash,
            "is_active": is_active,
        }
