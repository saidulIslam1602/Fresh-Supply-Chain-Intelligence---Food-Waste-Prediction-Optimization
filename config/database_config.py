"""
Database configuration for Fresh Supply Chain Intelligence System
Supports SQL Server with connection pooling and Redis caching
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pyodbc
import redis
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# SQL Server Configuration
SQL_SERVER_CONFIG = {
    'server': os.getenv('SQL_SERVER', 'localhost'),
    'database': os.getenv('SQL_DATABASE', 'FreshSupplyChain'),
    'username': os.getenv('SQL_USERNAME', 'sa'),
    'password': 'Saidul1602',
    'driver': os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server'),
    'port': os.getenv('SQL_PORT', '1433')
}

# Connection string for SQL Server
CONNECTION_STRING = (
    f"mssql+pyodbc://{SQL_SERVER_CONFIG['username']}:{SQL_SERVER_CONFIG['password']}"
    f"@{SQL_SERVER_CONFIG['server']},{SQL_SERVER_CONFIG['port']}/{SQL_SERVER_CONFIG['database']}"
    f"?driver={SQL_SERVER_CONFIG['driver'].replace(' ', '+')}"
)

# Redis Configuration for caching
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'decode_responses': True
}

# API Configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 8000)),
    'workers': int(os.getenv('API_WORKERS', 4)),
    'debug': os.getenv('DEBUG', 'True').lower() == 'true'
}

# Security Configuration
SECURITY_CONFIG = {
    'jwt_secret': os.getenv('JWT_SECRET_KEY', 'your-super-secret-jwt-key-here'),
    'api_key': os.getenv('API_KEY', 'your-api-key-here')
}

def get_database_engine():
    """Get SQLAlchemy engine with connection pooling"""
    try:
        engine = create_engine(
            CONNECTION_STRING,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=API_CONFIG['debug']
        )
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise

def get_redis_client():
    """Get Redis client for caching"""
    try:
        client = redis.Redis(
            host=REDIS_CONFIG['host'],
            port=REDIS_CONFIG['port'],
            db=REDIS_CONFIG['db'],
            decode_responses=REDIS_CONFIG['decode_responses'],
            socket_connect_timeout=5,
            socket_timeout=5
        )
        # Test connection
        client.ping()
        logger.info("Redis client connected successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise

def test_database_connection():
    """Test database connectivity"""
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

def test_redis_connection():
    """Test Redis connectivity"""
    try:
        client = get_redis_client()
        client.ping()
        logger.info("Redis connection test successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {str(e)}")
        return False