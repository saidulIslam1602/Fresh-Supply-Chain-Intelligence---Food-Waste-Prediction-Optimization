"""
Pytest configuration and shared fixtures for Fresh Supply Chain Intelligence System
Comprehensive test setup with database fixtures, mock services, and test utilities
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from PIL import Image
import torch
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import structlog

# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Configure test logging
structlog.configure(
    processors=[
        structlog.testing.LogCapture(),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.testing.TestingLoggerFactory(),
    cache_logger_on_first_use=True,
)

# Test database configuration
TEST_DATABASE_URL = "sqlite:///./test_fresh_supply.db"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for tests

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_database_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    
    # Create test tables
    with engine.connect() as conn:
        # Create basic test tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS Warehouses (
                WarehouseID INTEGER PRIMARY KEY,
                WarehouseName TEXT NOT NULL,
                Location TEXT,
                Capacity INTEGER,
                TemperatureMin REAL,
                TemperatureMax REAL
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS Products (
                ProductID INTEGER PRIMARY KEY,
                ProductName TEXT NOT NULL,
                Category TEXT,
                ShelfLifeDays INTEGER,
                OptimalTempMin REAL,
                OptimalTempMax REAL
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS TemperatureLogs (
                LogID INTEGER PRIMARY KEY,
                WarehouseID INTEGER,
                Temperature REAL,
                Humidity REAL,
                LogTime DATETIME,
                QualityScore REAL,
                FOREIGN KEY (WarehouseID) REFERENCES Warehouses(WarehouseID)
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS WasteEvents (
                EventID INTEGER PRIMARY KEY,
                ProductID INTEGER,
                WarehouseID INTEGER,
                WasteAmount REAL,
                WasteReason TEXT,
                EventDate DATETIME,
                CostImpact REAL,
                FOREIGN KEY (ProductID) REFERENCES Products(ProductID),
                FOREIGN KEY (WarehouseID) REFERENCES Warehouses(WarehouseID)
            )
        """))
        
        conn.commit()
    
    yield engine
    
    # Cleanup
    if os.path.exists("./test_fresh_supply.db"):
        os.remove("./test_fresh_supply.db")

@pytest.fixture
def test_database_session(test_database_engine):
    """Create test database session"""
    Session = sessionmaker(bind=test_database_engine)
    session = Session()
    
    # Insert test data
    with test_database_engine.connect() as conn:
        # Insert test warehouses
        conn.execute(text("""
            INSERT OR REPLACE INTO Warehouses 
            (WarehouseID, WarehouseName, Location, Capacity, TemperatureMin, TemperatureMax)
            VALUES 
            (1, 'Oslo Central', 'Oslo, Norway', 10000, 2.0, 6.0),
            (2, 'Bergen Hub', 'Bergen, Norway', 8000, 2.0, 6.0),
            (3, 'Trondheim Depot', 'Trondheim, Norway', 5000, 2.0, 6.0)
        """))
        
        # Insert test products
        conn.execute(text("""
            INSERT OR REPLACE INTO Products 
            (ProductID, ProductName, Category, ShelfLifeDays, OptimalTempMin, OptimalTempMax)
            VALUES 
            (1, 'Fresh Salmon', 'Fish', 3, 0.0, 4.0),
            (2, 'Organic Apples', 'Fruit', 14, 1.0, 4.0),
            (3, 'Leafy Greens', 'Vegetable', 7, 0.0, 4.0),
            (4, 'Dairy Milk', 'Dairy', 7, 2.0, 6.0),
            (5, 'Chicken Breast', 'Meat', 2, 0.0, 4.0)
        """))
        
        # Insert test temperature logs
        import datetime
        base_time = datetime.datetime.now() - datetime.timedelta(days=7)
        for i in range(100):
            log_time = base_time + datetime.timedelta(hours=i)
            temp = np.random.normal(4.0, 1.0)  # Normal around 4°C
            humidity = np.random.normal(85, 5)  # Normal around 85%
            quality_score = max(0.0, min(1.0, 1.0 - abs(temp - 4.0) / 10.0))
            
            conn.execute(text("""
                INSERT INTO TemperatureLogs 
                (WarehouseID, Temperature, Humidity, LogTime, QualityScore)
                VALUES (:warehouse_id, :temp, :humidity, :log_time, :quality_score)
            """), {
                "warehouse_id": (i % 3) + 1,
                "temp": temp,
                "humidity": humidity,
                "log_time": log_time,
                "quality_score": quality_score
            })
        
        conn.commit()
    
    yield session
    session.close()

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.ping.return_value = True
    mock_redis.info.return_value = {"redis_version": "6.2.0"}
    return mock_redis

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a simple RGB image
    image = Image.new('RGB', (224, 224), color='green')
    
    # Add some variation to make it more realistic
    pixels = np.array(image)
    noise = np.random.randint(0, 50, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255)
    
    return Image.fromarray(pixels.astype(np.uint8))

@pytest.fixture
def sample_image_path(sample_image):
    """Create a temporary image file"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        sample_image.save(tmp_file.name, 'JPEG')
        yield tmp_file.name
    
    # Cleanup
    if os.path.exists(tmp_file.name):
        os.remove(tmp_file.name)

@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for forecasting tests"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Create realistic demand patterns
    base_demand = 100
    seasonal_pattern = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Yearly
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly
    noise = np.random.normal(0, 5, n_days)
    trend = 0.1 * np.arange(n_days) / 365.25  # Slight upward trend
    
    demand = base_demand + seasonal_pattern + weekly_pattern + noise + trend
    demand = np.maximum(demand, 0)  # Ensure non-negative
    
    data = pd.DataFrame({
        'Date': dates,
        'Demand': demand,
        'Temperature': np.random.normal(4, 2, n_days),
        'Humidity': np.random.normal(85, 10, n_days),
        'DayOfWeek': dates.dayofweek,
        'Month': dates.month,
        'IsWeekend': dates.dayofweek.isin([5, 6]).astype(int)
    })
    
    return data

@pytest.fixture
def sample_supply_chain_network():
    """Create sample supply chain network data"""
    nodes_data = {
        'NodeID': [1, 2, 3, 4, 5, 6],
        'NodeType': ['SUPPLIER', 'WAREHOUSE', 'WAREHOUSE', 'RETAIL', 'RETAIL', 'RETAIL'],
        'NodeName': ['Nordic Supplier', 'Oslo Hub', 'Bergen Hub', 'Oslo Store 1', 'Oslo Store 2', 'Bergen Store 1'],
        'LocationLat': [60.0, 59.9139, 60.3913, 59.9200, 59.9300, 60.3900],
        'LocationLon': [10.0, 10.7522, 5.3221, 10.7600, 10.7700, 5.3300],
        'Capacity': [5000, 2000, 1500, 200, 200, 150],
        'LeadTimeDays': [2, 0, 0, 0, 0, 0]
    }
    
    edges_data = {
        'SourceNodeID': [1, 1, 2, 2, 3, 3],
        'TargetNodeID': [2, 3, 4, 5, 6, 2],
        'TransportMode': ['TRUCK', 'TRUCK', 'TRUCK', 'TRUCK', 'TRUCK', 'TRUCK'],
        'DistanceKM': [50, 300, 10, 15, 8, 300],
        'TransitTimeDays': [0.5, 3.0, 0.1, 0.2, 0.1, 3.0],
        'CostPerUnit': [0.10, 0.30, 0.02, 0.03, 0.02, 0.30]
    }
    
    return {
        'nodes': pd.DataFrame(nodes_data),
        'edges': pd.DataFrame(edges_data)
    }

@pytest.fixture
def mock_ml_models():
    """Mock ML models for testing"""
    mock_vision_model = Mock()
    mock_vision_model.predict_quality.return_value = ('Fresh', 0.85, np.array([0.85, 0.10, 0.03, 0.01, 0.01]))
    mock_vision_model.extract_features.return_value = np.random.random(512)
    mock_vision_model.num_classes = 5
    mock_vision_model.quality_labels = ['Fresh', 'Good', 'Fair', 'Poor', 'Spoiled']
    
    mock_forecasting_model = Mock()
    mock_forecasting_model.predict.return_value = {
        'forecast': np.random.random(7) * 100,
        'confidence_intervals': {
            'lower': np.random.random(7) * 80,
            'upper': np.random.random(7) * 120
        },
        'uncertainty': np.random.random(7) * 0.1
    }
    
    mock_gnn_model = Mock()
    mock_gnn_model.optimize_routes.return_value = {
        'optimal_routes': [{'from': 1, 'to': 2, 'cost': 100}],
        'total_cost': 500,
        'optimization_time': 2.5
    }
    
    return {
        'vision': mock_vision_model,
        'forecasting': mock_forecasting_model,
        'gnn': mock_gnn_model
    }

@pytest.fixture
def api_test_client():
    """Create FastAPI test client"""
    # Import here to avoid circular imports
    from api.main import app
    return TestClient(app)

@pytest.fixture
def authenticated_headers():
    """Create authenticated headers for API testing"""
    # Mock JWT token for testing
    return {
        "Authorization": "Bearer test_token_12345",
        "Content-Type": "application/json"
    }

@pytest.fixture
def mock_database_operations():
    """Mock database operations for testing"""
    with patch('sqlalchemy.create_engine') as mock_engine:
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.execute.return_value.fetchone.return_value = None
        yield mock_conn

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_business_metrics():
    """Sample business metrics data for testing"""
    return {
        'otif_rate': 0.95,
        'temperature_compliance': 0.92,
        'waste_reduction': 0.23,
        'cost_savings': 125000,
        'customer_satisfaction': 4.2,
        'energy_efficiency': 0.78,
        'carbon_footprint': 850.5
    }

@pytest.fixture
def mock_external_apis():
    """Mock external API calls"""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'success', 'data': {}}
        
        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        
        yield {
            'get': mock_get,
            'post': mock_post,
            'response': mock_response
        }

@pytest.fixture
def performance_test_data():
    """Generate data for performance testing"""
    return {
        'large_dataset': pd.DataFrame({
            'id': range(10000),
            'value': np.random.random(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='H')
        }),
        'stress_test_requests': [
            {'endpoint': '/api/v2/predict/quality', 'method': 'POST', 'data': {'image_url': f'test_{i}.jpg'}}
            for i in range(100)
        ]
    }

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test"""
    yield
    
    # Cleanup any test files that might have been created
    test_files = [
        'test_model.pth',
        'test_data.csv',
        'test_config.json'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)

# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as ML model test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "database: mark test as database test"
    )

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_response_structure(response_data: Dict[str, Any], required_fields: list):
        """Assert that response has required structure"""
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
    
    @staticmethod
    def assert_model_output_valid(prediction, confidence, probabilities=None):
        """Assert that model output is valid"""
        assert prediction is not None
        assert 0 <= confidence <= 1
        if probabilities is not None:
            assert len(probabilities) > 0
            assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)
    
    @staticmethod
    def create_mock_image_tensor(batch_size=1, channels=3, height=224, width=224):
        """Create mock image tensor for testing"""
        return torch.randn(batch_size, channels, height, width)
    
    @staticmethod
    def create_mock_time_series(length=100, features=5):
        """Create mock time series data"""
        return torch.randn(1, length, features)

@pytest.fixture
def test_utils():
    """Provide test utilities"""
    return TestUtils