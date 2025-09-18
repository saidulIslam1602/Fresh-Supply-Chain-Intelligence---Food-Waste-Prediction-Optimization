"""
Test suite for FastAPI endpoints in Fresh Supply Chain Intelligence System
"""

import pytest
import json
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# Create test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert "version" in data
        assert data["status"] == "operational"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        # This might fail if database is not connected, which is expected in tests
        assert response.status_code in [200, 503]
    
    def test_quality_prediction_endpoint_structure(self):
        """Test quality prediction endpoint structure"""
        # Test with invalid token (should fail authentication)
        response = client.post(
            "/api/v1/predict/quality",
            headers={"Authorization": "Bearer invalid_token"},
            json={
                "image_url": "https://example.com/test.jpg",
                "lot_number": "LOT_12345",
                "product_id": 1,
                "warehouse_id": 1
            }
        )
        
        # Should fail authentication
        assert response.status_code == 403
    
    def test_demand_forecast_endpoint_structure(self):
        """Test demand forecast endpoint structure"""
        # Test with invalid token
        response = client.post(
            "/api/v1/forecast/demand",
            headers={"Authorization": "Bearer invalid_token"},
            json={
                "product_id": 1,
                "warehouse_id": 1,
                "horizon_days": 7,
                "include_confidence": True
            }
        )
        
        # Should fail authentication
        assert response.status_code == 403
    
    def test_optimization_endpoint_structure(self):
        """Test optimization endpoint structure"""
        # Test with invalid token
        response = client.post(
            "/api/v1/optimize/distribution",
            headers={"Authorization": "Bearer invalid_token"},
            json={
                "products": [1, 2, 3],
                "warehouses": [1, 2],
                "optimize_for": "cost"
            }
        )
        
        # Should fail authentication
        assert response.status_code == 403
    
    def test_kpi_endpoint_structure(self):
        """Test KPI endpoint structure"""
        # Test with invalid token
        response = client.get(
            "/api/v1/metrics/kpi",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Should fail authentication
        assert response.status_code == 403
    
    def test_websocket_endpoint(self):
        """Test WebSocket endpoint"""
        with client.websocket_connect("/ws/temperature-monitor") as websocket:
            # Should be able to connect
            assert websocket is not None

class TestAPIValidation:
    """Test API request validation"""
    
    def test_quality_prediction_validation(self):
        """Test quality prediction request validation"""
        # Test missing required fields
        response = client.post(
            "/api/v1/predict/quality",
            headers={"Authorization": "Bearer valid_token"},
            json={
                "image_url": "https://example.com/test.jpg",
                # Missing lot_number, product_id, warehouse_id
            }
        )
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_demand_forecast_validation(self):
        """Test demand forecast request validation"""
        # Test invalid horizon_days
        response = client.post(
            "/api/v1/forecast/demand",
            headers={"Authorization": "Bearer valid_token"},
            json={
                "product_id": 1,
                "warehouse_id": 1,
                "horizon_days": -1,  # Invalid negative value
                "include_confidence": True
            }
        )
        
        # Should fail validation
        assert response.status_code == 422
    
    def test_optimization_validation(self):
        """Test optimization request validation"""
        # Test empty products list
        response = client.post(
            "/api/v1/optimize/distribution",
            headers={"Authorization": "Bearer valid_token"},
            json={
                "products": [],  # Empty list
                "warehouses": [1, 2],
                "optimize_for": "cost"
            }
        )
        
        # Should fail validation
        assert response.status_code == 422

class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint"""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed"""
        response = client.put("/api/v1/predict/quality")
        assert response.status_code == 405
    
    def test_malformed_json(self):
        """Test malformed JSON"""
        response = client.post(
            "/api/v1/predict/quality",
            headers={"Authorization": "Bearer valid_token"},
            data="invalid json"
        )
        assert response.status_code == 422

class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_swagger_ui(self):
        """Test Swagger UI endpoint"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc(self):
        """Test ReDoc endpoint"""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

class TestAPIMiddleware:
    """Test API middleware"""
    
    def test_cors_headers(self):
        """Test CORS headers"""
        response = client.options(
            "/api/v1/predict/quality",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should include CORS headers
        assert "access-control-allow-origin" in response.headers
    
    def test_cors_preflight(self):
        """Test CORS preflight request"""
        response = client.options(
            "/api/v1/predict/quality",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code == 200

# Mock authentication for testing
@pytest.fixture
def mock_auth_headers():
    """Mock authentication headers"""
    return {"Authorization": "Bearer valid_token"}

# Test with valid authentication (mock)
class TestAPIWithAuth:
    """Test API with mocked authentication"""
    
    def test_quality_prediction_with_mock_auth(self, mock_auth_headers):
        """Test quality prediction with mock authentication"""
        # This will fail due to missing database, but should pass authentication
        response = client.post(
            "/api/v1/predict/quality",
            headers=mock_auth_headers,
            json={
                "image_url": "https://example.com/test.jpg",
                "lot_number": "LOT_12345",
                "product_id": 1,
                "warehouse_id": 1
            }
        )
        
        # Should pass authentication but fail due to missing database/image
        assert response.status_code in [500, 404]  # Database error or image not found
    
    def test_demand_forecast_with_mock_auth(self, mock_auth_headers):
        """Test demand forecast with mock authentication"""
        response = client.post(
            "/api/v1/forecast/demand",
            headers=mock_auth_headers,
            json={
                "product_id": 1,
                "warehouse_id": 1,
                "horizon_days": 7,
                "include_confidence": True
            }
        )
        
        # Should pass authentication but fail due to missing database
        assert response.status_code in [500, 404]
    
    def test_optimization_with_mock_auth(self, mock_auth_headers):
        """Test optimization with mock authentication"""
        response = client.post(
            "/api/v1/optimize/distribution",
            headers=mock_auth_headers,
            json={
                "products": [1, 2, 3],
                "warehouses": [1, 2],
                "optimize_for": "cost"
            }
        )
        
        # Should pass authentication but fail due to missing database
        assert response.status_code in [500, 404]
    
    def test_kpi_with_mock_auth(self, mock_auth_headers):
        """Test KPI endpoint with mock authentication"""
        response = client.get(
            "/api/v1/metrics/kpi",
            headers=mock_auth_headers
        )
        
        # Should pass authentication but fail due to missing database
        assert response.status_code in [500, 404]

# Performance tests
class TestAPIPerformance:
    """Test API performance"""
    
    def test_response_time(self):
        """Test response time for root endpoint"""
        import time
        
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within reasonable time
        assert response_time < 1.0  # Less than 1 second
        assert response.status_code == 200
    
    def test_concurrent_requests(self):
        """Test concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10
        
        # Should complete within reasonable time
        total_time = end_time - start_time
        assert total_time < 5.0  # Less than 5 seconds for 10 concurrent requests

if __name__ == "__main__":
    pytest.main([__file__])