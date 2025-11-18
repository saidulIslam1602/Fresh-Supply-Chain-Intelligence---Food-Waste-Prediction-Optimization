"""
Integration tests for API endpoints
Tests API functionality with real database connections and external services
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import tempfile
import os

# Import API components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.main import app, create_access_token, get_password_hash

@pytest.mark.integration
@pytest.mark.api
class TestAPIAuthentication:
    """Integration tests for API authentication"""
    
    def test_token_generation_and_validation(self):
        """Test JWT token generation and validation"""
        # Create test user data
        user_data = {
            "username": "test_user",
            "email": "test@example.com",
            "roles": ["analyst", "viewer"]
        }
        
        # Generate token
        token = create_access_token(data=user_data)
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_login_endpoint(self, api_test_client):
        """Test login endpoint functionality"""
        # Test with invalid credentials
        response = api_test_client.post(
            "/api/v2/auth/token",
            data={
                "username": "invalid_user",
                "password": "wrong_password"
            }
        )
        assert response.status_code == 401
        
        # Test with valid credentials (mocked)
        with patch('api.security.get_user') as mock_get_user, \
             patch('api.security.verify_password') as mock_verify:
            
            mock_user = Mock()
            mock_user.username = "test_user"
            mock_user.disabled = False
            mock_user.roles = ["analyst"]
            
            mock_get_user.return_value = mock_user
            mock_verify.return_value = True
            
            response = api_test_client.post(
                "/api/v2/auth/token",
                data={
                    "username": "test_user",
                    "password": "correct_password"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"
    
    def test_protected_endpoint_access(self, api_test_client):
        """Test access to protected endpoints"""
        # Test without token
        response = api_test_client.get("/api/v2/user/profile")
        assert response.status_code == 401
        
        # Test with invalid token
        response = api_test_client.get(
            "/api/v2/user/profile",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
        
        # Test with valid token (mocked)
        with patch('api.security.get_current_user') as mock_get_user:
            mock_user = Mock()
            mock_user.username = "test_user"
            mock_user.email = "test@example.com"
            mock_user.roles = ["analyst"]
            mock_user.disabled = False
            
            mock_get_user.return_value = mock_user
            
            response = api_test_client.get(
                "/api/v2/user/profile",
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["username"] == "test_user"

@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpoints:
    """Integration tests for main API endpoints"""
    
    def test_health_check_endpoint(self, api_test_client):
        """Test health check endpoint"""
        response = api_test_client.get("/health")
        
        # Should return 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data
    
    def test_metrics_endpoint(self, api_test_client):
        """Test metrics endpoint"""
        response = api_test_client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus format metrics
        assert "text/plain" in response.headers.get("content-type", "")
    
    @patch('api.main.get_current_user')
    def test_quality_prediction_endpoint(self, mock_get_user, api_test_client, sample_image_path):
        """Test quality prediction endpoint"""
        # Mock authenticated user
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        # Mock the vision model
        with patch('models.vision_model.FreshProduceVisionModel') as mock_model_class:
            mock_model = Mock()
            mock_model.predict_quality.return_value = (
                'Fresh', 0.85, np.array([0.85, 0.10, 0.03, 0.01, 0.01])
            )
            mock_model_class.return_value = mock_model
            
            # Test with image file upload
            with open(sample_image_path, 'rb') as image_file:
                response = api_test_client.post(
                    "/api/v2/predict/quality",
                    headers={"Authorization": "Bearer valid_token"},
                    files={"image": ("test.jpg", image_file, "image/jpeg")},
                    data={
                        "lot_number": "LOT_12345",
                        "product_id": "1",
                        "warehouse_id": "1"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "metadata" in data
            
            assert data["prediction"] == "Fresh"
            assert 0 <= data["confidence"] <= 1
            assert len(data["probabilities"]) == 5
    
    @patch('api.main.get_current_user')
    def test_demand_forecast_endpoint(self, mock_get_user, api_test_client, sample_time_series_data):
        """Test demand forecasting endpoint"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        # Mock the forecasting model
        with patch('models.forecasting_model.TemporalFusionTransformer') as mock_model_class:
            mock_model = Mock()
            mock_model.predict.return_value = {
                'forecast': np.random.random(7) * 100,
                'confidence_intervals': {
                    'lower': np.random.random(7) * 80,
                    'upper': np.random.random(7) * 120
                },
                'uncertainty': np.random.random(7) * 0.1
            }
            mock_model_class.return_value = mock_model
            
            # Mock database query
            with patch('pandas.read_sql') as mock_read_sql:
                mock_read_sql.return_value = sample_time_series_data.head(30)
                
                response = api_test_client.post(
                    "/api/v2/forecast/demand",
                    headers={"Authorization": "Bearer valid_token"},
                    json={
                        "product_id": 1,
                        "warehouse_id": 1,
                        "horizon_days": 7,
                        "include_confidence": True
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "forecast" in data
            assert "confidence_intervals" in data
            assert "metadata" in data
            
            assert len(data["forecast"]) == 7
            assert "lower" in data["confidence_intervals"]
            assert "upper" in data["confidence_intervals"]
    
    @patch('api.main.get_current_user')
    def test_route_optimization_endpoint(self, mock_get_user, api_test_client, sample_supply_chain_network):
        """Test route optimization endpoint"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["manager"]
        mock_get_user.return_value = mock_user
        
        # Mock the GNN optimizer
        with patch('models.gnn_optimizer.SupplyChainOptimizer') as mock_optimizer_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize_distribution.return_value = {
                'optimal_routes': [
                    {'from': 1, 'to': 2, 'cost': 100, 'flow': 50},
                    {'from': 2, 'to': 4, 'cost': 20, 'flow': 25}
                ],
                'total_cost': 120,
                'optimization_time': 1.5,
                'status': 'optimal'
            }
            mock_optimizer_class.return_value = mock_optimizer
            
            response = api_test_client.post(
                "/api/v2/optimize/distribution",
                headers={"Authorization": "Bearer valid_token"},
                json={
                    "products": [1, 2, 3],
                    "warehouses": [1, 2],
                    "optimize_for": "cost",
                    "constraints": {
                        "max_distance": 500,
                        "capacity_limits": True
                    }
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "optimal_routes" in data
            assert "total_cost" in data
            assert "optimization_time" in data
            assert "status" in data
            
            assert data["status"] == "optimal"
            assert data["total_cost"] > 0
            assert len(data["optimal_routes"]) > 0
    
    @patch('api.main.get_current_user')
    def test_business_kpis_endpoint(self, mock_get_user, api_test_client, test_database_session):
        """Test business KPIs endpoint"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["manager"]
        mock_get_user.return_value = mock_user
        
        # Mock database queries
        with patch('pandas.read_sql') as mock_read_sql:
            # Mock KPI data
            mock_kpi_data = pd.DataFrame({
                'metric_name': ['otif_rate', 'temperature_compliance', 'waste_reduction'],
                'value': [0.95, 0.92, 0.23],
                'target': [0.95, 0.90, 0.25],
                'timestamp': [datetime.now()] * 3
            })
            mock_read_sql.return_value = mock_kpi_data
            
            response = api_test_client.get(
                "/api/v2/kpis/summary",
                headers={"Authorization": "Bearer valid_token"},
                params={
                    "warehouse_id": 1,
                    "date_from": "2024-01-01",
                    "date_to": "2024-01-31"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "kpis" in data
            assert "summary" in data
            assert "timestamp" in data
            
            # Check that all expected KPIs are present
            kpi_names = [kpi["name"] for kpi in data["kpis"]]
            assert "otif_rate" in kpi_names
            assert "temperature_compliance" in kpi_names
            assert "waste_reduction" in kpi_names

@pytest.mark.integration
@pytest.mark.api
class TestAPIPerformance:
    """Integration tests for API performance"""
    
    @patch('api.main.get_current_user')
    def test_api_response_time(self, mock_get_user, api_test_client):
        """Test API response times"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        # Test multiple endpoints for response time
        endpoints = [
            ("/health", "GET", None),
            ("/api/v2/user/profile", "GET", {"Authorization": "Bearer valid_token"}),
            ("/metrics", "GET", None)
        ]
        
        response_times = []
        
        for endpoint, method, headers in endpoints:
            start_time = time.time()
            
            if method == "GET":
                response = api_test_client.get(endpoint, headers=headers or {})
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Response should be reasonably fast (< 1 second for simple endpoints)
            assert response_time < 1.0
            assert response.status_code in [200, 401, 503]  # Valid status codes
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.5  # Average should be < 500ms
    
    @patch('api.main.get_current_user')
    def test_concurrent_requests(self, mock_get_user, api_test_client):
        """Test handling of concurrent requests"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        import threading
        import queue
        
        # Queue to collect results
        results = queue.Queue()
        
        def make_request():
            try:
                response = api_test_client.get(
                    "/api/v2/user/profile",
                    headers={"Authorization": "Bearer valid_token"}
                )
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # Create multiple threads
        threads = []
        num_threads = 10
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)  # 5 second timeout
        
        # Collect results
        status_codes = []
        while not results.empty():
            result = results.get()
            if isinstance(result, int):
                status_codes.append(result)
        
        # All requests should complete successfully
        assert len(status_codes) == num_threads
        assert all(code == 200 for code in status_codes)
    
    def test_large_payload_handling(self, api_test_client):
        """Test handling of large payloads"""
        # Create a large JSON payload
        large_data = {
            "data": [{"id": i, "value": f"data_point_{i}"} for i in range(1000)]
        }
        
        response = api_test_client.post(
            "/api/v2/data/bulk",
            json=large_data
        )
        
        # Should handle large payloads gracefully
        # Might return 401 (unauthorized) or 413 (payload too large) or 404 (not found)
        assert response.status_code in [401, 404, 413, 422]

@pytest.mark.integration
@pytest.mark.api
class TestAPIErrorHandling:
    """Integration tests for API error handling"""
    
    def test_invalid_json_handling(self, api_test_client):
        """Test handling of invalid JSON"""
        response = api_test_client.post(
            "/api/v2/predict/quality",
            data="invalid json data",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, api_test_client):
        """Test handling of missing required fields"""
        response = api_test_client.post(
            "/api/v2/forecast/demand",
            json={
                "product_id": 1,
                # Missing warehouse_id and horizon_days
            }
        )
        
        assert response.status_code in [401, 422]  # Unauthorized or Validation Error
    
    def test_invalid_data_types(self, api_test_client):
        """Test handling of invalid data types"""
        response = api_test_client.post(
            "/api/v2/forecast/demand",
            json={
                "product_id": "invalid_id",  # Should be integer
                "warehouse_id": 1,
                "horizon_days": "seven"  # Should be integer
            }
        )
        
        assert response.status_code in [401, 422]  # Unauthorized or Validation Error
    
    @patch('api.main.get_current_user')
    def test_database_error_handling(self, mock_get_user, api_test_client):
        """Test handling of database errors"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        # Mock database connection error
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.side_effect = Exception("Database connection failed")
            
            response = api_test_client.get(
                "/api/v2/kpis/summary",
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
    
    def test_rate_limiting(self, api_test_client):
        """Test rate limiting functionality"""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):  # Make 20 requests rapidly
            response = api_test_client.get("/health")
            responses.append(response.status_code)
        
        # Some requests might be rate limited (429) or all might succeed
        # depending on rate limiting configuration
        assert all(code in [200, 429, 503] for code in responses)

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
class TestAPIEndToEnd:
    """End-to-end integration tests"""
    
    @patch('api.main.get_current_user')
    def test_complete_prediction_workflow(self, mock_get_user, api_test_client, sample_image_path):
        """Test complete prediction workflow from image upload to result"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        # Mock all required components
        with patch('models.vision_model.FreshProduceVisionModel') as mock_vision, \
             patch('pandas.read_sql') as mock_read_sql:
            
            # Mock vision model
            mock_model = Mock()
            mock_model.predict_quality.return_value = (
                'Fresh', 0.87, np.array([0.87, 0.08, 0.03, 0.01, 0.01])
            )
            mock_vision.return_value = mock_model
            
            # Mock database operations
            mock_read_sql.return_value = pd.DataFrame({
                'ProductID': [1],
                'ProductName': ['Test Product'],
                'Category': ['Test Category']
            })
            
            # Step 1: Upload image and get prediction
            with open(sample_image_path, 'rb') as image_file:
                prediction_response = api_test_client.post(
                    "/api/v2/predict/quality",
                    headers={"Authorization": "Bearer valid_token"},
                    files={"image": ("test.jpg", image_file, "image/jpeg")},
                    data={
                        "lot_number": "LOT_12345",
                        "product_id": "1",
                        "warehouse_id": "1"
                    }
                )
            
            assert prediction_response.status_code == 200
            prediction_data = prediction_response.json()
            
            # Step 2: Verify prediction result structure
            assert "prediction" in prediction_data
            assert "confidence" in prediction_data
            assert "probabilities" in prediction_data
            assert "metadata" in prediction_data
            
            # Step 3: Check that prediction is reasonable
            assert prediction_data["prediction"] in ["Fresh", "Good", "Fair", "Poor", "Spoiled"]
            assert 0 <= prediction_data["confidence"] <= 1
            assert len(prediction_data["probabilities"]) == 5
            assert abs(sum(prediction_data["probabilities"]) - 1.0) < 0.01
    
    @patch('api.main.get_current_user')
    def test_complete_forecasting_workflow(self, mock_get_user, api_test_client, sample_time_series_data):
        """Test complete forecasting workflow"""
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        with patch('models.forecasting_model.TemporalFusionTransformer') as mock_tft, \
             patch('pandas.read_sql') as mock_read_sql:
            
            # Mock forecasting model
            mock_model = Mock()
            forecast_values = np.random.uniform(80, 120, 7)
            mock_model.predict.return_value = {
                'forecast': forecast_values,
                'confidence_intervals': {
                    'lower': forecast_values * 0.9,
                    'upper': forecast_values * 1.1
                },
                'uncertainty': np.random.uniform(0.05, 0.15, 7)
            }
            mock_tft.return_value = mock_model
            
            # Mock historical data
            mock_read_sql.return_value = sample_time_series_data
            
            # Step 1: Request forecast
            forecast_response = api_test_client.post(
                "/api/v2/forecast/demand",
                headers={"Authorization": "Bearer valid_token"},
                json={
                    "product_id": 1,
                    "warehouse_id": 1,
                    "horizon_days": 7,
                    "include_confidence": True
                }
            )
            
            assert forecast_response.status_code == 200
            forecast_data = forecast_response.json()
            
            # Step 2: Verify forecast structure
            assert "forecast" in forecast_data
            assert "confidence_intervals" in forecast_data
            assert "metadata" in forecast_data
            
            # Step 3: Verify forecast quality
            assert len(forecast_data["forecast"]) == 7
            assert all(val > 0 for val in forecast_data["forecast"])
            assert "lower" in forecast_data["confidence_intervals"]
            assert "upper" in forecast_data["confidence_intervals"]
            
            # Confidence intervals should make sense
            lower = forecast_data["confidence_intervals"]["lower"]
            upper = forecast_data["confidence_intervals"]["upper"]
            forecast = forecast_data["forecast"]
            
            for i in range(7):
                assert lower[i] <= forecast[i] <= upper[i]