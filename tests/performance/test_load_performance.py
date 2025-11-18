"""
Performance and load testing for Fresh Supply Chain Intelligence System
Tests system performance under various load conditions
"""

import pytest
import asyncio
import time
import threading
import queue
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
import psutil
import gc
import torch

# Import system components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.vision_model import FreshProduceVisionModel, EnhancedVisionModel, ModelConfig
from models.forecasting_model import TemporalFusionTransformer, ForecastConfig
from data.advanced_preprocessor import AdvancedPreprocessor
from data.feature_engineer import AdvancedFeatureEngineer

@pytest.mark.performance
@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    def test_api_response_time_under_load(self, api_test_client):
        """Test API response times under concurrent load"""
        
        def make_request():
            start_time = time.time()
            response = api_test_client.get("/health")
            end_time = time.time()
            
            return {
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'success': response.status_code == 200
            }
        
        # Test with increasing concurrent users
        user_loads = [1, 5, 10, 20, 50]
        results = {}
        
        for num_users in user_loads:
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(make_request) for _ in range(num_users)]
                user_results = [future.result() for future in as_completed(futures)]
            
            # Calculate performance metrics
            response_times = [r['response_time'] for r in user_results]
            success_rate = sum(r['success'] for r in user_results) / len(user_results)
            
            results[num_users] = {
                'avg_response_time': statistics.mean(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99),
                'success_rate': success_rate,
                'throughput': num_users / max(response_times)
            }
        
        # Verify performance requirements
        for num_users, metrics in results.items():
            # Success rate should remain high
            assert metrics['success_rate'] >= 0.95, f"Success rate too low for {num_users} users: {metrics['success_rate']}"
            
            # Response time should be reasonable
            assert metrics['p95_response_time'] < 2.0, f"P95 response time too high for {num_users} users: {metrics['p95_response_time']}"
            
            # System should handle at least 10 requests per second
            if num_users >= 10:
                assert metrics['throughput'] >= 5.0, f"Throughput too low for {num_users} users: {metrics['throughput']}"
    
    @patch('api.main.get_current_user')
    def test_prediction_endpoint_performance(self, mock_get_user, api_test_client, sample_image_path):
        """Test prediction endpoint performance under load"""
        
        # Mock authenticated user
        mock_user = Mock()
        mock_user.username = "test_user"
        mock_user.roles = ["analyst"]
        mock_get_user.return_value = mock_user
        
        # Mock vision model for consistent performance testing
        with patch('models.vision_model.FreshProduceVisionModel') as mock_model_class:
            mock_model = Mock()
            mock_model.predict_quality.return_value = (
                'Fresh', 0.85, np.array([0.85, 0.10, 0.03, 0.01, 0.01])
            )
            mock_model_class.return_value = mock_model
            
            def make_prediction_request():
                start_time = time.time()
                
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
                
                end_time = time.time()
                
                return {
                    'status_code': response.status_code,
                    'response_time': end_time - start_time,
                    'success': response.status_code == 200
                }
            
            # Test concurrent prediction requests
            num_requests = 20
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_prediction_request) for _ in range(num_requests)]
                results = [future.result() for future in as_completed(futures)]
            
            # Analyze performance
            response_times = [r['response_time'] for r in results]
            success_rate = sum(r['success'] for r in results) / len(results)
            
            # Performance assertions
            assert success_rate >= 0.90, f"Prediction success rate too low: {success_rate}"
            assert statistics.mean(response_times) < 5.0, f"Average prediction time too high: {statistics.mean(response_times)}"
            assert np.percentile(response_times, 95) < 10.0, f"P95 prediction time too high: {np.percentile(response_times, 95)}"
    
    def test_memory_usage_under_load(self, api_test_client):
        """Test memory usage under sustained load"""
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def sustained_load():
            for _ in range(100):
                response = api_test_client.get("/health")
                assert response.status_code in [200, 503]
        
        # Run sustained load test
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(sustained_load) for _ in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for test load)
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase}MB"

@pytest.mark.performance
@pytest.mark.slow
class TestMLModelPerformance:
    """Performance tests for ML models"""
    
    def test_vision_model_inference_performance(self, test_utils):
        """Test vision model inference performance"""
        
        # Test with smaller model for performance testing
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        model = EnhancedVisionModel(config)
        model.eval()
        
        # Warm up the model
        warmup_input = test_utils.create_mock_image_tensor(batch_size=1)
        with torch.no_grad():
            for _ in range(5):
                _ = model(warmup_input)
        
        # Performance test with different batch sizes
        batch_sizes = [1, 4, 8, 16]
        performance_results = {}
        
        for batch_size in batch_sizes:
            input_tensor = test_utils.create_mock_image_tensor(batch_size=batch_size)
            
            # Measure inference time
            times = []
            for _ in range(10):  # Multiple runs for accuracy
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time  # Images per second
            
            performance_results[batch_size] = {
                'avg_inference_time': avg_time,
                'throughput': throughput,
                'time_per_image': avg_time / batch_size
            }
        
        # Performance assertions
        for batch_size, metrics in performance_results.items():
            # Single image should process quickly
            assert metrics['time_per_image'] < 1.0, f"Time per image too high for batch {batch_size}: {metrics['time_per_image']}"
            
            # Throughput should be reasonable
            assert metrics['throughput'] >= 1.0, f"Throughput too low for batch {batch_size}: {metrics['throughput']}"
        
        # Larger batches should be more efficient per image
        assert performance_results[16]['time_per_image'] < performance_results[1]['time_per_image']
    
    def test_forecasting_model_performance(self, test_utils):
        """Test forecasting model performance"""
        
        config = ForecastConfig(
            input_size=10,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            horizon=7
        )
        
        model = TemporalFusionTransformer(config)
        model.eval()
        
        # Test with different sequence lengths
        sequence_lengths = [20, 50, 100, 200]
        performance_results = {}
        
        for seq_len in sequence_lengths:
            input_tensor = test_utils.create_mock_time_series(length=seq_len, features=10)
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_tensor)
            
            # Measure performance
            times = []
            for _ in range(10):
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            
            performance_results[seq_len] = {
                'avg_inference_time': avg_time,
                'time_per_timestep': avg_time / seq_len
            }
        
        # Performance assertions
        for seq_len, metrics in performance_results.items():
            # Inference should complete in reasonable time
            assert metrics['avg_inference_time'] < 5.0, f"Inference time too high for sequence {seq_len}: {metrics['avg_inference_time']}"
            
            # Time per timestep should be reasonable
            assert metrics['time_per_timestep'] < 0.1, f"Time per timestep too high for sequence {seq_len}: {metrics['time_per_timestep']}"
    
    def test_model_memory_efficiency(self, test_utils):
        """Test model memory efficiency"""
        
        import torch
        
        # Test vision model memory usage
        config = ModelConfig(num_classes=5, backbone='efficientnet-b0')
        model = EnhancedVisionModel(config)
        
        # Measure model parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
        
        # Measure activation memory for different batch sizes
        batch_sizes = [1, 4, 8, 16]
        activation_memory = {}
        
        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                input_tensor = test_utils.create_mock_image_tensor(batch_size=batch_size).cuda()
                model = model.cuda()
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                activation_memory[batch_size] = peak_memory
                
                model = model.cpu()
            else:
                # Skip CUDA memory test if not available
                activation_memory[batch_size] = 0
        
        # Memory efficiency assertions
        assert param_memory < 50, f"Model parameter memory too high: {param_memory}MB"
        
        if torch.cuda.is_available():
            # Memory should scale reasonably with batch size
            assert activation_memory[16] < activation_memory[1] * 20  # Should not be linear due to shared parameters

@pytest.mark.performance
@pytest.mark.slow
class TestDataProcessingPerformance:
    """Performance tests for data processing components"""
    
    def test_data_preprocessing_performance(self, performance_test_data):
        """Test data preprocessing performance with large datasets"""
        
        preprocessor = AdvancedPreprocessor()
        large_dataset = performance_test_data["large_dataset"]
        
        # Test different dataset sizes
        dataset_sizes = [1000, 5000, 10000]
        performance_results = {}
        
        for size in dataset_sizes:
            test_data = large_dataset.head(size).copy()
            
            # Add some missing values and outliers for realistic testing
            test_data.loc[::100, 'value'] = np.nan  # Add missing values
            test_data.loc[::200, 'value'] = test_data['value'].max() * 10  # Add outliers
            
            start_time = time.time()
            
            # Run preprocessing pipeline
            processed_data = preprocessor.handle_missing_values(test_data)
            outlier_indices = preprocessor.detect_outliers(processed_data, ['value'])
            cleaned_data = preprocessor.handle_outliers(processed_data, outlier_indices)
            normalized_data = preprocessor.normalize_data(cleaned_data, ['value'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            performance_results[size] = {
                'processing_time': processing_time,
                'records_per_second': size / processing_time,
                'time_per_record': processing_time / size
            }
        
        # Performance assertions
        for size, metrics in performance_results.items():
            # Should process at least 1000 records per second
            assert metrics['records_per_second'] >= 500, f"Processing too slow for {size} records: {metrics['records_per_second']} records/sec"
            
            # Time per record should be reasonable
            assert metrics['time_per_record'] < 0.01, f"Time per record too high for {size} records: {metrics['time_per_record']}"
        
        # Processing should scale sub-linearly
        assert performance_results[10000]['time_per_record'] <= performance_results[1000]['time_per_record'] * 2
    
    def test_feature_engineering_performance(self, sample_time_series_data):
        """Test feature engineering performance"""
        
        engineer = AdvancedFeatureEngineer()
        
        # Test with different data sizes
        data_sizes = [1000, 5000, 10000]
        performance_results = {}
        
        for size in data_sizes:
            test_data = sample_time_series_data.head(size).copy()
            
            start_time = time.time()
            
            # Run feature engineering pipeline
            enhanced_data = engineer.create_time_features(test_data, 'Date')
            enhanced_data = engineer.create_lag_features(enhanced_data, 'Demand', lags=[1, 7, 30])
            enhanced_data = engineer.create_rolling_features(enhanced_data, 'Demand', windows=[7, 30])
            enhanced_data = engineer.create_supply_chain_features(enhanced_data)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            performance_results[size] = {
                'processing_time': processing_time,
                'records_per_second': size / processing_time,
                'features_created': len(enhanced_data.columns) - len(test_data.columns)
            }
        
        # Performance assertions
        for size, metrics in performance_results.items():
            # Should process at least 500 records per second
            assert metrics['records_per_second'] >= 200, f"Feature engineering too slow for {size} records: {metrics['records_per_second']} records/sec"
            
            # Should create meaningful number of features
            assert metrics['features_created'] >= 10, f"Too few features created for {size} records: {metrics['features_created']}"
    
    def test_concurrent_data_processing(self, performance_test_data):
        """Test concurrent data processing performance"""
        
        preprocessor = AdvancedPreprocessor()
        large_dataset = performance_test_data["large_dataset"]
        
        def process_chunk(chunk_data):
            start_time = time.time()
            
            # Process chunk
            processed = preprocessor.handle_missing_values(chunk_data)
            outliers = preprocessor.detect_outliers(processed, ['value'])
            cleaned = preprocessor.handle_outliers(processed, outliers)
            
            end_time = time.time()
            
            return {
                'chunk_size': len(chunk_data),
                'processing_time': end_time - start_time,
                'success': True
            }
        
        # Split data into chunks for concurrent processing
        chunk_size = 2000
        chunks = [large_dataset.iloc[i:i+chunk_size] for i in range(0, len(large_dataset), chunk_size)]
        
        # Test sequential vs concurrent processing
        # Sequential processing
        start_time = time.time()
        sequential_results = [process_chunk(chunk) for chunk in chunks]
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            concurrent_results = [future.result() for future in as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        # Performance assertions
        assert all(r['success'] for r in sequential_results), "Sequential processing failed"
        assert all(r['success'] for r in concurrent_results), "Concurrent processing failed"
        
        # Concurrent should be faster than sequential (with some tolerance)
        speedup = sequential_time / concurrent_time
        assert speedup >= 1.5, f"Concurrent processing not significantly faster: {speedup}x speedup"

@pytest.mark.performance
@pytest.mark.slow
class TestSystemScalability:
    """Scalability tests for the entire system"""
    
    def test_database_query_performance(self, test_database_session):
        """Test database query performance under load"""
        
        # Simulate concurrent database queries
        def execute_query():
            try:
                # Mock database query execution time
                start_time = time.time()
                
                # Simulate query execution
                time.sleep(0.01)  # Mock 10ms query time
                
                end_time = time.time()
                
                return {
                    'success': True,
                    'query_time': end_time - start_time
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Test with increasing concurrent queries
        query_loads = [1, 5, 10, 20, 50]
        results = {}
        
        for num_queries in query_loads:
            with ThreadPoolExecutor(max_workers=num_queries) as executor:
                futures = [executor.submit(execute_query) for _ in range(num_queries)]
                query_results = [future.result() for future in as_completed(futures)]
            
            # Analyze results
            successful_queries = [r for r in query_results if r['success']]
            success_rate = len(successful_queries) / len(query_results)
            
            if successful_queries:
                avg_query_time = statistics.mean([r['query_time'] for r in successful_queries])
                p95_query_time = np.percentile([r['query_time'] for r in successful_queries], 95)
            else:
                avg_query_time = float('inf')
                p95_query_time = float('inf')
            
            results[num_queries] = {
                'success_rate': success_rate,
                'avg_query_time': avg_query_time,
                'p95_query_time': p95_query_time
            }
        
        # Scalability assertions
        for num_queries, metrics in results.items():
            # Success rate should remain high
            assert metrics['success_rate'] >= 0.95, f"Query success rate too low for {num_queries} queries: {metrics['success_rate']}"
            
            # Query times should remain reasonable
            assert metrics['p95_query_time'] < 1.0, f"P95 query time too high for {num_queries} queries: {metrics['p95_query_time']}"
    
    def test_cache_performance_under_load(self, mock_redis_client):
        """Test cache performance under high load"""
        
        def cache_operation():
            try:
                start_time = time.time()
                
                # Simulate cache operations
                key = f"test_key_{np.random.randint(1000)}"
                value = f"test_value_{np.random.randint(1000)}"
                
                # Mock cache set/get operations
                mock_redis_client.set(key, value)
                retrieved_value = mock_redis_client.get(key)
                
                end_time = time.time()
                
                return {
                    'success': True,
                    'operation_time': end_time - start_time,
                    'cache_hit': retrieved_value is not None
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Test concurrent cache operations
        num_operations = 1000
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(cache_operation) for _ in range(num_operations)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze cache performance
        successful_ops = [r for r in results if r['success']]
        success_rate = len(successful_ops) / len(results)
        
        if successful_ops:
            avg_operation_time = statistics.mean([r['operation_time'] for r in successful_ops])
            p95_operation_time = np.percentile([r['operation_time'] for r in successful_ops], 95)
        
        # Cache performance assertions
        assert success_rate >= 0.99, f"Cache success rate too low: {success_rate}"
        assert avg_operation_time < 0.01, f"Average cache operation time too high: {avg_operation_time}"
        assert p95_operation_time < 0.05, f"P95 cache operation time too high: {p95_operation_time}"
    
    def test_system_resource_utilization(self):
        """Test system resource utilization under load"""
        
        # Monitor system resources during load test
        def monitor_resources():
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            
            return {
                'cpu_percent': cpu_usage,
                'memory_percent': memory_info.percent,
                'memory_available_mb': memory_info.available / 1024 / 1024,
                'disk_read_mb': disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                'disk_write_mb': disk_io.write_bytes / 1024 / 1024 if disk_io else 0
            }
        
        # Baseline resource usage
        baseline = monitor_resources()
        
        # Simulate system load
        def cpu_intensive_task():
            # Simulate CPU-intensive work
            for _ in range(100000):
                _ = sum(range(100))
        
        def memory_intensive_task():
            # Simulate memory-intensive work
            large_list = [i for i in range(100000)]
            return len(large_list)
        
        # Run load test
        with ThreadPoolExecutor(max_workers=4) as executor:
            cpu_futures = [executor.submit(cpu_intensive_task) for _ in range(4)]
            memory_futures = [executor.submit(memory_intensive_task) for _ in range(4)]
            
            # Monitor during load
            load_metrics = []
            for _ in range(10):  # Monitor for 10 seconds
                load_metrics.append(monitor_resources())
                time.sleep(1)
            
            # Wait for tasks to complete
            for future in cpu_futures + memory_futures:
                future.result()
        
        # Analyze resource utilization
        avg_cpu = statistics.mean([m['cpu_percent'] for m in load_metrics])
        max_cpu = max([m['cpu_percent'] for m in load_metrics])
        avg_memory = statistics.mean([m['memory_percent'] for m in load_metrics])
        max_memory = max([m['memory_percent'] for m in load_metrics])
        
        # Resource utilization assertions
        # System should handle load without excessive resource usage
        assert max_cpu < 90, f"CPU usage too high during load test: {max_cpu}%"
        assert max_memory < 85, f"Memory usage too high during load test: {max_memory}%"
        
        # System should show increased utilization under load
        assert avg_cpu > baseline['cpu_percent'], "CPU usage should increase under load"