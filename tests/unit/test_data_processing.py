"""
Unit tests for data processing modules
Tests for data validation, preprocessing, feature engineering, and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Import modules to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_validator import DataValidator, ValidationRule, ValidationResult
from data.advanced_preprocessor import AdvancedPreprocessor
from data.feature_engineer import AdvancedFeatureEngineer
from data.error_handler import AdvancedErrorHandler, ErrorSeverity
from data.data_lineage import DataLineageTracker, DataSource, DataOperation

@pytest.mark.unit
class TestDataValidator:
    """Unit tests for DataValidator"""
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = DataValidator()
        assert validator.rules == []
        assert validator.validation_history == []
    
    def test_add_validation_rule(self):
        """Test adding validation rules"""
        validator = DataValidator()
        
        rule = ValidationRule(
            name="temperature_range",
            description="Temperature should be between -10 and 50",
            validation_function=lambda df: df['temperature'].between(-10, 50).all(),
            severity="error"
        )
        
        validator.add_rule(rule)
        assert len(validator.rules) == 1
        assert validator.rules[0].name == "temperature_range"
    
    def test_temperature_validation(self):
        """Test temperature validation rule"""
        validator = DataValidator()
        
        # Valid data
        valid_data = pd.DataFrame({
            'temperature': [2.5, 3.0, 4.5, 2.0, 5.0],
            'humidity': [85, 87, 82, 90, 88]
        })
        
        # Invalid data
        invalid_data = pd.DataFrame({
            'temperature': [2.5, 15.0, 4.5, -5.0, 5.0],  # 15.0 and -5.0 are out of range
            'humidity': [85, 87, 82, 90, 88]
        })
        
        rule = ValidationRule(
            name="temperature_range",
            description="Temperature should be between 0 and 10",
            validation_function=lambda df: df['temperature'].between(0, 10).all(),
            severity="error"
        )
        
        validator.add_rule(rule)
        
        # Test valid data
        result_valid = validator.validate_dataframe(valid_data)
        assert result_valid.is_valid
        assert len(result_valid.errors) == 0
        
        # Test invalid data
        result_invalid = validator.validate_dataframe(invalid_data)
        assert not result_invalid.is_valid
        assert len(result_invalid.errors) > 0
    
    def test_missing_values_validation(self):
        """Test missing values validation"""
        validator = DataValidator()
        
        data_with_nulls = pd.DataFrame({
            'temperature': [2.5, None, 4.5, 2.0, 5.0],
            'humidity': [85, 87, None, 90, 88]
        })
        
        rule = ValidationRule(
            name="no_missing_values",
            description="No missing values allowed",
            validation_function=lambda df: not df.isnull().any().any(),
            severity="warning"
        )
        
        validator.add_rule(rule)
        result = validator.validate_dataframe(data_with_nulls)
        
        assert not result.is_valid
        assert len(result.warnings) > 0
    
    def test_data_types_validation(self):
        """Test data types validation"""
        validator = DataValidator()
        
        data = pd.DataFrame({
            'temperature': [2.5, 3.0, 4.5, 2.0, 5.0],
            'product_id': [1, 2, 3, 4, 5],
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })
        
        rule = ValidationRule(
            name="correct_data_types",
            description="Check data types",
            validation_function=lambda df: (
                df['temperature'].dtype in ['float64', 'float32'] and
                df['product_id'].dtype in ['int64', 'int32']
            ),
            severity="error"
        )
        
        validator.add_rule(rule)
        result = validator.validate_dataframe(data)
        
        assert result.is_valid

@pytest.mark.unit
class TestAdvancedPreprocessor:
    """Unit tests for AdvancedPreprocessor"""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = AdvancedPreprocessor()
        assert preprocessor.imputation_strategy == 'knn'
        assert preprocessor.outlier_method == 'iqr'
        assert preprocessor.scaling_method == 'standard'
    
    def test_missing_value_imputation(self):
        """Test missing value imputation"""
        preprocessor = AdvancedPreprocessor(imputation_strategy='mean')
        
        data = pd.DataFrame({
            'temperature': [2.5, np.nan, 4.5, 2.0, 5.0],
            'humidity': [85, 87, np.nan, 90, 88]
        })
        
        processed_data = preprocessor.handle_missing_values(data)
        
        assert not processed_data.isnull().any().any()
        assert len(processed_data) == len(data)
    
    def test_outlier_detection_iqr(self):
        """Test outlier detection using IQR method"""
        preprocessor = AdvancedPreprocessor(outlier_method='iqr')
        
        # Create data with obvious outliers
        normal_data = np.random.normal(4, 1, 100)
        outliers = [20, -10, 25]  # Clear outliers
        data = pd.DataFrame({
            'temperature': np.concatenate([normal_data, outliers])
        })
        
        outlier_indices = preprocessor.detect_outliers(data, ['temperature'])
        
        assert len(outlier_indices) > 0
        # The last 3 indices should be detected as outliers
        assert any(idx >= 100 for idx in outlier_indices)
    
    def test_outlier_detection_zscore(self):
        """Test outlier detection using Z-score method"""
        preprocessor = AdvancedPreprocessor(outlier_method='zscore')
        
        # Create data with outliers
        normal_data = np.random.normal(4, 1, 100)
        outliers = [15, -8]  # Clear outliers (>3 standard deviations)
        data = pd.DataFrame({
            'temperature': np.concatenate([normal_data, outliers])
        })
        
        outlier_indices = preprocessor.detect_outliers(data, ['temperature'])
        
        assert len(outlier_indices) > 0
    
    def test_data_normalization(self):
        """Test data normalization"""
        preprocessor = AdvancedPreprocessor(scaling_method='standard')
        
        data = pd.DataFrame({
            'temperature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'humidity': [70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
        })
        
        normalized_data = preprocessor.normalize_data(data, ['temperature', 'humidity'])
        
        # Check that mean is approximately 0 and std is approximately 1
        assert abs(normalized_data['temperature'].mean()) < 0.1
        assert abs(normalized_data['temperature'].std() - 1.0) < 0.1
        assert abs(normalized_data['humidity'].mean()) < 0.1
        assert abs(normalized_data['humidity'].std() - 1.0) < 0.1
    
    def test_text_normalization(self):
        """Test text normalization"""
        preprocessor = AdvancedPreprocessor()
        
        data = pd.DataFrame({
            'product_name': ['  Fresh SALMON  ', 'organic apples', 'LEAFY greens', None],
            'category': ['FISH', 'fruit', 'Vegetable', 'dairy']
        })
        
        normalized_data = preprocessor.normalize_text_data(data, ['product_name', 'category'])
        
        assert normalized_data['product_name'].iloc[0] == 'fresh salmon'
        assert normalized_data['category'].iloc[0] == 'fish'
        assert normalized_data['category'].iloc[1] == 'fruit'
        assert pd.isna(normalized_data['product_name'].iloc[3])  # Should remain NaN
    
    def test_data_type_optimization(self):
        """Test data type optimization"""
        preprocessor = AdvancedPreprocessor()
        
        data = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],  # Can be int8
            'large_int': [1000000, 2000000, 3000000],  # Needs int32
            'float_data': [1.1, 2.2, 3.3, 4.4, 5.5],
            'category_data': ['A', 'B', 'A', 'C', 'B']
        })
        
        optimized_data = preprocessor.optimize_data_types(data)
        
        # Check that small integers are optimized
        assert optimized_data['small_int'].dtype in ['int8', 'int16']
        # Check that categorical data is converted
        assert optimized_data['category_data'].dtype.name == 'category'

@pytest.mark.unit
class TestAdvancedFeatureEngineer:
    """Unit tests for FeatureEngineer"""
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization"""
        engineer = AdvancedFeatureEngineer()
        assert engineer.feature_history == []
    
    def test_time_based_features(self):
        """Test time-based feature creation"""
        engineer = AdvancedFeatureEngineer()
        
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'value': range(10)
        })
        
        enhanced_data = engineer.create_time_features(data, 'timestamp')
        
        expected_features = ['hour', 'day', 'month', 'year', 'dayofweek', 'quarter', 'is_weekend']
        for feature in expected_features:
            assert feature in enhanced_data.columns
        
        # Check weekend detection
        assert enhanced_data['is_weekend'].sum() > 0  # Should have some weekends
    
    def test_lag_features(self):
        """Test lag feature creation"""
        engineer = AdvancedFeatureEngineer()
        
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        lagged_data = engineer.create_lag_features(data, 'value', lags=[1, 2, 3])
        
        assert 'value_lag_1' in lagged_data.columns
        assert 'value_lag_2' in lagged_data.columns
        assert 'value_lag_3' in lagged_data.columns
        
        # Check lag values
        assert lagged_data['value_lag_1'].iloc[1] == 1  # Previous value
        assert lagged_data['value_lag_2'].iloc[2] == 1  # Two periods ago
    
    def test_rolling_features(self):
        """Test rolling window feature creation"""
        engineer = AdvancedFeatureEngineer()
        
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        rolling_data = engineer.create_rolling_features(data, 'value', windows=[3, 5])
        
        assert 'value_rolling_mean_3' in rolling_data.columns
        assert 'value_rolling_std_3' in rolling_data.columns
        assert 'value_rolling_mean_5' in rolling_data.columns
        
        # Check rolling mean calculation
        assert rolling_data['value_rolling_mean_3'].iloc[4] == 3.0  # Mean of [1,2,3,4,5] for window 3
    
    def test_domain_specific_features(self):
        """Test domain-specific feature creation"""
        engineer = AdvancedFeatureEngineer()
        
        data = pd.DataFrame({
            'temperature': [2.0, 4.0, 6.0, 8.0, 3.0],
            'humidity': [85, 90, 80, 75, 88],
            'product_category': ['fish', 'fruit', 'vegetable', 'dairy', 'meat']
        })
        
        enhanced_data = engineer.create_supply_chain_features(data)
        
        # Check temperature compliance feature
        assert 'temp_compliance' in enhanced_data.columns
        assert 'freshness_risk' in enhanced_data.columns
        assert 'storage_quality_score' in enhanced_data.columns
        
        # Temperature compliance should be binary
        assert enhanced_data['temp_compliance'].dtype == bool or enhanced_data['temp_compliance'].dtype == 'int64'
    
    def test_interaction_features(self):
        """Test interaction feature creation"""
        engineer = AdvancedFeatureEngineer()
        
        data = pd.DataFrame({
            'temperature': [2.0, 4.0, 6.0, 8.0],
            'humidity': [85, 90, 80, 75]
        })
        
        interaction_data = engineer.create_interaction_features(
            data, 
            [('temperature', 'humidity')]
        )
        
        assert 'temperature_x_humidity' in interaction_data.columns
        
        # Check interaction calculation
        expected_interaction = data['temperature'] * data['humidity']
        pd.testing.assert_series_equal(
            interaction_data['temperature_x_humidity'], 
            expected_interaction, 
            check_names=False
        )

@pytest.mark.unit
class TestAdvancedErrorHandler:
    """Unit tests for AdvancedErrorHandler"""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization"""
        handler = AdvancedErrorHandler()
        assert handler.error_log == []
        assert handler.recovery_strategies == {}
    
    def test_error_recording(self):
        """Test error recording functionality"""
        handler = AdvancedErrorHandler()
        
        test_error = Exception("Test error message")
        handler.record_error(
            error=test_error,
            context={'operation': 'data_validation', 'dataset': 'temperature_logs'},
            severity=ErrorSeverity.HIGH
        )
        
        assert len(handler.error_log) == 1
        assert handler.error_log[0]['error_type'] == 'Exception'
        assert handler.error_log[0]['message'] == 'Test error message'
        assert handler.error_log[0]['severity'] == ErrorSeverity.HIGH
    
    def test_error_recovery_strategy(self):
        """Test error recovery strategy"""
        handler = AdvancedErrorHandler()
        
        # Define a recovery strategy
        def retry_strategy(error, context, max_retries=3):
            return {'action': 'retry', 'max_attempts': max_retries}
        
        handler.add_recovery_strategy('connection_error', retry_strategy)
        
        test_error = ConnectionError("Database connection failed")
        recovery = handler.get_recovery_strategy('connection_error', test_error, {})
        
        assert recovery['action'] == 'retry'
        assert recovery['max_attempts'] == 3
    
    def test_error_analysis(self):
        """Test error analysis functionality"""
        handler = AdvancedErrorHandler()
        
        # Record multiple errors
        errors = [
            (ValueError("Invalid temperature"), ErrorSeverity.MEDIUM),
            (ConnectionError("DB connection lost"), ErrorSeverity.HIGH),
            (ValueError("Invalid humidity"), ErrorSeverity.MEDIUM),
            (TimeoutError("Request timeout"), ErrorSeverity.LOW)
        ]
        
        for error, severity in errors:
            handler.record_error(error, {}, severity)
        
        analysis = handler.analyze_error_patterns()
        
        assert 'error_counts' in analysis
        assert 'severity_distribution' in analysis
        assert analysis['error_counts']['ValueError'] == 2
        assert analysis['error_counts']['ConnectionError'] == 1

@pytest.mark.unit
class TestDataLineageTracker:
    """Unit tests for DataLineageTracker"""
    
    def test_lineage_tracker_initialization(self):
        """Test lineage tracker initialization"""
        tracker = DataLineageTracker()
        assert tracker.operations == []
        assert tracker.data_sources == {}
    
    def test_data_source_registration(self):
        """Test data source registration"""
        tracker = DataLineageTracker()
        
        source = DataSource(
            source_id="temp_sensors",
            source_type="IoT",
            location="warehouse_1",
            description="Temperature sensors in warehouse 1"
        )
        
        tracker.register_data_source(source)
        
        assert "temp_sensors" in tracker.data_sources
        assert tracker.data_sources["temp_sensors"].source_type == "IoT"
    
    def test_operation_logging(self):
        """Test operation logging"""
        tracker = DataLineageTracker()
        
        operation = DataOperation(
            operation_id="validate_temp_001",
            operation_type="validation",
            input_sources=["temp_sensors"],
            output_destination="validated_temp_data",
            transformation_details={"rules": ["temperature_range", "missing_values"]},
            timestamp=datetime.now()
        )
        
        tracker.log_operation(operation)
        
        assert len(tracker.operations) == 1
        assert tracker.operations[0].operation_type == "validation"
        assert "temp_sensors" in tracker.operations[0].input_sources
    
    def test_lineage_query(self):
        """Test lineage querying"""
        tracker = DataLineageTracker()
        
        # Register data sources
        source1 = DataSource("source1", "database", "warehouse", "Raw data")
        source2 = DataSource("source2", "api", "external", "External API data")
        tracker.register_data_source(source1)
        tracker.register_data_source(source2)
        
        # Log operations
        op1 = DataOperation("op1", "extraction", ["source1"], "intermediate1", {}, datetime.now())
        op2 = DataOperation("op2", "transformation", ["intermediate1"], "final_output", {}, datetime.now())
        
        tracker.log_operation(op1)
        tracker.log_operation(op2)
        
        # Query lineage
        lineage = tracker.get_data_lineage("final_output")
        
        assert len(lineage) > 0
        assert any(op.operation_id == "op2" for op in lineage)
    
    def test_impact_analysis(self):
        """Test impact analysis"""
        tracker = DataLineageTracker()
        
        # Create a chain of operations
        operations = [
            DataOperation("op1", "extraction", ["source1"], "data1", {}, datetime.now()),
            DataOperation("op2", "validation", ["data1"], "data2", {}, datetime.now()),
            DataOperation("op3", "transformation", ["data2"], "data3", {}, datetime.now()),
            DataOperation("op4", "aggregation", ["data3"], "final", {}, datetime.now())
        ]
        
        for op in operations:
            tracker.log_operation(op)
        
        # Analyze impact of changing source1
        impact = tracker.analyze_impact("source1")
        
        assert len(impact) > 0
        # Should include all downstream operations
        operation_ids = [op.operation_id for op in impact]
        assert "op1" in operation_ids
        assert "op2" in operation_ids
        assert "op3" in operation_ids
        assert "op4" in operation_ids

@pytest.mark.unit
class TestDataProcessingIntegration:
    """Integration tests for data processing components"""
    
    def test_full_data_processing_pipeline(self):
        """Test complete data processing pipeline"""
        # Initialize components
        validator = DataValidator()
        preprocessor = AdvancedPreprocessor()
        engineer = AdvancedFeatureEngineer()
        error_handler = AdvancedErrorHandler()
        lineage_tracker = DataLineageTracker()
        
        # Create test data with issues
        raw_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'temperature': np.concatenate([
                np.random.normal(4, 1, 95),  # Normal data
                [20, -5, np.nan, 15, np.nan]  # Outliers and missing values
            ]),
            'humidity': np.random.normal(85, 5, 100),
            'product_id': np.random.randint(1, 6, 100)
        })
        
        # Step 1: Validation
        temp_rule = ValidationRule(
            name="temperature_range",
            description="Temperature should be between -10 and 15",
            validation_function=lambda df: df['temperature'].dropna().between(-10, 15).all(),
            severity="warning"
        )
        validator.add_rule(temp_rule)
        
        validation_result = validator.validate_dataframe(raw_data)
        
        # Step 2: Preprocessing
        try:
            # Handle missing values
            processed_data = preprocessor.handle_missing_values(raw_data)
            
            # Detect and handle outliers
            outlier_indices = preprocessor.detect_outliers(processed_data, ['temperature'])
            cleaned_data = preprocessor.handle_outliers(processed_data, outlier_indices)
            
            # Step 3: Feature engineering
            enhanced_data = engineer.create_time_features(cleaned_data, 'timestamp')
            enhanced_data = engineer.create_lag_features(enhanced_data, 'temperature', lags=[1, 24])
            enhanced_data = engineer.create_rolling_features(enhanced_data, 'temperature', windows=[24])
            
            # Verify pipeline success
            assert len(enhanced_data) > 0
            assert not enhanced_data['temperature'].isnull().any()
            assert 'hour' in enhanced_data.columns
            assert 'temperature_lag_1' in enhanced_data.columns
            assert 'temperature_rolling_mean_24' in enhanced_data.columns
            
        except Exception as e:
            error_handler.record_error(e, {'pipeline_step': 'preprocessing'}, ErrorSeverity.HIGH)
            raise
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline"""
        error_handler = AdvancedErrorHandler()
        
        # Simulate various errors
        errors_to_test = [
            (ValueError("Invalid data format"), {'step': 'validation'}),
            (ConnectionError("Database unavailable"), {'step': 'data_loading'}),
            (MemoryError("Insufficient memory"), {'step': 'processing'}),
            (TimeoutError("Operation timeout"), {'step': 'feature_engineering'})
        ]
        
        for error, context in errors_to_test:
            error_handler.record_error(error, context, ErrorSeverity.HIGH)
        
        # Analyze error patterns
        analysis = error_handler.analyze_error_patterns()
        
        assert len(error_handler.error_log) == 4
        assert 'error_counts' in analysis
        assert analysis['total_errors'] == 4