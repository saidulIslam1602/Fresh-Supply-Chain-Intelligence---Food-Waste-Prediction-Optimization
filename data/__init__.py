"""
Enhanced Data Processing Module for Fresh Supply Chain Intelligence System

This module provides comprehensive data processing capabilities including:
- Advanced data validation and quality checks
- Sophisticated preprocessing and feature engineering
- Real-time streaming data processing
- Data lineage tracking and governance
- Robust error handling and recovery mechanisms

The enhanced data processing pipeline transforms standard data operations into
production-ready, enterprise-grade capabilities suitable for critical supply chain applications.
"""

from .data_validator import DataValidator, ValidationResult, ValidationSeverity
from .advanced_preprocessor import AdvancedPreprocessor, PreprocessingConfig
from .stream_processor import StreamProcessor, IoTSensorReading, QualityAlert, InventoryUpdate
from .feature_engineer import AdvancedFeatureEngineer, FeatureConfig
from .data_lineage import DataLineageTracker, DataAsset, DataOperation, DataSource
from .error_handler import AdvancedErrorHandler, ErrorSeverity, ErrorCategory, RecoveryStrategy
from .data_loader import FreshSupplyDataLoader

__version__ = "2.0.0"
__author__ = "Fresh Supply Chain Intelligence Team"

# Enhanced capabilities summary
ENHANCED_FEATURES = {
    "data_validation": {
        "description": "Comprehensive data quality validation framework",
        "capabilities": [
            "Schema validation with custom rules",
            "Business logic validation for supply chain data",
            "Statistical outlier detection",
            "Temporal consistency checks for IoT data",
            "GDPR compliance validation",
            "Automated data quality scoring"
        ]
    },
    "advanced_preprocessing": {
        "description": "Sophisticated data cleaning and transformation pipeline",
        "capabilities": [
            "Intelligent missing value imputation (KNN, statistical)",
            "Advanced outlier handling with multiple strategies",
            "Automated feature engineering with domain knowledge",
            "Text preprocessing and standardization",
            "Feature selection using multiple algorithms",
            "Memory-efficient data type optimization"
        ]
    },
    "real_time_streaming": {
        "description": "Production-ready streaming data processing",
        "capabilities": [
            "Real-time IoT sensor data processing",
            "Automated quality alert generation",
            "WebSocket-based live data streaming",
            "Batch processing with configurable windows",
            "Circuit breaker pattern for resilience",
            "Real-time aggregations and metrics"
        ]
    },
    "feature_engineering": {
        "description": "Advanced feature engineering for ML models",
        "capabilities": [
            "Time series features (lags, rolling statistics, trends)",
            "Domain-specific supply chain features",
            "Interaction and polynomial features",
            "Automated feature generation and selection",
            "Cyclical encoding for temporal data",
            "Feature importance scoring and ranking"
        ]
    },
    "data_lineage": {
        "description": "Complete data governance and lineage tracking",
        "capabilities": [
            "End-to-end data lineage tracking",
            "GDPR compliance and data anonymization",
            "Audit trails for all data operations",
            "Data quality lineage analysis",
            "Automated compliance reporting",
            "Data retention policy enforcement"
        ]
    },
    "error_handling": {
        "description": "Enterprise-grade error handling and recovery",
        "capabilities": [
            "Automatic error detection and classification",
            "Multiple recovery strategies (retry, fallback, circuit breaker)",
            "Data backup and recovery mechanisms",
            "Real-time error monitoring and alerting",
            "Comprehensive error reporting and analytics",
            "Proactive system health monitoring"
        ]
    }
}

def get_enhancement_summary():
    """Get summary of all enhanced data processing capabilities"""
    return {
        "version": __version__,
        "total_enhancements": len(ENHANCED_FEATURES),
        "features": ENHANCED_FEATURES,
        "production_ready": True,
        "enterprise_grade": True,
        "compliance_features": [
            "GDPR data protection",
            "Audit trail generation", 
            "Data retention policies",
            "Automated compliance reporting"
        ],
        "performance_features": [
            "Real-time processing",
            "Memory optimization",
            "Parallel processing",
            "Caching mechanisms",
            "Circuit breaker patterns"
        ],
        "reliability_features": [
            "Comprehensive error handling",
            "Automatic recovery mechanisms",
            "Data backup and restore",
            "System health monitoring",
            "Graceful degradation"
        ]
    }

# Convenience functions for quick setup
def create_enhanced_data_pipeline(connection_string: str = None, 
                                redis_host: str = 'localhost',
                                enable_streaming: bool = True,
                                enable_lineage: bool = True) -> dict:
    """
    Create a complete enhanced data processing pipeline
    
    Returns:
        dict: Dictionary containing all initialized components
    """
    
    components = {}
    
    # Data validation
    components['validator'] = DataValidator(connection_string)
    
    # Advanced preprocessing
    components['preprocessor'] = AdvancedPreprocessor()
    
    # Feature engineering
    components['feature_engineer'] = AdvancedFeatureEngineer()
    
    # Error handling
    components['error_handler'] = AdvancedErrorHandler(connection_string, redis_host)
    components['error_handler'].start_error_processing()
    
    # Optional components
    if enable_streaming:
        components['stream_processor'] = StreamProcessor(connection_string, redis_host)
        components['stream_processor'].start_processing()
    
    if enable_lineage:
        components['lineage_tracker'] = DataLineageTracker(connection_string)
    
    # Original data loader
    components['data_loader'] = FreshSupplyDataLoader(connection_string)
    
    return components

def validate_and_process_data(df, 
                            target_column: str = None,
                            connection_string: str = None) -> tuple:
    """
    Complete data validation and processing pipeline
    
    Args:
        df: Input DataFrame
        target_column: Target column for supervised learning
        connection_string: Database connection string
    
    Returns:
        tuple: (processed_df, validation_results, processing_summary)
    """
    
    # Initialize components
    validator = DataValidator(connection_string)
    preprocessor = AdvancedPreprocessor()
    feature_engineer = AdvancedFeatureEngineer()
    
    # Step 1: Validate data
    validation_results = []
    
    # Detect data type and validate accordingly
    if 'Temperature' in df.columns and 'Humidity' in df.columns:
        validation_results.extend(validator.validate_iot_data(df))
    elif 'ProductCode' in df.columns and 'Category' in df.columns:
        validation_results.extend(validator.validate_usda_products(df))
    else:
        # Generic validation
        validation_results.extend(validator._validate_data_quality(df, "Generic Dataset"))
    
    # Step 2: Advanced preprocessing
    processed_df = preprocessor.fit_transform(df, target_column)
    
    # Step 3: Feature engineering
    engineered_df = feature_engineer.engineer_features(processed_df, target_column)
    
    # Generate processing summary
    processing_summary = {
        'original_shape': df.shape,
        'processed_shape': engineered_df.shape,
        'validation_results': len(validation_results),
        'validation_passed': sum(1 for r in validation_results if r.passed),
        'features_created': len(engineered_df.columns) - len(df.columns),
        'preprocessing_steps': len(preprocessor.preprocessing_log),
        'feature_importance': feature_engineer.get_feature_importance(10)
    }
    
    return engineered_df, validation_results, processing_summary

__all__ = [
    # Core classes
    'DataValidator', 'AdvancedPreprocessor', 'StreamProcessor', 
    'AdvancedFeatureEngineer', 'DataLineageTracker', 'AdvancedErrorHandler',
    'FreshSupplyDataLoader',
    
    # Configuration classes
    'PreprocessingConfig', 'FeatureConfig',
    
    # Data classes
    'ValidationResult', 'IoTSensorReading', 'QualityAlert', 'InventoryUpdate',
    'DataAsset',
    
    # Enums
    'ValidationSeverity', 'DataOperation', 'DataSource', 
    'ErrorSeverity', 'ErrorCategory', 'RecoveryStrategy',
    
    # Utility functions
    'get_enhancement_summary', 'create_enhanced_data_pipeline', 
    'validate_and_process_data',
    
    # Constants
    'ENHANCED_FEATURES'
]