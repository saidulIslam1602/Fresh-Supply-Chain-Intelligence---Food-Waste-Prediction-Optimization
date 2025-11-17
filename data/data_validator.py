"""
Advanced Data Validation Framework for Fresh Supply Chain Intelligence System
Provides comprehensive data quality checks, schema validation, and anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
import re
from sqlalchemy import create_engine
import warnings

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    affected_rows: int = 0
    details: Dict[str, Any] = None

class DataValidator:
    """Comprehensive data validation framework"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string) if connection_string else None
        self.validation_results: List[ValidationResult] = []
        
        # Define validation rules
        self.usda_schema = {
            'ProductCode': {'type': str, 'pattern': r'^USDA_\d+$', 'required': True},
            'ProductName': {'type': str, 'min_length': 3, 'max_length': 200, 'required': True},
            'Category': {'type': str, 'allowed_values': ['Fruits', 'Vegetables', 'Dairy', 'Other Fresh Produce'], 'required': True},
            'ShelfLifeDays': {'type': int, 'min_value': 1, 'max_value': 365, 'required': True},
            'OptimalTempMin': {'type': float, 'min_value': -5, 'max_value': 15, 'required': True},
            'OptimalTempMax': {'type': float, 'min_value': -5, 'max_value': 15, 'required': True},
            'OptimalHumidityMin': {'type': float, 'min_value': 0, 'max_value': 100, 'required': True},
            'OptimalHumidityMax': {'type': float, 'min_value': 0, 'max_value': 100, 'required': True},
            'UnitCost': {'type': float, 'min_value': 0, 'max_value': 100, 'required': True},
            'UnitPrice': {'type': float, 'min_value': 0, 'max_value': 200, 'required': True}
        }
        
        self.iot_schema = {
            'LogTime': {'type': datetime, 'required': True},
            'DeviceID': {'type': str, 'pattern': r'^SENSOR_[A-Z]{3}_\d{2}$', 'required': True},
            'WarehouseID': {'type': int, 'min_value': 1, 'required': True},
            'Temperature': {'type': float, 'min_value': -10, 'max_value': 25, 'required': True},
            'Humidity': {'type': float, 'min_value': 0, 'max_value': 100, 'required': True},
            'CO2Level': {'type': float, 'min_value': 300, 'max_value': 2000, 'required': True},
            'EthyleneLevel': {'type': float, 'min_value': 0, 'max_value': 1, 'required': True},
            'QualityScore': {'type': float, 'min_value': 0, 'max_value': 1, 'required': True}
        }
    
    def validate_usda_products(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Comprehensive validation for USDA product data"""
        logger.info(f"Validating USDA products dataset with {len(df)} records")
        results = []
        
        # 1. Schema validation
        results.extend(self._validate_schema(df, self.usda_schema, "USDA Products"))
        
        # 2. Business logic validation
        results.extend(self._validate_usda_business_rules(df))
        
        # 3. Data quality checks
        results.extend(self._validate_data_quality(df, "USDA Products"))
        
        # 4. Duplicate detection
        results.extend(self._detect_duplicates(df, ['ProductCode'], "USDA Products"))
        
        # 5. Outlier detection
        results.extend(self._detect_outliers(df, ['ShelfLifeDays', 'UnitCost', 'UnitPrice'], "USDA Products"))
        
        self.validation_results.extend(results)
        return results
    
    def validate_iot_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Comprehensive validation for IoT sensor data"""
        logger.info(f"Validating IoT sensor dataset with {len(df)} records")
        results = []
        
        # 1. Schema validation
        results.extend(self._validate_schema(df, self.iot_schema, "IoT Data"))
        
        # 2. Temporal consistency checks
        results.extend(self._validate_temporal_consistency(df))
        
        # 3. Sensor range validation
        results.extend(self._validate_sensor_ranges(df))
        
        # 4. Data completeness over time
        results.extend(self._validate_time_series_completeness(df))
        
        # 5. Anomaly detection
        results.extend(self._detect_sensor_anomalies(df))
        
        self.validation_results.extend(results)
        return results
    
    def validate_warehouse_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validation for warehouse data"""
        logger.info(f"Validating warehouse dataset with {len(df)} records")
        results = []
        
        # Geographic coordinate validation
        if 'LocationLat' in df.columns and 'LocationLon' in df.columns:
            invalid_coords = df[
                (df['LocationLat'] < -90) | (df['LocationLat'] > 90) |
                (df['LocationLon'] < -180) | (df['LocationLon'] > 180)
            ]
            
            if not invalid_coords.empty:
                results.append(ValidationResult(
                    check_name="Geographic Coordinates",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Invalid geographic coordinates found",
                    affected_rows=len(invalid_coords),
                    details={'invalid_warehouses': invalid_coords['WarehouseCode'].tolist()}
                ))
        
        # Capacity validation
        if 'CapacityUnits' in df.columns:
            invalid_capacity = df[df['CapacityUnits'] <= 0]
            if not invalid_capacity.empty:
                results.append(ValidationResult(
                    check_name="Warehouse Capacity",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Warehouses with invalid capacity found",
                    affected_rows=len(invalid_capacity)
                ))
        
        self.validation_results.extend(results)
        return results
    
    def _validate_schema(self, df: pd.DataFrame, schema: Dict, dataset_name: str) -> List[ValidationResult]:
        """Validate dataframe against schema definition"""
        results = []
        
        # Check required columns
        missing_cols = [col for col in schema.keys() if col not in df.columns]
        if missing_cols:
            results.append(ValidationResult(
                check_name=f"{dataset_name} - Required Columns",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required columns: {missing_cols}",
                details={'missing_columns': missing_cols}
            ))
            return results
        
        # Validate each column
        for col, rules in schema.items():
            if col not in df.columns:
                continue
                
            col_results = self._validate_column(df[col], col, rules, dataset_name)
            results.extend(col_results)
        
        return results
    
    def _validate_column(self, series: pd.Series, col_name: str, rules: Dict, dataset_name: str) -> List[ValidationResult]:
        """Validate individual column against rules"""
        results = []
        
        # Check for required values (null check)
        if rules.get('required', False):
            null_count = series.isnull().sum()
            if null_count > 0:
                results.append(ValidationResult(
                    check_name=f"{dataset_name} - {col_name} Null Check",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Found {null_count} null values in required column {col_name}",
                    affected_rows=null_count
                ))
        
        # Type validation
        expected_type = rules.get('type')
        if expected_type and not series.empty:
            if expected_type == str:
                invalid_type = ~series.astype(str).str.len().notna()
            elif expected_type == int:
                invalid_type = ~pd.to_numeric(series, errors='coerce').notna()
            elif expected_type == float:
                invalid_type = ~pd.to_numeric(series, errors='coerce').notna()
            else:
                invalid_type = pd.Series([False] * len(series))
            
            if invalid_type.any():
                results.append(ValidationResult(
                    check_name=f"{dataset_name} - {col_name} Type Check",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Invalid data types found in {col_name}",
                    affected_rows=invalid_type.sum()
                ))
        
        # Pattern validation
        pattern = rules.get('pattern')
        if pattern and expected_type == str:
            invalid_pattern = ~series.astype(str).str.match(pattern, na=False)
            if invalid_pattern.any():
                results.append(ValidationResult(
                    check_name=f"{dataset_name} - {col_name} Pattern Check",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Values not matching pattern {pattern} in {col_name}",
                    affected_rows=invalid_pattern.sum()
                ))
        
        # Range validation
        if expected_type in [int, float]:
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            min_val = rules.get('min_value')
            if min_val is not None:
                below_min = numeric_series < min_val
                if below_min.any():
                    results.append(ValidationResult(
                        check_name=f"{dataset_name} - {col_name} Min Value",
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"Values below minimum {min_val} in {col_name}",
                        affected_rows=below_min.sum()
                    ))
            
            max_val = rules.get('max_value')
            if max_val is not None:
                above_max = numeric_series > max_val
                if above_max.any():
                    results.append(ValidationResult(
                        check_name=f"{dataset_name} - {col_name} Max Value",
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"Values above maximum {max_val} in {col_name}",
                        affected_rows=above_max.sum()
                    ))
        
        # Length validation for strings
        if expected_type == str:
            min_len = rules.get('min_length')
            max_len = rules.get('max_length')
            
            if min_len is not None:
                too_short = series.astype(str).str.len() < min_len
                if too_short.any():
                    results.append(ValidationResult(
                        check_name=f"{dataset_name} - {col_name} Min Length",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Values shorter than {min_len} characters in {col_name}",
                        affected_rows=too_short.sum()
                    ))
            
            if max_len is not None:
                too_long = series.astype(str).str.len() > max_len
                if too_long.any():
                    results.append(ValidationResult(
                        check_name=f"{dataset_name} - {col_name} Max Length",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Values longer than {max_len} characters in {col_name}",
                        affected_rows=too_long.sum()
                    ))
        
        # Allowed values validation
        allowed_values = rules.get('allowed_values')
        if allowed_values:
            invalid_values = ~series.isin(allowed_values)
            if invalid_values.any():
                results.append(ValidationResult(
                    check_name=f"{dataset_name} - {col_name} Allowed Values",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Invalid values found in {col_name}. Allowed: {allowed_values}",
                    affected_rows=invalid_values.sum(),
                    details={'invalid_values': series[invalid_values].unique().tolist()}
                ))
        
        return results
    
    def _validate_usda_business_rules(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate USDA-specific business rules"""
        results = []
        
        # Temperature range consistency
        if 'OptimalTempMin' in df.columns and 'OptimalTempMax' in df.columns:
            invalid_temp_range = df['OptimalTempMin'] >= df['OptimalTempMax']
            if invalid_temp_range.any():
                results.append(ValidationResult(
                    check_name="USDA Products - Temperature Range Logic",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Products with minimum temperature >= maximum temperature",
                    affected_rows=invalid_temp_range.sum()
                ))
        
        # Humidity range consistency
        if 'OptimalHumidityMin' in df.columns and 'OptimalHumidityMax' in df.columns:
            invalid_humidity_range = df['OptimalHumidityMin'] >= df['OptimalHumidityMax']
            if invalid_humidity_range.any():
                results.append(ValidationResult(
                    check_name="USDA Products - Humidity Range Logic",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Products with minimum humidity >= maximum humidity",
                    affected_rows=invalid_humidity_range.sum()
                ))
        
        # Price consistency
        if 'UnitCost' in df.columns and 'UnitPrice' in df.columns:
            negative_margin = df['UnitPrice'] <= df['UnitCost']
            if negative_margin.any():
                results.append(ValidationResult(
                    check_name="USDA Products - Price Logic",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message="Products with selling price <= cost price",
                    affected_rows=negative_margin.sum()
                ))
        
        # Category-specific shelf life validation
        if 'Category' in df.columns and 'ShelfLifeDays' in df.columns:
            category_shelf_life_rules = {
                'Dairy': (1, 30),
                'Fruits': (1, 21),
                'Vegetables': (1, 14),
                'Other Fresh Produce': (1, 30)
            }
            
            for category, (min_shelf, max_shelf) in category_shelf_life_rules.items():
                category_data = df[df['Category'] == category]
                if not category_data.empty:
                    invalid_shelf_life = (
                        (category_data['ShelfLifeDays'] < min_shelf) |
                        (category_data['ShelfLifeDays'] > max_shelf)
                    )
                    if invalid_shelf_life.any():
                        results.append(ValidationResult(
                            check_name=f"USDA Products - {category} Shelf Life",
                            severity=ValidationSeverity.WARNING,
                            passed=False,
                            message=f"{category} products with unusual shelf life (expected {min_shelf}-{max_shelf} days)",
                            affected_rows=invalid_shelf_life.sum()
                        ))
        
        return results
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate temporal consistency in IoT data"""
        results = []
        
        if 'LogTime' not in df.columns:
            return results
        
        # Convert to datetime if not already
        df['LogTime'] = pd.to_datetime(df['LogTime'])
        
        # Check for future timestamps
        future_timestamps = df['LogTime'] > datetime.now()
        if future_timestamps.any():
            results.append(ValidationResult(
                check_name="IoT Data - Future Timestamps",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message="Found timestamps in the future",
                affected_rows=future_timestamps.sum()
            ))
        
        # Check for very old timestamps (more than 1 year)
        very_old = df['LogTime'] < (datetime.now() - timedelta(days=365))
        if very_old.any():
            results.append(ValidationResult(
                check_name="IoT Data - Very Old Timestamps",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="Found timestamps older than 1 year",
                affected_rows=very_old.sum()
            ))
        
        # Check for duplicate timestamps per device
        if 'DeviceID' in df.columns:
            duplicates = df.duplicated(subset=['DeviceID', 'LogTime'])
            if duplicates.any():
                results.append(ValidationResult(
                    check_name="IoT Data - Duplicate Timestamps",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Found duplicate timestamps for same device",
                    affected_rows=duplicates.sum()
                ))
        
        return results
    
    def _validate_sensor_ranges(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate sensor readings are within realistic ranges"""
        results = []
        
        # Temperature validation for cold storage
        if 'Temperature' in df.columns:
            # Extreme temperatures (equipment failure)
            extreme_temp = (df['Temperature'] < -20) | (df['Temperature'] > 30)
            if extreme_temp.any():
                results.append(ValidationResult(
                    check_name="IoT Data - Extreme Temperatures",
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message="Extreme temperature readings detected (possible sensor failure)",
                    affected_rows=extreme_temp.sum()
                ))
            
            # Unusual but possible temperatures
            unusual_temp = ((df['Temperature'] < -5) | (df['Temperature'] > 15)) & ~extreme_temp
            if unusual_temp.any():
                results.append(ValidationResult(
                    check_name="IoT Data - Unusual Temperatures",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message="Unusual temperature readings for fresh produce storage",
                    affected_rows=unusual_temp.sum()
                ))
        
        # Quality score validation
        if 'QualityScore' in df.columns:
            # Check for impossible quality scores
            invalid_quality = (df['QualityScore'] < 0) | (df['QualityScore'] > 1)
            if invalid_quality.any():
                results.append(ValidationResult(
                    check_name="IoT Data - Invalid Quality Scores",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Quality scores outside valid range (0-1)",
                    affected_rows=invalid_quality.sum()
                ))
        
        return results
    
    def _validate_time_series_completeness(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Check for gaps in time series data"""
        results = []
        
        if 'LogTime' not in df.columns or 'DeviceID' not in df.columns:
            return results
        
        df['LogTime'] = pd.to_datetime(df['LogTime'])
        
        # Check each device for data gaps
        for device_id in df['DeviceID'].unique():
            device_data = df[df['DeviceID'] == device_id].sort_values('LogTime')
            
            if len(device_data) < 2:
                continue
            
            # Calculate time differences
            time_diffs = device_data['LogTime'].diff().dt.total_seconds() / 60  # in minutes
            
            # Expected interval is 30 minutes, allow up to 60 minutes
            large_gaps = time_diffs > 60
            if large_gaps.any():
                gap_count = large_gaps.sum()
                max_gap = time_diffs.max() / 60  # in hours
                
                results.append(ValidationResult(
                    check_name=f"IoT Data - Data Gaps for {device_id}",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Found {gap_count} data gaps > 1 hour (max gap: {max_gap:.1f} hours)",
                    affected_rows=gap_count,
                    details={'device_id': device_id, 'max_gap_hours': max_gap}
                ))
        
        return results
    
    def _detect_duplicates(self, df: pd.DataFrame, key_columns: List[str], dataset_name: str) -> List[ValidationResult]:
        """Detect duplicate records"""
        results = []
        
        available_columns = [col for col in key_columns if col in df.columns]
        if not available_columns:
            return results
        
        duplicates = df.duplicated(subset=available_columns)
        if duplicates.any():
            results.append(ValidationResult(
                check_name=f"{dataset_name} - Duplicate Records",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Found duplicate records based on {available_columns}",
                affected_rows=duplicates.sum()
            ))
        
        return results
    
    def _detect_outliers(self, df: pd.DataFrame, numeric_columns: List[str], dataset_name: str) -> List[ValidationResult]:
        """Detect statistical outliers using IQR method"""
        results = []
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.isnull().all():
                continue
            
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (numeric_data < lower_bound) | (numeric_data > upper_bound)
            if outliers.any():
                results.append(ValidationResult(
                    check_name=f"{dataset_name} - {col} Outliers",
                    severity=ValidationSeverity.INFO,
                    passed=False,
                    message=f"Statistical outliers detected in {col}",
                    affected_rows=outliers.sum(),
                    details={
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_percentage': (outliers.sum() / len(df)) * 100
                    }
                ))
        
        return results
    
    def _detect_sensor_anomalies(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Detect anomalies in sensor readings"""
        results = []
        
        if 'Temperature' not in df.columns or 'DeviceID' not in df.columns:
            return results
        
        # Detect sudden temperature changes (> 5°C in 30 minutes)
        df_sorted = df.sort_values(['DeviceID', 'LogTime'])
        df_sorted['temp_diff'] = df_sorted.groupby('DeviceID')['Temperature'].diff().abs()
        
        sudden_changes = df_sorted['temp_diff'] > 5
        if sudden_changes.any():
            results.append(ValidationResult(
                check_name="IoT Data - Sudden Temperature Changes",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="Sudden temperature changes detected (>5°C)",
                affected_rows=sudden_changes.sum(),
                details={'max_change': df_sorted['temp_diff'].max()}
            ))
        
        return results
    
    def _validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> List[ValidationResult]:
        """General data quality checks"""
        results = []
        
        # Check data completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_rate = ((total_cells - missing_cells) / total_cells) * 100
        
        if completeness_rate < 95:
            results.append(ValidationResult(
                check_name=f"{dataset_name} - Data Completeness",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Data completeness is {completeness_rate:.1f}% (below 95% threshold)",
                details={'completeness_rate': completeness_rate}
            ))
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1)
        if empty_rows.any():
            results.append(ValidationResult(
                check_name=f"{dataset_name} - Empty Rows",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message="Completely empty rows found",
                affected_rows=empty_rows.sum()
            ))
        
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        # Categorize results by severity
        by_severity = {}
        for severity in ValidationSeverity:
            by_severity[severity.value] = [
                r for r in self.validation_results if r.severity == severity
            ]
        
        # Calculate summary statistics
        total_checks = len(self.validation_results)
        passed_checks = len([r for r in self.validation_results if r.passed])
        failed_checks = total_checks - passed_checks
        
        # Critical issues that must be fixed
        critical_issues = by_severity.get('critical', [])
        error_issues = by_severity.get('error', [])
        
        report = {
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
                'critical_issues': len(critical_issues),
                'error_issues': len(error_issues),
                'data_quality_score': self._calculate_quality_score()
            },
            'issues_by_severity': {
                severity: [
                    {
                        'check_name': r.check_name,
                        'message': r.message,
                        'affected_rows': r.affected_rows,
                        'details': r.details
                    }
                    for r in results
                ]
                for severity, results in by_severity.items()
                if results
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if not self.validation_results:
            return 100.0
        
        # Weight different severity levels
        severity_weights = {
            ValidationSeverity.CRITICAL: -20,
            ValidationSeverity.ERROR: -10,
            ValidationSeverity.WARNING: -5,
            ValidationSeverity.INFO: -1
        }
        
        total_penalty = 0
        for result in self.validation_results:
            if not result.passed:
                penalty = severity_weights.get(result.severity, -1)
                total_penalty += penalty
        
        # Base score is 100, subtract penalties
        quality_score = max(0, 100 + total_penalty)
        return quality_score
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        critical_issues = [r for r in self.validation_results 
                          if r.severity == ValidationSeverity.CRITICAL and not r.passed]
        error_issues = [r for r in self.validation_results 
                       if r.severity == ValidationSeverity.ERROR and not r.passed]
        
        if critical_issues:
            recommendations.append(
                "CRITICAL: Fix critical data issues immediately before proceeding with analysis"
            )
        
        if error_issues:
            recommendations.append(
                "HIGH PRIORITY: Address data errors to ensure reliable model performance"
            )
        
        # Specific recommendations based on common issues
        issue_types = [r.check_name for r in self.validation_results if not r.passed]
        
        if any('Null Check' in issue for issue in issue_types):
            recommendations.append(
                "NOTE: Implement data imputation strategies for missing values"
            )
        
        if any('Outliers' in issue for issue in issue_types):
            recommendations.append(
                "DATA: Review outliers - they may indicate data quality issues or interesting patterns"
            )
        
        if any('Temperature' in issue for issue in issue_types):
            recommendations.append(
                "SENSOR: Check IoT sensor calibration and cold chain monitoring systems"
            )
        
        if any('Duplicate' in issue for issue in issue_types):
            recommendations.append(
                "PROCESS: Implement deduplication processes in data ingestion pipeline"
            )
        
        return recommendations
    
    def export_validation_report(self, filepath: str, format: str = 'json'):
        """Export validation report to file"""
        report = self.generate_validation_report()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Export detailed results to CSV
            results_data = []
            for result in self.validation_results:
                results_data.append({
                    'check_name': result.check_name,
                    'severity': result.severity.value,
                    'passed': result.passed,
                    'message': result.message,
                    'affected_rows': result.affected_rows
                })
            
            pd.DataFrame(results_data).to_csv(filepath, index=False)
        
        logger.info(f"Validation report exported to {filepath}")

# Usage example and testing functions
def validate_sample_data():
    """Example usage of the validation framework"""
    validator = DataValidator()
    
    # Sample USDA data for testing
    sample_usda = pd.DataFrame({
        'ProductCode': ['USDA_123', 'USDA_456', 'INVALID_789'],
        'ProductName': ['Fresh Apples', 'Organic Milk', ''],
        'Category': ['Fruits', 'Dairy', 'InvalidCategory'],
        'ShelfLifeDays': [14, 7, -5],
        'OptimalTempMin': [2, 1, 10],
        'OptimalTempMax': [6, 4, 5],  # Invalid: min > max
        'OptimalHumidityMin': [85, 80, 90],
        'OptimalHumidityMax': [95, 90, 85],  # Invalid: min > max
        'UnitCost': [1.5, 2.0, -1.0],  # Invalid: negative cost
        'UnitPrice': [2.5, 3.0, 0.5]   # Invalid: price < cost for last item
    })
    
    # Validate USDA data
    usda_results = validator.validate_usda_products(sample_usda)
    
    # Generate and print report
    report = validator.generate_validation_report()
    print("Data Validation Report:")
    print("=" * 50)
    print(f"Quality Score: {report['summary']['data_quality_score']:.1f}/100")
    print(f"Total Checks: {report['summary']['total_checks']}")
    print(f"Failed Checks: {report['summary']['failed_checks']}")
    
    if report.get('recommendations'):
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")

if __name__ == "__main__":
    validate_sample_data()